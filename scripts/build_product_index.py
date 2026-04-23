#!/usr/bin/env python
"""
Product Embedding Index Builder

This script builds a FAISS index for product image search.
It fetches products from the main API database, downloads images from S3,
generates CLIP embeddings, and stores them in a FAISS index.

Usage:
    cd ml_service
    python scripts/build_product_index.py
    
Environment Variables:
    DATABASE_URL - PostgreSQL connection string
    S3_BUCKET - S3 bucket name for images
    AWS_ACCESS_KEY_ID - AWS access key
    AWS_SECRET_ACCESS_KEY - AWS secret key
    AWS_REGION - AWS region
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import boto3
import requests
from PIL import Image
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductEmbeddingBuilder:
    """Build FAISS index from product images."""
    
    def __init__(self):
        self.index_dir = Path("data/product_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 client
        self.s3_client = None
        self.s3_bucket = os.getenv("S3_BUCKET", "your-bucket-name")
        
        # Database connection (from main API)
        self.database_url = os.getenv("DATABASE_URL")
        
        # CLIP model (lazy load)
        self._clip_model = None
        self._clip_preprocess = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Embedding storage
        self.embeddings: List[np.ndarray] = []
        self.product_ids: List[int] = []
        self.product_metadata: Dict[int, dict] = {}
    
    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            logger.info("Loading CLIP model...")
            import clip
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self._device)
            logger.info(f"CLIP model loaded on {self._device}")
        return self._clip_model
    
    @property
    def clip_preprocess(self):
        """Lazy load CLIP preprocess."""
        if self._clip_preprocess is None:
            import clip
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self._device)
        return self._clip_preprocess
    
    def init_s3_client(self):
        """Initialize S3 client."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        logger.info("S3 client initialized")
    
    def fetch_products_from_db(self) -> List[Dict]:
        """
        Fetch all products with images from database.
        
        Returns list of products with:
        - product_id
        - name
        - image_url
        - category_name
        - selling_price
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        if not self.database_url:
            logger.error("DATABASE_URL not set")
            return []
        
        logger.info("Fetching products from database...")
        
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT 
            p.product_id,
            p.name,
            p.image_url,
            p.selling_price,
            p.description,
            c.name as category_name
        FROM products p
        LEFT JOIN categories c ON p.category_id = c.categoryid
        WHERE p.image_url IS NOT NULL
        AND p.status = 'Active'
        ORDER BY p.product_id
        """
        
        cursor.execute(query)
        products = cursor.fetchall()
        cursor.close()
        conn.close()
        
        logger.info(f"Found {len(products)} products with images")
        return [dict(p) for p in products]
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        """Download image from URL (S3 or public URL)."""
        try:
            # Handle S3 URLs
            if image_url.startswith('s3://'):
                # Parse S3 URL
                parts = image_url[5:].split('/', 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ''
                
                if not self.s3_client:
                    self.init_s3_client()
                
                # Download to temp file
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                image_data = response['Body'].read()
                
                from io import BytesIO
                return Image.open(BytesIO(image_data)).convert('RGB')
            
            # Handle HTTP URLs (CloudFront, etc.)
            elif image_url.startswith('http'):
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                
                from io import BytesIO
                return Image.open(BytesIO(response.content)).convert('RGB')
            
            else:
                logger.warning(f"Unsupported URL format: {image_url[:50]}...")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to download image: {e}")
            return None
    
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding for an image."""
        import torch
        
        # Preprocess image
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self._device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
        return embedding.cpu().numpy().flatten()
    
    def build_index(self):
        """Build the complete product index."""
        logger.info("="*60)
        logger.info("Starting product embedding index build")
        logger.info("="*60)
        
        # Fetch products
        products = self.fetch_products_from_db()
        
        if not products:
            logger.error("No products found. Check database connection.")
            return
        
        # Process each product
        success_count = 0
        fail_count = 0
        
        for i, product in enumerate(products):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing {i+1}/{len(products)}...")
            
            try:
                # Download image
                image = self.download_image(product['image_url'])
                
                if image is None:
                    fail_count += 1
                    continue
                
                # Generate embedding
                embedding = self.generate_embedding(image)
                
                # Store
                self.embeddings.append(embedding)
                self.product_ids.append(product['product_id'])
                self.product_metadata[product['product_id']] = {
                    'product_id': product['product_id'],
                    'name': product['name'],
                    'category': product['category_name'],
                    'price': float(product['selling_price']) if product['selling_price'] else None,
                    'image_url': product['image_url']
                }
                
                success_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process product {product['product_id']}: {e}")
                fail_count += 1
        
        logger.info(f"\nProcessed: {success_count} success, {fail_count} failed")
        
        if not self.embeddings:
            logger.error("No embeddings generated. Aborting.")
            return
        
        # Save index
        self._save_index()
        
        logger.info("="*60)
        logger.info("Product index build complete!")
        logger.info(f"Total products indexed: {len(self.product_ids)}")
        logger.info(f"Index saved to: {self.index_dir}")
        logger.info("="*60)
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        import faiss
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]  # 512 for CLIP ViT-B/32
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        index.add(embeddings_array)
        
        # Save FAISS index
        index_path = self.index_dir / "product_embeddings.faiss"
        faiss.write_index(index, str(index_path))
        logger.info(f"FAISS index saved: {index_path}")
        
        # Save product ID mapping
        ids_path = self.index_dir / "product_ids.json"
        with open(ids_path, 'w') as f:
            json.dump(self.product_ids, f)
        logger.info(f"Product IDs saved: {ids_path}")
        
        # Save metadata
        metadata_path = self.index_dir / "product_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.product_metadata, f)
        logger.info(f"Metadata saved: {metadata_path}")
        
        # Save index info
        info = {
            'total_products': len(self.product_ids),
            'embedding_dimension': dimension,
            'model': 'clip-ViT-B-32',
            'index_type': 'IndexFlatIP'
        }
        info_path = self.index_dir / "index_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Index info saved: {info_path}")


def main():
    # Check environment
    required_vars = ['DATABASE_URL']
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        logger.error("Please set DATABASE_URL and optionally AWS credentials for S3")
        sys.exit(1)
    
    # Build index
    builder = ProductEmbeddingBuilder()
    builder.build_index()


if __name__ == "__main__":
    main()
