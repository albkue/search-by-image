"""ML Search Service API Endpoints."""
import asyncio
import logging
import re
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from PIL import Image

from ..config import settings
from ..pipeline.adaptive_preprocessor import AdaptivePreprocessor
from ..pipeline.brand_matcher import get_brand_matcher
from ..pipeline.embedding import CLIPEmbedding, get_clip_model_dimension
from ..pipeline.ocr_extractor import OCRExtractor
from ..pipeline.preprocessor import ImagePreprocessor, validate_image, validate_image_full
from ..pipeline.yolo_detector import YOLOPartDetector
from ..search.catalog_client import CatalogClient
from ..search.faiss_index import FAISSIndex
from ..search.merger import ResultMerger
from .schemas import (
    ImageSearchQuery,
    ImageSearchResponse,
    IndexProductResponse,
    RebuildIndexResponse,
    SearchResult as SearchResultSchema,
)


def extract_part_number(text: str) -> Optional[str]:
    """Extract part number from OCR text.
    
    Part numbers are typically alphanumeric codes like:
    - BP1234, BP-1234
    - W712/80, W712-80
    - 12345ABC
    
    Args:
        text: OCR extracted text
        
    Returns:
        Part number string if found, None otherwise
    """
    if not text:
        return None
    
    # Common part number patterns
    patterns = [
        # Pattern: 2-4 letters + numbers (e.g., BP1234, W712)
        r'\b[A-Z]{2,4}\d{2,5}[A-Z]?\b',
        # Pattern: letters + numbers + / or - + numbers (e.g., W712/80, BP-1234)
        r'\b[A-Z]{1,4}\d{2,4}[/-]\d{1,4}\b',
        # Pattern: numbers + letters (e.g., 12345ABC)
        r'\b\d{4,6}[A-Z]{1,4}\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            # Return the longest match (most likely to be complete part number)
            return max(matches, key=len)
    
    return None

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components (lazy-loaded via properties)
class Components:
    """Lazy-loaded components."""
    _preprocessor = None
    _adaptive_preprocessor = None
    _detector = None
    _ocr = None
    _embedder = None
    _faiss_index = None
    _catalog_client = None
    _merger = None
    _brand_matcher = None
    
    @property
    def brand_matcher(self):
        if self._brand_matcher is None:
            self._brand_matcher = get_brand_matcher()
        return self._brand_matcher
    
    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = ImagePreprocessor()
        return self._preprocessor
    
    @property
    def adaptive_preprocessor(self):
        if self._adaptive_preprocessor is None:
            self._adaptive_preprocessor = AdaptivePreprocessor()
        return self._adaptive_preprocessor
    
    @property
    def detector(self):
        if self._detector is None:
            self._detector = YOLOPartDetector(
                settings.YOLO_MODEL,
                settings.YOLO_CONFIDENCE_THRESHOLD,
                settings.USE_GPU
            )
        return self._detector
    
    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = OCRExtractor(
                settings.OCR_CONFIDENCE_THRESHOLD,
                settings.USE_GPU
            )
        return self._ocr
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = CLIPEmbedding(
                settings.CLIP_MODEL,
                settings.USE_GPU
            )
        return self._embedder
    
    @property
    def faiss_index(self):
        if self._faiss_index is None:
            dimension = get_clip_model_dimension(settings.CLIP_MODEL)
            self._faiss_index = FAISSIndex(
                dimension=dimension,
                index_path=settings.FAISS_INDEX_PATH
            )
            # Try to load existing index
            self._faiss_index.load_index()
        return self._faiss_index
    
    @property
    def catalog_client(self):
        if self._catalog_client is None:
            self._catalog_client = CatalogClient(settings.MAIN_API_URL)
        return self._catalog_client
    
    @property
    def merger(self):
        if self._merger is None:
            self._merger = ResultMerger()
        return self._merger


components = Components()


@router.post("/search-by-image", response_model=ImageSearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    """Search products by image using ML.
    
    This endpoint:
    1. Preprocesses the uploaded image
    2. Runs YOLOv8 detection for part type
    3. Runs OCR for brand/text extraction
    4. Generates CLIP embedding for similarity search
    5. Searches FAISS index and catalog DB
    6. Merges and ranks results
    
    Returns:
        ImageSearchResponse with query details and matching products
    """
    # 1. Full validation pipeline (magic bytes, size, EXIF strip, quality checks)
    validation = validate_image_full(await file.read())
    if not validation.valid:
        raise HTTPException(
            status_code=validation.http_status,
            detail=validation.error
        )
    image = validation.image
    validation_warning = validation.warning
    
    # 2. Preprocess image (using adaptive preprocessor)
    try:
        processed = components.adaptive_preprocessor.preprocess(image)
        analysis = components.adaptive_preprocessor.get_last_analysis()
        logger.info(f"Image analysis: brightness={analysis.get('brightness', 0):.1f}, "
                   f"contrast={analysis.get('contrast', 0):.1f}, "
                   f"dark={analysis.get('is_dark', False)}, "
                   f"bright={analysis.get('is_bright', False)}")
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(500, "Image preprocessing failed")
    
    # 3. Run detection and OCR in parallel on PROCESSED image
    # Both use the same 640x640 enhanced image for consistency
    loop = asyncio.get_event_loop()
    
    # Run YOLO detection on processed image (640x640, enhanced)
    detection_task = loop.run_in_executor(
        None,
        components.detector.detect,
        processed  # Use processed image for consistency
    )
    
    # Run OCR on same processed image (640x640, enhanced)
    ocr_task = loop.run_in_executor(
        None,
        components.ocr.extract,
        processed
    )
    
    # Wait for both to complete
    detection_result, ocr_result = await asyncio.gather(
        detection_task, ocr_task
    )
    
    # 4. Build query parameters
    part_type = detection_result.part_type if detection_result else None
    
    # Extract brand and part number from OCR text
    ocr_text = ocr_result.text if ocr_result else None
    brand_name = None
    brand_confidence = 0.0
    part_number = None
    
    if ocr_text:
        # Match brand
        brand_name, brand_confidence = components.brand_matcher.match_with_confidence(ocr_text)
        if brand_name:
            logger.info(f"Matched brand: {brand_name} (confidence: {brand_confidence:.2f}) from OCR: {ocr_text}")
        
        # Extract part number (alphanumeric code like BP1234, W712/80)
        part_number = extract_part_number(ocr_text)
        if part_number:
            logger.info(f"Extracted part number: {part_number} from OCR: {ocr_text}")
    
    confidence = 0.0
    if detection_result or ocr_result:
        confidence = max(
            detection_result.confidence if detection_result else 0,
            ocr_result.confidence if ocr_result else 0,
            brand_confidence
        )
    
    logger.info(f"Query: part_type={part_type}, brand={brand_name}, part_number={part_number}, confidence={confidence:.2f}")
    
    # 5. Generate embedding for vector search
    try:
        # If detection succeeded, crop to bbox with 10-20% padding (PDF spec)
        if detection_result and detection_result.bbox:
            logger.debug(f"Cropping to detection bbox: {detection_result.bbox}")
            processed_pil = Image.fromarray(processed)
            w, h = processed_pil.size
            x1, y1, x2, y2 = detection_result.bbox
            # Add 15% padding, clamped to image boundaries
            bw, bh = x2 - x1, y2 - y1
            pad_x = max(15, min(50, int(bw * 0.15)))
            pad_y = max(15, min(50, int(bh * 0.15)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            image_for_clip = processed_pil.crop((x1, y1, x2, y2))
            logger.debug(f"Padded bbox: [{x1},{y1},{x2},{y2}]")
        else:
            # Fallback to whole processed image if no detection
            logger.debug("No detection, using whole processed image for CLIP")
            image_for_clip = Image.fromarray(processed)
        
        # CLIP expects 224x224, embedder handles resize internally
        embedding = await loop.run_in_executor(
            None,
            components.embedder.encode_image,
            image_for_clip
        )
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(500, "Embedding generation failed")
    
    # 6. Search FAISS index
    vector_results = components.faiss_index.search(embedding, k=top_k * 2)
    
    # --- Dynamic weight calculation (PDF spec section 8) ---
    yolo_conf = detection_result.confidence if detection_result else 0.0
    ocr_conf  = ocr_result.confidence if ocr_result else 0.0
    
    # OCR confidence gate: if OCR unreliable, zero it out
    if ocr_conf < 0.5:
        alpha, beta, gamma = 0.75, 0.0, 0.25   # image-only scenario
    else:
        alpha, beta, gamma = 0.4, 0.4, 0.2     # balanced scenario
    
    # YOLO gate: if YOLO low confidence, don't filter by category
    if yolo_conf < 0.4:
        part_type = None
        logger.info('YOLO confidence low - skipping category filter')
    
    # No-match threshold
    NO_MATCH_THRESHOLD = 0.35
    if not vector_results:
        logger.warning('No vector search results found')
    elif max(r[1] for r in vector_results) < NO_MATCH_THRESHOLD:
        logger.warning(f'Low similarity scores (max: {max(r[1] for r in vector_results):.3f})')
    
    # 7. Search catalog (if we have part type, brand, or part number)
    catalog_results = []
    search_params = {"limit": top_k * 2}
    
    if part_type:
        search_params["category"] = part_type
    if brand_name:
        search_params["brand"] = brand_name
    if part_number:
        search_params["name"] = part_number  # Search part number in product name
    
    if search_params:
        try:
            catalog_results = await components.catalog_client.search_by_params(**search_params)
            logger.info(f"Catalog search found {len(catalog_results)} results with params: {search_params}")
        except Exception as e:
            logger.warning(f"Catalog search failed: {e}")
    
    # 8. Merge results with dynamic weights
    merged_results = components.merger.merge(
        catalog_results,
        vector_results,
        confidence,
        max_results=top_k,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    
    # 9. Format response
    results = [
        SearchResultSchema(
            product_id=r.product_id,
            score=round(r.score, 3),
            match_type=r.match_type
        )
        for r in merged_results
    ]
    
    # Add warning messages
    warning_message = None
    if validation_warning:
        warning_message = f'Image quality warning: {validation_warning}'
    elif not results:
        warning_message = 'No matching products found. Try a clearer photo or different angle.'
    elif results and results[0].score < NO_MATCH_THRESHOLD:
        warning_message = 'No confident match found. Please verify the product or try another photo.'
    elif results and results[0].score < 0.6:
        warning_message = 'Low confidence match. Please verify the product or try another photo.'
    
    response_data = {
        "query": ImageSearchQuery(
            part_type=part_type,
            brand_name=brand_name,
            part_number=part_number,
            confidence=round(confidence, 3)
        ),
        "results": results
    }
    
    if warning_message:
        response_data["message"] = warning_message
    
    return ImageSearchResponse(**response_data)


@router.post("/index-product", response_model=IndexProductResponse)
async def index_product(
    product_id: int = Query(..., description="Product ID to index"),
    image_url: str = Query(..., description="URL of product image")
):
    """Add a product image to the FAISS index.
    
    This endpoint downloads the product image, generates an embedding,
    and adds it to the FAISS index for future searches.
    
    Returns:
        IndexProductResponse with indexing status
    """
    import httpx
    
    # Download image
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(image_url)
            if response.status_code != 200:
                raise HTTPException(400, f"Could not download image from {image_url}")
            
            image_data = response.content
        except httpx.RequestError as e:
            logger.error(f"Error downloading image: {e}")
            raise HTTPException(400, f"Could not download image: {str(e)}")
    
    # Validate image
    image = validate_image(image_data)
    if image is None:
        raise HTTPException(400, "Invalid image file")
    
    # Generate embedding
    try:
        embedding = components.embedder.encode_image(image)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(500, "Embedding generation failed")
    
    # Add to FAISS index
    try:
        components.faiss_index.add_embeddings(
            embedding.reshape(1, -1),
            [product_id]
        )
        components.faiss_index.save_index()
    except Exception as e:
        logger.error(f"Error adding to index: {e}")
        raise HTTPException(500, "Failed to add to index")
    
    return IndexProductResponse(
        status="indexed",
        product_id=product_id
    )


@router.post("/rebuild-index", response_model=RebuildIndexResponse)
async def rebuild_index(
    background_tasks: BackgroundTasks,
    batch_size: int = Query(100, ge=10, le=500, description="Products per batch")
):
    """Rebuild the FAISS index from all product images.
    
    This is a background task that:
    1. Fetches all products with images from the main API
    2. Generates embeddings for each image
    3. Creates a new FAISS index
    
    Returns:
        RebuildIndexResponse with status
    """
    async def rebuild_task():
        logger.info("Starting index rebuild...")
        
        try:
            # Clear existing index
            components.faiss_index.clear()
            
            # Fetch products in batches
            skip = 0
            total_indexed = 0
            
            while True:
                products = await components.catalog_client.get_all_products_with_images(
                    skip=skip,
                    limit=batch_size
                )
                
                if not products:
                    break
                
                # Process batch
                for product in products:
                    try:
                        # Download and index
                        import httpx
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(product["image_url"])
                            if response.status_code != 200:
                                continue
                            
                            image = validate_image(response.content)
                            if image is None:
                                continue
                            
                            embedding = components.embedder.encode_image(image)
                            components.faiss_index.add_embeddings(
                                embedding.reshape(1, -1),
                                [product["product_id"]]
                            )
                            total_indexed += 1
                            
                    except Exception as e:
                        logger.warning(f"Error indexing product {product.get('product_id')}: {e}")
                        continue
                
                skip += batch_size
            
            # Save the new index
            components.faiss_index.save_index()
            logger.info(f"Index rebuild complete. Indexed {total_indexed} products.")
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
    
    # Add to background tasks
    background_tasks.add_task(rebuild_task)
    
    return RebuildIndexResponse(
        status="started",
        message="Index rebuild started in background"
    )


@router.get("/index/stats")
async def get_index_stats():
    """Get statistics about the FAISS index.
    
    Returns:
        Dictionary with index statistics
    """
    return components.faiss_index.get_stats()


@router.get("/health/catalog")
async def check_catalog_health():
    """Check if main API catalog is accessible.
    
    Returns:
        Health status of catalog connection
    """
    is_healthy = await components.catalog_client.health_check()
    
    return {
        "catalog_healthy": is_healthy,
        "catalog_url": settings.MAIN_API_URL
    }


@router.get("/brands")
async def get_known_brands():
    """Get list of known auto parts brands.
    
    Returns:
        List of brand names recognized by OCR
    """
    return {
        "brands": components.brand_matcher.get_all_brands(),
        "count": len(components.brand_matcher.get_all_brands())
    }


@router.get("/categories")
async def get_supported_categories():
    """Get list of supported auto part categories.
    
    Returns:
        List of category names for detection
    """
    return {
        "categories": components.detector.get_supported_categories()
    }
