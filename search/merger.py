"""Result Merger Module.

This module provides functionality to merge and rank results from
multiple search sources (catalog DB and FAISS vector search).
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with metadata."""
    product_id: int
    score: float
    match_type: str  # 'catalog', 'vector', 'combined'
    confidence: float
    source_scores: Optional[Dict[str, float]] = None


class ResultMerger:
    """Merges and ranks results from multiple search sources."""
    
    def __init__(
        self,
        catalog_weight: float = 0.6,
        vector_weight: float = 0.4,
        confidence_threshold: float = 0.5,
        low_confidence_catalog_weight: float = 0.3,
        low_confidence_vector_weight: float = 0.7
    ):
        """Initialize the result merger.
        
        Args:
            catalog_weight: Weight for catalog matches (high confidence)
            vector_weight: Weight for vector matches (high confidence)
            confidence_threshold: Threshold to switch weight strategies
            low_confidence_catalog_weight: Catalog weight when confidence is low
            low_confidence_vector_weight: Vector weight when confidence is low
        """
        self.catalog_weight = catalog_weight
        self.vector_weight = vector_weight
        self.confidence_threshold = confidence_threshold
        self.low_confidence_catalog_weight = low_confidence_catalog_weight
        self.low_confidence_vector_weight = low_confidence_vector_weight
        
        logger.info(
            f"ResultMerger initialized (catalog_weight={catalog_weight}, "
            f"vector_weight={vector_weight}, threshold={confidence_threshold})"
        )
    
    def merge(
        self,
        catalog_results: List[Dict],
        vector_results: List[tuple],
        detection_confidence: float,
        max_results: int = 20,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> List[SearchResult]:
        """Merge catalog and vector search results.
        
        Args:
            catalog_results: Results from catalog search
            vector_results: Results from FAISS vector search
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            alpha: Dynamic image/vector weight (PDF spec §8). Overrides internal logic if set.
            beta: Dynamic text/catalog weight (PDF spec §8). Overrides internal logic if set.
            gamma: Dynamic meta weight (PDF spec §8). Acknowledged but not used separately.
            
        Returns:
            Merged and ranked list of SearchResult objects
        """
        # Use dynamic weights from caller if provided (PDF spec §8)
        # alpha = image signal → vector_weight
        # beta  = text/OCR signal → catalog_weight
        # gamma = meta signal (acknowledged, not a separate bucket)
        if alpha is not None and beta is not None:
            catalog_weight = beta
            vector_weight = alpha
            logger.debug(
                f"Dynamic weights applied: alpha(vector)={alpha}, "
                f"beta(catalog)={beta}, gamma(meta)={gamma}"
            )
        elif detection_confidence < self.confidence_threshold:
            # Low confidence: rely more on visual similarity
            catalog_weight = self.low_confidence_catalog_weight
            vector_weight = self.low_confidence_vector_weight
            logger.debug(
                f"Low confidence ({detection_confidence:.2f}), "
                f"using weights: catalog={catalog_weight}, vector={vector_weight}"
            )
        else:
            catalog_weight = self.catalog_weight
            vector_weight = self.vector_weight
        
        # Aggregate scores by product ID
        product_scores = defaultdict(lambda: {"catalog": 0.0, "vector": 0.0})
        
        # Add catalog scores
        for result in catalog_results:
            product_id = result.get('product_id')
            if product_id is None:
                continue
            # Catalog matches get full weight (score is typically 1.0 for exact match)
            score = result.get('score', 1.0)
            product_scores[product_id]["catalog"] = catalog_weight * score
        
        # Add vector scores
        for product_id, similarity in vector_results:
            product_scores[product_id]["vector"] = vector_weight * similarity
        
        # Calculate combined scores and determine match types
        results = []
        catalog_ids = {r.get('product_id') for r in catalog_results if r.get('product_id')}
        vector_ids = {p[0] for p in vector_results}
        
        for product_id, scores in product_scores.items():
            combined_score = scores["catalog"] + scores["vector"]
            
            # Determine match type
            in_catalog = product_id in catalog_ids
            in_vector = product_id in vector_ids
            
            if in_catalog and in_vector:
                match_type = 'combined'
            elif in_catalog:
                match_type = 'catalog'
            else:
                match_type = 'vector'
            
            results.append(SearchResult(
                product_id=product_id,
                score=combined_score,
                match_type=match_type,
                confidence=detection_confidence,
                source_scores={
                    "catalog": scores["catalog"],
                    "vector": scores["vector"]
                }
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Log top results
        if results:
            top_result = results[0]
            logger.info(
                f"Top result: product_id={top_result.product_id}, "
                f"score={top_result.score:.3f}, match_type={top_result.match_type}"
            )
        
        return results[:max_results]
    
    def merge_with_diversity(
        self,
        catalog_results: List[Dict],
        vector_results: List[tuple],
        detection_confidence: float,
        max_results: int = 20,
        diversity_threshold: float = 0.8
    ) -> List[SearchResult]:
        """Merge results with diversity to avoid similar products.
        
        Args:
            catalog_results: Results from catalog search
            vector_results: Results from FAISS vector search
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            diversity_threshold: Minimum score difference to include
            
        Returns:
            Diversified list of SearchResult objects
        """
        # Get initial merged results
        all_results = self.merge(
            catalog_results,
            vector_results,
            detection_confidence,
            max_results * 2  # Get more to allow filtering
        )
        
        if not all_results:
            return []
        
        # Apply diversity filtering
        diversified = [all_results[0]]  # Always include top result
        last_score = all_results[0].score
        
        for result in all_results[1:]:
            # Include if score is significantly different or different match type
            if (last_score - result.score) > (1 - diversity_threshold) * last_score:
                diversified.append(result)
                last_score = result.score
            
            if len(diversified) >= max_results:
                break
        
        return diversified
    
    def rerank_by_match_type(
        self,
        results: List[SearchResult],
        type_weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """Re-rank results based on match type preferences.
        
        Args:
            results: List of search results
            type_weights: Weights for each match type
            
        Returns:
            Re-ranked list of results
        """
        if type_weights is None:
            type_weights = {
                'combined': 1.2,  # Boost combined matches
                'catalog': 1.0,
                'vector': 0.9
            }
        
        # Apply type weights
        for result in results:
            weight = type_weights.get(result.match_type, 1.0)
            result.score = result.score * weight
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


def format_results_for_response(
    results: List[SearchResult],
    products_data: Optional[Dict[int, Dict]] = None
) -> List[Dict]:
    """Format search results for API response.
    
    Args:
        results: List of SearchResult objects
        products_data: Optional product details by ID
        
    Returns:
        List of formatted result dictionaries
    """
    formatted = []
    
    for result in results:
        item = {
            "product_id": result.product_id,
            "score": round(result.score, 3),
            "match_type": result.match_type,
            "confidence": round(result.confidence, 3)
        }
        
        # Add product details if available
        if products_data and result.product_id in products_data:
            item["product"] = products_data[result.product_id]
        
        formatted.append(item)
    
    return formatted
