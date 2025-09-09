"""
Advanced AI Forgery Detection Module for Justice Chain
Implements real computer vision and machine learning techniques for document authenticity verification
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import pytesseract
import pdfplumber
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import logging
from scipy import ndimage
from skimage import feature, measure, filters, segmentation, morphology
from sklearn.cluster import DBSCAN
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedForgeryResult:
    """Enhanced result structure for AI forgery detection"""
    status: str  # "authentic", "suspicious", "forged"
    confidence_percentage: int
    checks: Dict[str, bool]
    detected_issues: List[str]
    ai_scores: Dict[str, float]
    forensic_analysis: Dict[str, any]
    timestamp: str

class RealAIForgeryDetector:
    """Advanced AI-powered document forgery detection using computer vision and ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_copy_move_forgery(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect copy-move forgery using ORB feature matching (free alternative to SURF)
        Returns: (is_forged, confidence, analysis_data)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Use ORB detector (free alternative to SURF)
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) < 50:
                return False, 0.0, {"reason": "insufficient_features"}
            
            # Match features with themselves to find duplicates
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(descriptors, descriptors, k=2)
            
            # Apply ratio test and distance filtering
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance and m.distance > 0:
                        # Exclude self-matches
                        if m.queryIdx != m.trainIdx:
                            good_matches.append(m)
            
            # Analyze spatial distribution of matches
            if len(good_matches) > 10:
                query_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
                train_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
                
                # Calculate distances between matched points
                distances = np.linalg.norm(query_pts - train_pts, axis=1)
                
                # Check for systematic displacement (indicating copy-move)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                
                # Cluster analysis to find consistent displacements
                displacement_vectors = query_pts - train_pts
                if len(displacement_vectors) > 5:
                    clustering = DBSCAN(eps=10, min_samples=3).fit(displacement_vectors)
                    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                    
                    if n_clusters > 0:
                        confidence = min(0.9, (n_clusters * len(good_matches)) / 100)
                        return True, confidence, {
                            "clusters_found": n_clusters,
                            "matching_points": len(good_matches),
                            "mean_displacement": float(mean_distance)
                        }
            
            return False, 0.1, {"matching_points": len(good_matches)}
            
        except Exception as e:
            self.logger.warning(f"Copy-move detection failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def detect_noise_inconsistencies(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect noise pattern inconsistencies that indicate tampering
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Divide image into blocks
            h, w = gray.shape
            block_size = 64
            noise_levels = []
            
            for i in range(0, h - block_size, block_size // 2):
                for j in range(0, w - block_size, block_size // 2):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Calculate noise level using Laplacian variance
                    noise_level = cv2.Laplacian(block, cv2.CV_64F).var()
                    noise_levels.append(noise_level)
            
            if len(noise_levels) < 4:
                return False, 0.0, {"reason": "insufficient_blocks"}
            
            # Analyze noise distribution
            noise_array = np.array(noise_levels)
            mean_noise = np.mean(noise_array)
            std_noise = np.std(noise_array)
            
            # Detect outliers (potential tampered regions)
            outliers = noise_array[np.abs(noise_array - mean_noise) > 2 * std_noise]
            outlier_ratio = len(outliers) / len(noise_levels)
            
            # High coefficient of variation indicates inconsistent noise
            cv_noise = std_noise / mean_noise if mean_noise > 0 else 0
            
            is_suspicious = cv_noise > 0.8 or outlier_ratio > 0.15
            confidence = min(0.85, cv_noise + outlier_ratio)
            
            return is_suspicious, confidence, {
                "noise_cv": float(cv_noise),
                "outlier_ratio": float(outlier_ratio),
                "mean_noise": float(mean_noise)
            }
            
        except Exception as e:
            self.logger.warning(f"Noise analysis failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def detect_jpeg_compression_artifacts(self, image_path: str) -> Tuple[bool, float, Dict]:
        """
        Analyze JPEG compression artifacts to detect tampering
        """
        try:
            # Load image and analyze DCT coefficients
            image = cv2.imread(image_path)
            if image is None:
                return False, 0.0, {"error": "Could not load image"}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simulate JPEG compression analysis
            # In real implementation, this would analyze DCT blocks
            
            # Check for double JPEG compression indicators
            # 1. Histogram analysis of DCT coefficients
            dct_blocks = []
            for i in range(0, gray.shape[0] - 8, 8):
                for j in range(0, gray.shape[1] - 8, 8):
                    block = gray[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_blocks.append(dct_block.flatten())
            
            if len(dct_blocks) == 0:
                return False, 0.0, {"reason": "no_dct_blocks"}
            
            # Analyze coefficient distribution
            all_coeffs = np.concatenate(dct_blocks)
            
            # Look for double quantization artifacts
            hist, bins = np.histogram(all_coeffs, bins=100, range=(-50, 50))
            
            # Check for periodic patterns in histogram (indicator of double compression)
            # Simplified detection: look for multiple peaks
            peaks = feature.peak_local_maxima(hist, min_distance=3)[0]
            
            is_double_compressed = len(peaks) > 8
            confidence = min(0.8, len(peaks) / 20) if is_double_compressed else 0.1
            
            return is_double_compressed, confidence, {
                "histogram_peaks": len(peaks),
                "coefficient_range": (float(np.min(all_coeffs)), float(np.max(all_coeffs)))
            }
            
        except Exception as e:
            self.logger.warning(f"JPEG analysis failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def detect_edge_inconsistencies(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect edge inconsistencies that may indicate splicing
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply different edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            edges_sobel = np.uint8(np.absolute(edges_sobel))
            
            # Analyze edge consistency
            # Look for abrupt changes in edge characteristics
            
            # Divide image into regions and analyze edge density
            h, w = gray.shape
            region_size = min(h, w) // 4
            edge_densities = []
            
            for i in range(0, h - region_size, region_size // 2):
                for j in range(0, w - region_size, region_size // 2):
                    region_edges = edges_canny[i:i+region_size, j:j+region_size]
                    edge_density = np.sum(region_edges > 0) / (region_size * region_size)
                    edge_densities.append(edge_density)
            
            if len(edge_densities) < 4:
                return False, 0.0, {"reason": "insufficient_regions"}
            
            # Check for significant variations in edge density
            density_array = np.array(edge_densities)
            mean_density = np.mean(density_array)
            std_density = np.std(density_array)
            
            cv_density = std_density / mean_density if mean_density > 0 else 0
            
            # High coefficient of variation indicates potential splicing
            is_inconsistent = cv_density > 0.6
            confidence = min(0.9, cv_density)
            
            return is_inconsistent, confidence, {
                "edge_cv": float(cv_density),
                "mean_edge_density": float(mean_density),
                "regions_analyzed": len(edge_densities)
            }
            
        except Exception as e:
            self.logger.warning(f"Edge analysis failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def detect_lighting_inconsistencies(self, image: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect lighting inconsistencies that may indicate composite images
        """
        try:
            # Convert to different color spaces for analysis
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze lightness channel (L in LAB)
            lightness = lab[:, :, 0]
            
            # Segment image to find distinct regions
            # Apply watershed or similar segmentation
            markers = segmentation.watershed(lightness)
            
            # Analyze lighting direction for each segment
            regions = measure.regionprops(markers, intensity_image=lightness)
            
            if len(regions) < 2:
                return False, 0.0, {"reason": "insufficient_regions"}
            
            # Calculate lighting statistics for each region
            lighting_stats = []
            for region in regions:
                if region.area > 100:  # Filter small regions
                    mean_intensity = np.mean(region.intensity_image)
                    std_intensity = np.std(region.intensity_image)
                    lighting_stats.append((mean_intensity, std_intensity))
            
            if len(lighting_stats) < 2:
                return False, 0.0, {"reason": "insufficient_valid_regions"}
            
            # Check for inconsistent lighting patterns
            intensities = [stat[0] for stat in lighting_stats]
            intensity_range = max(intensities) - min(intensities)
            
            # Normalized intensity range
            normalized_range = intensity_range / 255.0
            
            # High range indicates potential lighting inconsistencies
            is_inconsistent = normalized_range > 0.4
            confidence = min(0.8, normalized_range)
            
            return is_inconsistent, confidence, {
                "intensity_range": float(intensity_range),
                "normalized_range": float(normalized_range),
                "regions_found": len(lighting_stats)
            }
            
        except Exception as e:
            self.logger.warning(f"Lighting analysis failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def analyze_metadata_consistency(self, image_path: str) -> Tuple[bool, float, Dict]:
        """
        Analyze EXIF metadata for signs of manipulation
        """
        try:
            file_extension = os.path.splitext(image_path)[1].lower()
            
            if file_extension == '.pdf':
                # For PDF files, analyze PDF metadata instead of EXIF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(image_path)
                    metadata = doc.metadata
                    doc.close()
                    
                    metadata_issues = []
                    analysis = {}
                    
                    # Check PDF metadata
                    for key, value in metadata.items():
                        if value:
                            analysis[key] = str(value)
                            
                            # Check for editing software
                            if key in ['creator', 'producer']:
                                editing_software = ['adobe acrobat', 'pdftk', 'libreoffice', 'microsoft']
                                if any(software in str(value).lower() for software in editing_software):
                                    metadata_issues.append(f"PDF editing software detected: {value}")
                    
                    # Check for missing expected metadata
                    expected_tags = ['creator', 'producer', 'creationDate']
                    missing_tags = [tag for tag in expected_tags if tag not in analysis]
                    
                    if len(missing_tags) > 1:
                        metadata_issues.append(f"Missing PDF metadata: {', '.join(missing_tags)}")
                    
                    is_suspicious = len(metadata_issues) > 0
                    confidence = min(0.7, len(metadata_issues) * 0.3)
                    
                    return is_suspicious, confidence, {
                        "issues_found": metadata_issues,
                        "metadata_tags": len(analysis),
                        "analysis": analysis,
                        "file_type": "PDF"
                    }
                    
                except Exception as e:
                    self.logger.warning(f"PDF metadata analysis failed: {e}")
                    return False, 0.0, {"error": f"PDF metadata analysis failed: {str(e)}"}
            
            else:
                # Handle image files with EXIF data
                image = Image.open(image_path)
                exifdata = image.getexif()
                
                if not exifdata:
                    return False, 0.0, {"reason": "no_exif_data"}
                
                metadata_issues = []
                analysis = {}
                
                # Check for common manipulation indicators
                for tag_id in exifdata:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)
                    analysis[tag] = str(data)
                    
                    # Check for software indicators
                    if tag in ['Software', 'ProcessingSoftware']:
                        editing_software = ['photoshop', 'gimp', 'paint.net', 'pixlr', 'canva']
                        if any(software in str(data).lower() for software in editing_software):
                            metadata_issues.append(f"Editing software detected: {data}")
                    
                    # Check for inconsistent timestamps
                    if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        try:
                            timestamp = datetime.strptime(str(data), "%Y:%m:%d %H:%M:%S")
                            if timestamp.year < 2000 or timestamp.year > 2030:
                                metadata_issues.append(f"Suspicious timestamp: {data}")
                        except:
                            pass
                
                # Check for missing expected metadata
                expected_tags = ['Make', 'Model', 'DateTime']
                missing_tags = [tag for tag in expected_tags if tag not in analysis]
                
                if len(missing_tags) > 1:
                    metadata_issues.append(f"Missing metadata: {', '.join(missing_tags)}")
                
                is_suspicious = len(metadata_issues) > 0
                confidence = min(0.7, len(metadata_issues) * 0.3)
                
                return is_suspicious, confidence, {
                    "issues_found": metadata_issues,
                    "metadata_tags": len(analysis),
                    "analysis": analysis,
                    "file_type": "Image"
                }
            
        except Exception as e:
            self.logger.warning(f"Metadata analysis failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def load_image_from_file(self, file_path: str) -> np.ndarray:
        """
        Load image from file, handling both images and PDFs
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            # Handle PDF files
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                page = doc[0]  # Get first page
                
                # Render page as image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from io import BytesIO
                pil_image = Image.open(BytesIO(img_data))
                
                # Convert to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                doc.close()
                
                return image
                
            except ImportError:
                # Fallback: Try with pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        page = pdf.pages[0]
                        
                        # Convert page to image
                        page_image = page.to_image(resolution=150)
                        pil_image = page_image.original
                        
                        # Convert to OpenCV format
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        
                        return image
                        
                except Exception as e:
                    self.logger.warning(f"PDF processing failed: {e}")
                    raise ValueError(f"Cannot process PDF file: {file_path}")
                    
        else:
            # Handle regular image files
            image = cv2.imread(file_path)
            if image is None:
                # Try with PIL for other formats
                pil_image = Image.open(file_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return image

    def comprehensive_forgery_analysis(self, image_path: str) -> AdvancedForgeryResult:
        """
        Run comprehensive AI-powered forgery detection
        """
        try:
            # Load image (handles both images and PDFs)
            image = self.load_image_from_file(image_path)
            
            # Run all detection methods
            copy_move_result = self.detect_copy_move_forgery(image)
            noise_result = self.detect_noise_inconsistencies(image)
            compression_result = self.detect_jpeg_compression_artifacts(image_path)
            edge_result = self.detect_edge_inconsistencies(image)
            lighting_result = self.detect_lighting_inconsistencies(image)
            metadata_result = self.analyze_metadata_consistency(image_path)
            
            # Compile results
            checks = {
                "copy_move_detection": not copy_move_result[0],
                "noise_consistency": not noise_result[0],
                "compression_analysis": not compression_result[0],
                "edge_consistency": not edge_result[0],
                "lighting_consistency": not lighting_result[0],
                "metadata_integrity": not metadata_result[0]
            }
            
            # Calculate weighted confidence score
            weights = {
                "copy_move_detection": 0.25,
                "noise_consistency": 0.15,
                "compression_analysis": 0.20,
                "edge_consistency": 0.15,
                "lighting_consistency": 0.15,
                "metadata_integrity": 0.10
            }
            
            # Calculate forgery probability
            forgery_scores = {
                "copy_move": copy_move_result[1],
                "noise_inconsistency": noise_result[1],
                "compression_artifacts": compression_result[1],
                "edge_inconsistency": edge_result[1],
                "lighting_inconsistency": lighting_result[1],
                "metadata_issues": metadata_result[1]
            }
            
            weighted_score = sum(
                forgery_scores[key.replace("_detection", "").replace("_consistency", "_inconsistency").replace("_analysis", "_artifacts").replace("_integrity", "_issues")] * weight
                for key, weight in weights.items()
            )
            
            confidence_percentage = int(weighted_score * 100)
            
            # Determine status
            if confidence_percentage <= 10:
                status = "authentic"
            elif confidence_percentage <= 55:
                status = "suspicious"
            else:
                status = "forged"
            
            # Collect detected issues
            detected_issues = []
            if copy_move_result[0]:
                detected_issues.append("Copy-move forgery patterns detected")
            if noise_result[0]:
                detected_issues.append("Inconsistent noise patterns")
            if compression_result[0]:
                detected_issues.append("Double JPEG compression artifacts")
            if edge_result[0]:
                detected_issues.append("Edge inconsistencies indicating splicing")
            if lighting_result[0]:
                detected_issues.append("Inconsistent lighting patterns")
            if metadata_result[0]:
                detected_issues.append("Suspicious metadata modifications")
            
            forensic_analysis = {
                "copy_move_analysis": copy_move_result[2],
                "noise_analysis": noise_result[2],
                "compression_analysis": compression_result[2],
                "edge_analysis": edge_result[2],
                "lighting_analysis": lighting_result[2],
                "metadata_analysis": metadata_result[2]
            }
            
            return AdvancedForgeryResult(
                status=status,
                confidence_percentage=confidence_percentage,
                checks=checks,
                detected_issues=detected_issues,
                ai_scores=forgery_scores,
                forensic_analysis=forensic_analysis,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return AdvancedForgeryResult(
                status="error",
                confidence_percentage=0,
                checks={},
                detected_issues=[f"Analysis failed: {str(e)}"],
                ai_scores={},
                forensic_analysis={},
                timestamp=datetime.now().isoformat()
            )

# Create global instance
advanced_ai_detector = RealAIForgeryDetector()

def run_advanced_ai_checks(file_path: str, doc_type: str) -> AdvancedForgeryResult:
    """
    Main function to run advanced AI forgery detection
    """
    return advanced_ai_detector.comprehensive_forgery_analysis(file_path)
