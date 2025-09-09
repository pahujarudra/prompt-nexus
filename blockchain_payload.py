"""
Blockchain Payload Module for Justice Chain
Prepares document verification results for blockchain storage
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_blockchain_payload(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare document verification result for blockchain storage
    
    Args:
        result_json: Complete API response from document upload/analysis
        
    Returns:
        Dictionary formatted for blockchain storage with essential verification data
        
    Example:
        Input: Full API response with file info and AI analysis
        Output: {
            "file_hash": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            "ai_verification": {
                "status": "suspicious", 
                "confidence": 68
            },
            "timestamp": "2024-12-09T14:30:22.123456Z"
        }
    """
    try:
        # Extract file hash
        file_hash = result_json.get("hash") or result_json.get("file_hash", "")
        
        # Extract AI verification results
        ai_analysis = result_json.get("ai_analysis", {})
        
        # Prepare AI verification summary
        ai_verification = {
            "status": ai_analysis.get("status", "unknown"),
            "confidence": ai_analysis.get("confidence", 0)
        }
        
        # Use existing timestamp or create new one
        timestamp = result_json.get("timestamp") or result_json.get("analysis_timestamp")
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        # Ensure timestamp has 'Z' suffix for ISO format
        if not timestamp.endswith('Z') and not timestamp.endswith('+00:00'):
            if '.' in timestamp:
                timestamp = timestamp + "Z"
            else:
                timestamp = timestamp + ".000000Z"
        
        # Prepare blockchain payload
        blockchain_payload = {
            "file_hash": file_hash,
            "ai_verification": ai_verification,
            "timestamp": timestamp
        }
        
        # Add optional metadata if available
        metadata = {}
        
        # Add document type if available
        if "doc_type" in result_json:
            metadata["doc_type"] = result_json["doc_type"]
        elif "detected_doc_type" in result_json:
            metadata["doc_type"] = result_json["detected_doc_type"]
        
        # Add original filename if available
        if "original_filename" in result_json:
            metadata["original_filename"] = result_json["original_filename"]
        
        # Add file size if available
        if "file_size" in result_json:
            metadata["file_size"] = result_json["file_size"]
        
        # Add detected issues if AI analysis was performed
        if ai_analysis and "detected_issues" in ai_analysis:
            issues = ai_analysis["detected_issues"]
            if issues:
                metadata["detected_issues_count"] = len(issues)
                metadata["primary_issue"] = issues[0] if issues else None
        
        # Add individual checks summary if available
        if ai_analysis and "checks" in ai_analysis:
            checks = ai_analysis["checks"]
            passed_checks = sum(1 for check in checks.values() if check)
            total_checks = len(checks)
            metadata["checks_passed"] = f"{passed_checks}/{total_checks}"
        
        # Only add metadata if it contains useful information
        if metadata:
            blockchain_payload["metadata"] = metadata
        
        logger.info(f"Prepared blockchain payload for file hash: {file_hash[:16]}...")
        return blockchain_payload
        
    except Exception as e:
        logger.error(f"Error preparing blockchain payload: {str(e)}")
        
        # Return minimal payload with error indication
        return {
            "file_hash": result_json.get("hash", result_json.get("file_hash", "unknown")),
            "ai_verification": {
                "status": "error",
                "confidence": 0
            },
            "timestamp": datetime.now().isoformat() + "Z",
            "error": f"Payload preparation failed: {str(e)}"
        }

def validate_blockchain_payload(payload: Dict[str, Any]) -> bool:
    """
    Validate that blockchain payload has required fields
    
    Args:
        payload: Blockchain payload dictionary
        
    Returns:
        True if payload is valid, False otherwise
    """
    required_fields = ["file_hash", "ai_verification", "timestamp"]
    
    try:
        # Check required top-level fields
        for field in required_fields:
            if field not in payload:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate ai_verification structure
        ai_verification = payload["ai_verification"]
        if not isinstance(ai_verification, dict):
            logger.error("ai_verification must be a dictionary")
            return False
        
        if "status" not in ai_verification or "confidence" not in ai_verification:
            logger.error("ai_verification missing status or confidence")
            return False
        
        # Validate status values
        valid_statuses = ["authentic", "suspicious", "forged", "error", "unknown"]
        if ai_verification["status"] not in valid_statuses:
            logger.error(f"Invalid status: {ai_verification['status']}")
            return False
        
        # Validate confidence range
        confidence = ai_verification["confidence"]
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
            logger.error(f"Invalid confidence value: {confidence}")
            return False
        
        # Validate file hash format (should be hex string)
        file_hash = payload["file_hash"]
        if not file_hash or not isinstance(file_hash, str):
            logger.error("Invalid file hash")
            return False
        
        # Validate timestamp format
        timestamp = payload["timestamp"]
        if not timestamp or not isinstance(timestamp, str):
            logger.error("Invalid timestamp")
            return False
        
        logger.info("Blockchain payload validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False

def create_blockchain_summary(payload: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of blockchain payload
    
    Args:
        payload: Blockchain payload dictionary
        
    Returns:
        Formatted summary string
    """
    try:
        ai_verification = payload["ai_verification"]
        status = ai_verification["status"].upper()
        confidence = ai_verification["confidence"]
        file_hash_short = payload["file_hash"][:16] + "..."
        
        summary = f"Document Verification Summary:\n"
        summary += f"File Hash: {file_hash_short}\n"
        summary += f"AI Status: {status} ({confidence}% confidence)\n"
        summary += f"Timestamp: {payload['timestamp']}\n"
        
        # Add metadata if available
        if "metadata" in payload:
            metadata = payload["metadata"]
            if "doc_type" in metadata:
                summary += f"Document Type: {metadata['doc_type'].upper()}\n"
            if "checks_passed" in metadata:
                summary += f"Checks Passed: {metadata['checks_passed']}\n"
            if "detected_issues_count" in metadata:
                summary += f"Issues Detected: {metadata['detected_issues_count']}\n"
        
        return summary
        
    except Exception as e:
        return f"Error creating summary: {str(e)}"

# Example usage and testing functions
def example_usage():
    """Example of how to use the blockchain payload functions"""
    
    # Example API response from Justice Chain
    sample_api_response = {
        "filename": "aadhaar_20241209_143022_document.pdf",
        "hash": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        "timestamp": "2024-12-09T14:30:22.123456",
        "doc_type": "aadhaar",
        "original_filename": "document.pdf",
        "file_size": 1024,
        "ai_analysis": {
            "status": "suspicious",
            "confidence": 35,
            "checks": {
                "ocr_pattern": False,
                "template_match": True,
                "ela_tampering": True,
                "copy_move": True,
                "metadata_clean": True
            },
            "detected_issues": ["OCR pattern validation failed"],
            "analysis_timestamp": "2024-12-09T14:30:23.456789"
        }
    }
    
    print("Sample API Response:")
    print(json.dumps(sample_api_response, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Prepare blockchain payload
    blockchain_payload = prepare_blockchain_payload(sample_api_response)
    
    print("Blockchain Payload:")
    print(json.dumps(blockchain_payload, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Validate payload
    is_valid = validate_blockchain_payload(blockchain_payload)
    print(f"Payload Valid: {is_valid}")
    print("\n" + "="*25 + "\n")
    
    # Create summary
    summary = create_blockchain_summary(blockchain_payload)
    print(summary)

if __name__ == "__main__":
    example_usage()
