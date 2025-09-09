from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import hashlib
import os
import logging
from datetime import datetime
from typing import Literal, Optional
import shutil
from advanced_ai_detection import run_advanced_ai_checks, AdvancedForgeryResult
from blockchain_payload import prepare_blockchain_payload, validate_blockchain_payload

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(title="Justice Chain", description="Document upload and verification system")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Valid document types
DocType = Literal["aadhaar", "pan", "stamp", "court", "bank"]

def generate_file_hash(file_path: str) -> str:
    """Generate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_document(
    doc_type: DocType = Form(...),
    file: UploadFile = File(...),
    run_ai_checks: bool = Form(False),
    prepare_blockchain: bool = Form(False)
):
    """
    Upload a document with specified type
    
    - **doc_type**: Type of document (aadhaar, pan, stamp, court, bank)
    - **file**: PDF or image file to upload
    - **run_ai_checks**: Whether to run AI-based forgery detection (optional)
    - **prepare_blockchain**: Whether to prepare blockchain payload (optional)
    """
    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename with timestamp
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{doc_type}_{timestamp_str}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate SHA256 hash
        file_hash = generate_file_hash(file_path)
        
        # Prepare response
        response_data = {
            "filename": filename,
            "hash": file_hash,
            "timestamp": timestamp.isoformat(),
            "doc_type": doc_type,
            "original_filename": file.filename,
            "file_size": os.path.getsize(file_path)
        }
        
        # Run AI checks if requested
        if run_ai_checks:
            try:
                ai_result = await run_ai_forgery_check(file_path, doc_type)
                response_data["ai_analysis"] = {
                    "status": ai_result.status,
                    "confidence": ai_result.confidence_percentage,
                    "checks": ai_result.checks,
                    "detected_issues": ai_result.detected_issues,
                    "ai_scores": ai_result.ai_scores,
                    "forensic_analysis": ai_result.forensic_analysis,
                    "analysis_timestamp": ai_result.timestamp
                }
            except Exception as e:
                response_data["ai_analysis"] = {
                    "status": "error",
                    "confidence": 0,
                    "checks": {},
                    "error": f"AI analysis failed: {str(e)}",
                    "analysis_timestamp": datetime.now().isoformat()
                }
        
        # Prepare blockchain payload if requested
        if prepare_blockchain:
            try:
                blockchain_payload = prepare_blockchain_payload(response_data)
                response_data["blockchain_payload"] = blockchain_payload
                response_data["blockchain_ready"] = validate_blockchain_payload(blockchain_payload)
            except Exception as e:
                response_data["blockchain_payload"] = None
                response_data["blockchain_ready"] = False
                response_data["blockchain_error"] = f"Blockchain preparation failed: {str(e)}"
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
    
    except Exception as e:
        # Clean up file if something went wrong
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def run_ai_forgery_check(file_path: str, doc_type: str):
    """
    Run advanced AI-based forgery detection
    
    Args:
        file_path: Path to the uploaded file
        doc_type: Type of document
        
    Returns:
        AdvancedForgeryResult
    """
    try:
        # Use advanced AI detection for comprehensive analysis
        advanced_result = run_advanced_ai_checks(file_path, doc_type)
        
        return advanced_result
        
    except Exception as e:
        logger.error(f"AI forgery check failed: {e}")
        # Ultimate fallback
        from advanced_ai_detection import AdvancedForgeryResult
        return AdvancedForgeryResult(
            status="error",
            confidence_percentage=0,
            checks={},
            detected_issues=[f"Analysis failed: {str(e)}"],
            ai_scores={},
            forensic_analysis={},
            timestamp=datetime.now().isoformat()
        )

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    doc_type: Optional[DocType] = Form(None),
    prepare_blockchain: bool = Form(False)
):
    """
    Analyze a document for forgery without saving it permanently
    
    - **file**: PDF or image file to analyze
    - **doc_type**: Type of document (optional - will auto-detect if not provided)
    - **prepare_blockchain**: Whether to prepare blockchain payload (optional)
    """
    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file for analysis
    temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    try:
        # Save temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Auto-detect document type if not provided
        if not doc_type:
            # For now, default to generic analysis if no type specified
            doc_type = "unknown"
        
        # Run AI forgery check
        if doc_type and doc_type != "unknown":
            ai_result = await run_ai_forgery_check(temp_file_path, doc_type)
        else:
            # Generic image forgery check using advanced AI
            ai_result = run_advanced_ai_checks(temp_file_path, "generic")
        
        # Generate file hash for reference
        file_hash = generate_file_hash(temp_file_path)
        
        response_data = {
            "original_filename": file.filename,
            "detected_doc_type": doc_type or "unknown",
            "file_hash": file_hash,
            "analysis_timestamp": datetime.now().isoformat(),
            "ai_analysis": {
                "status": ai_result.status,
                "confidence": ai_result.confidence_percentage,
                "checks": ai_result.checks,
                "detected_issues": ai_result.detected_issues,
                "metadata": ai_result.metadata
            }
        }
        
        # Prepare blockchain payload if requested
        if prepare_blockchain:
            try:
                blockchain_payload = prepare_blockchain_payload(response_data)
                response_data["blockchain_payload"] = blockchain_payload
                response_data["blockchain_ready"] = validate_blockchain_payload(blockchain_payload)
            except Exception as e:
                response_data["blockchain_payload"] = None
                response_data["blockchain_ready"] = False
                response_data["blockchain_error"] = f"Blockchain preparation failed: {str(e)}"
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/prepare-blockchain")
async def prepare_blockchain_from_result(result_json: dict):
    """
    Prepare blockchain payload from existing API result
    
    - **result_json**: Complete API response from upload or analysis endpoint
    
    Returns blockchain-ready payload with validation status
    """
    try:
        # Prepare blockchain payload
        blockchain_payload = prepare_blockchain_payload(result_json)
        
        # Validate the payload
        is_valid = validate_blockchain_payload(blockchain_payload)
        
        return JSONResponse(
            status_code=200,
            content={
                "blockchain_payload": blockchain_payload,
                "blockchain_ready": is_valid,
                "payload_size_bytes": len(str(blockchain_payload)),
                "preparation_timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to prepare blockchain payload: {str(e)}"
        )

@app.get("/blockchain-info")
async def get_blockchain_info():
    """
    Get information about blockchain payload format and requirements
    """
    return {
        "blockchain_payload_format": {
            "required_fields": [
                "file_hash",
                "ai_verification",
                "timestamp"
            ],
            "ai_verification_fields": [
                "status",
                "confidence"
            ],
            "optional_metadata": [
                "doc_type",
                "original_filename", 
                "file_size",
                "detected_issues_count",
                "checks_passed"
            ]
        },
        "status_values": [
            "authentic",
            "suspicious", 
            "forged",
            "error",
            "unknown"
        ],
        "confidence_range": "0-100 (integer percentage)",
        "example_payload": {
            "file_hash": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            "ai_verification": {
                "status": "suspicious",
                "confidence": 68
            },
            "timestamp": "2024-12-09T14:30:22.123456Z",
            "metadata": {
                "doc_type": "aadhaar",
                "checks_passed": "4/5",
                "detected_issues_count": 1
            }
        }
    }

@app.get("/scoring-info")
async def get_scoring_info():
    """
    Get information about the AI scoring system
    """
    return {
        "scoring_system": {
            "confidence_ranges": {
                "0-10%": "Authentic",
                "10-55%": "Suspicious", 
                "55-100%": "Forged"
            },
            "check_types": {
                "ocr_pattern": {
                    "description": "OCR pattern validation (Aadhaar/PAN number format)",
                    "weight": "High (3.0-3.5x)"
                },
                "template_match": {
                    "description": "Logo/watermark template matching",
                    "weight": "High (2.0-2.5x)"
                },
                "keyword_validation": {
                    "description": "Required keywords detection",
                    "weight": "High (3.0-3.5x)"
                },
                "ela_tampering": {
                    "description": "Error Level Analysis for image tampering",
                    "weight": "Medium (2.0-2.5x)"
                },
                "copy_move": {
                    "description": "Copy-move forgery detection",
                    "weight": "Medium (1.5-2.0x)"
                },
                "metadata_clean": {
                    "description": "EXIF/PDF metadata tampering check",
                    "weight": "Low-Medium (1.0-2.5x)"
                },
                "format_validation": {
                    "description": "Document format consistency check",
                    "weight": "Low-Medium (1.0-2.5x)"
                },
                "duplicate_detection": {
                    "description": "Duplicate transaction/element detection",
                    "weight": "High (3.0x)"
                }
            },
            "document_specific_weights": {
                "aadhaar": {
                    "ocr_pattern": 3.0,
                    "template_match": 2.5,
                    "ela_tampering": 2.0,
                    "copy_move": 1.5,
                    "metadata_clean": 1.0
                },
                "pan": {
                    "ocr_pattern": 3.0,
                    "template_match": 2.5,
                    "ela_tampering": 2.0,
                    "copy_move": 1.5,
                    "metadata_clean": 1.0
                },
                "stamp": {
                    "keyword_validation": 3.0,
                    "template_match": 2.0,
                    "ela_tampering": 2.5,
                    "copy_move": 1.5,
                    "metadata_clean": 1.0
                },
                "court": {
                    "keyword_validation": 3.5,
                    "metadata_clean": 2.5,
                    "format_validation": 1.0
                },
                "bank": {
                    "duplicate_detection": 3.0,
                    "format_validation": 2.5,
                    "metadata_clean": 2.0
                }
            }
        },
        "example_response": {
            "status": "suspicious",
            "confidence": 68,
            "checks": {
                "ocr_pattern": True,
                "template_match": False,
                "metadata_clean": True,
                "ela_tampering": False
            }
        }
    }

@app.get("/templates")
async def list_templates():
    """
    List available template files for logo matching
    """
    template_dir = "templates"
    if not os.path.exists(template_dir):
        return {"templates": [], "note": "Templates directory not found"}
    
    templates = []
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            templates.append(filename)
    
    return {
        "templates": templates,
        "template_directory": template_dir,
        "note": "Add template images (aadhaar_logo.png, pan_logo.png, stamp_watermark.png) for better detection"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
