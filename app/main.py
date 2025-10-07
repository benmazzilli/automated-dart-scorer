"""
FastAPI application for automated dart scoring.

Provides a REST API to upload dartboard images and receive scores.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import Dict, List
import io
import logging
from pathlib import Path
from fastapi.staticfiles import StaticFiles

from app.services.dart_scorer import DartScorer

# Initialise FastAPI app
app = FastAPI(
    title="Automated Dart Scorer API",
    description="Upload dartboard images and get automated scoring with computer vision",
    version="1.0.0"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise scorer
scorer = DartScorer(debug=False)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    """Serve the demo HTML page."""
    demo_path = Path(__file__).parent.parent / "demo.html"
    if demo_path.exists():
        return FileResponse(demo_path)
    return {
        "service": "Automated Dart Scorer API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "service": "Automated Dart Scorer API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/score")
async def score_dart_image(file: UploadFile = File(...)):
    """
    Score a dartboard image with darts.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with:
        - total_score: Total points from all darts
        - darts: List of individual dart scores with positions
        - success: Whether scoring was successful
        - message: Status message
    """
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Score the image
        result = scorer.score_image(image)
        
        # Format response
        response = {
            "success": result["success"],
            "message": result["message"],
            "total_score": result["total_score"],
            "darts": []
        }
        
        # Add individual dart scores
        for dart_score in result.get("dart_scores", []):
            response["darts"].append({
                "dart_id": dart_score["dart_id"],
                "position": {
                    "x": dart_score["position"][0],
                    "y": dart_score["position"][1]
                },
                "score": dart_score["score"],
                "multiplier": dart_score["multiplier"],
                "total": dart_score["total"],
                "region": dart_score["region"],
                "confidence": round(dart_score["confidence"], 3)
            })
        
        logger.info(f"Scored image: {result['total_score']} points from {len(response['darts'])} darts")
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/score-with-image")
async def score_with_visualisation(file: UploadFile = File(...)):
    """
    Score a dartboard image and return both the score JSON and the annotated image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Multipart response with:
        - JSON score data
        - Annotated image showing dartboard segments and dart positions
    """
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Score the image
        result = scorer.score_image(image)
        
        # Create visualisation
        visualisation = scorer.visualise_score(image, result)
        
        # Encode visualisation to JPEG
        _, buffer = cv2.imencode('.jpg', visualisation)
        img_bytes = buffer.tobytes()
        
        # Format JSON response
        score_data = {
            "success": result["success"],
            "message": result["message"],
            "total_score": result["total_score"],
            "darts": []
        }
        
        for dart_score in result.get("dart_scores", []):
            score_data["darts"].append({
                "dart_id": dart_score["dart_id"],
                "position": {
                    "x": dart_score["position"][0],
                    "y": dart_score["position"][1]
                },
                "score": dart_score["score"],
                "multiplier": dart_score["multiplier"],
                "total": dart_score["total"],
                "region": dart_score["region"],
                "confidence": round(dart_score["confidence"], 3)
            })
        
        logger.info(f"Scored image with visualisation: {result['total_score']} points")
        
        # Return image with score data in headers
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={
                "X-Score-Data": str(score_data),
                "X-Total-Score": str(result["total_score"]),
                "X-Darts-Count": str(len(score_data["darts"]))
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "service": "dart-scorer",
        "components": {
            "dartboard_detector": "operational",
            "dart_detector": "operational",
            "scorer": "operational"
        }
    }


@app.get("/mobile")
async def mobile_app():
    """Serve the mobile PWA."""
    mobile_path = Path(__file__).parent.parent / "mobile-app.html"
    if mobile_path.exists():
        return FileResponse(mobile_path)
    raise HTTPException(status_code=404, detail="Mobile app not found")


@app.get("/manifest.json")
async def manifest():
    """Serve PWA manifest."""
    manifest_path = Path(__file__).parent.parent / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/service-worker.js")
async def service_worker():
    """Serve service worker."""
    sw_path = Path(__file__).parent.parent / "service-worker.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


@app.get("/icon-192.png")
async def icon_192():
    """Serve 192x192 icon."""
    icon_path = Path(__file__).parent.parent / "icon-192.png"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Icon not found")


@app.get("/icon-512.png")
async def icon_512():
    """Serve 512x512 icon."""
    icon_path = Path(__file__).parent.parent / "icon-512.png"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Icon not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

