from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

from app.pose.estimator import PoseEstimator
from app.pose.normalizer import normalize_pose
from app.matching.searcher import PoseSearcher

app = FastAPI(title="MEMER App", description="Real-time Pose to Meme Similarity Search Engine")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent
MEMES_DIR = PROJECT_ROOT / "data" / "memes"
STATIC_DIR = PROJECT_ROOT / "app" / "static"

# Ensure directories exist
MEMES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount memes dataset images
app.mount("/memes", StaticFiles(directory=str(MEMES_DIR)), name="memes")


@app.websocket("/stream")
async def pose_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected")
    
    estimator = PoseEstimator()
    searcher = PoseSearcher()
    
    try:
        while True:
            # Receive raw image frames as bytes (e.g. from canvas.toBlob)
            frame_bytes = await websocket.receive_bytes()
            
            # Estimate pose landmarks
            result = estimator.process_bytes(frame_bytes)
            if not result or not result.valid:
                await websocket.send_json({
                    "matched": False,
                    "reason": "No human detected. Please step into the camera frame."
                })
                continue
                
            # Normalize coordinates
            vector = normalize_pose(result)
            if vector is None:
                await websocket.send_json({
                    "matched": False,
                    "reason": "Aligning pose... make sure both shoulders are clearly visible."
                })
                continue
                
            # Query the index for the closest memes
            matches = searcher.search(vector, k=3)
            if not matches:
                await websocket.send_json({
                    "matched": False,
                    "reason": "Searching... but no matching memes found in index."
                })
                continue
                
            # Construct response paths
            response_matches = []
            for m in matches:
                filename = Path(m["filename"]).name
                response_matches.append({
                    "name": m["name"],
                    "url": f"/memes/{filename}",
                    "similarity": m["similarity"],
                    "source": m["source"]
                })
                
            # Prepare skeleton coordinates for real-time overlay visualization
            landmarks_data = {
                "body": result.body_landmarks.tolist(),
                "left_hand": result.left_hand.tolist() if result.left_hand_detected else [],
                "right_hand": result.right_hand.tolist() if result.right_hand_detected else [],
                "left_hand_detected": result.left_hand_detected,
                "right_hand_detected": result.right_hand_detected
            }
            
            await websocket.send_json({
                "matched": True,
                "matches": response_matches,
                "landmarks": landmarks_data
            })
            
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error during stream processing: {e}")
    finally:
        estimator.close()
        print("[WebSocket] Stream resources cleaned up")


# Mount frontend static files last, serving index.html at the root "/"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")