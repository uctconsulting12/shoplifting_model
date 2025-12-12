import cv2
import json
import base64
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from src.models.ppe_local import ppe_detection
from src.store_s3.ppe_store import upload_to_s3
from database.shoplifting_query import insert_ppe_frame
from PIL import Image

logger = logging.getLogger("queue_monitoring")
logger.setLevel(logging.INFO)

    

def run_ppe_detection(client_id: str, video_url: str, camera_id: int, user_id: int, org_id: int, sessions: dict, loop: asyncio.AbstractEventLoop, storage_executor: ThreadPoolExecutor):
    """
    Runs PPE detection in a separate thread.
    Sends WebSocket messages safely and stores frames to S3/DB in background threads to avoid blocking inference.
    """
    cap = cv2.VideoCapture(video_url)
    frame_num = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
        ret, frame = cap.read()
        if not ret:
            # No more frames
            continue

        # Convert to RGB and PIL Image for YOLO input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        
        
        frame_num += 1
        try:
            # ---------------- PPE inference ----------------
            result, error, annotated_frame = ppe_detection(image_pil)
            ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            payload = {}

            ws = sessions[client_id]["ws"]

            if result and annotated_frame is not None:
                success, buffer = cv2.imencode(".jpg", annotated_frame)
                if not success:
                    continue

                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                payload = {
                    "frame_num": frame_num,
                    "user_id": user_id,
                    "camera_id": camera_id,
                    "org_id": org_id,
                    "time_stamp": ts,
                    "detections": result["detections"],
                    "annotated_frame": frame_base64,
                }

            # ---------------- WebSocket send ----------------
            
                if payload:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps(payload)),
                        loop
                    )
                elif error:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps(error)),
                        loop
                    )
                    break

                # ---------------- Background storage ----------------
                if payload:
                    # Store every 20th frame only
                    if frame_num % 20 == 0:
                        def store_frame(payload, frame_num):
                            try:
                                s3_url = upload_to_s3(payload["annotated_frame"], frame_num)
                                insert_ppe_frame(payload, s3_url)
                                logger.info(f"[{client_id}] Frame {frame_num} stored successfully")
                            except Exception as e:
                                logger.error(f"[{client_id}] Frame {frame_num}:  storage error -> {e}")

                        # Schedule storage in another thread so it doesn't block inference
                        storage_executor.submit(store_frame, payload, frame_num)

            else:
                if ws:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps({"success": False, "message": error})),
                        loop
                    )
                logger.warning(f"[{client_id}] Frame {frame_num}: No detections - {error}")
                break


        except Exception as e:
            print(f"[{client_id}] Frame {frame_num} pipeline error -> {e}")

    cap.release()
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] PPE Detection stopped and resources released")