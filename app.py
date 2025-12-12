from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException

from src.websocket.shoplifting_w_local1 import run_shoplifting_detection

from src.handlers.shoplifting_handler import shoplifting_websocket_handler

from fastapi.middleware.cors import CORSMiddleware


from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores ----------------
shoplifting_sessions = {}


detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- PPE WebSocket ----------------
@app.websocket("/ws/shoplifting/{client_id}")
async def websocket_ppe(ws: WebSocket,client_id: str):
    await shoplifting_websocket_handler(detection_executor, storage_executor, ws,client_id, shoplifting_sessions, run_shoplifting_detection, "shoplifting")



