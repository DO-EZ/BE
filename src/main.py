from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from routers import captcha
from typing import List
import os
import httpx
import logging

app = FastAPI()

# 원격 ML 서비스 URL
os.environ["REMOTE_ML_SERVICE_URL"] = "http://remote-mlflow-server" # 실제 서버 주소로 변경

# CORS 설정 (프론트와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 미들웨어 (캡차 통과 여부 등 저장용)
app.add_middleware(
    SessionMiddleware,
    secret_key="super-secret-key",
    max_age=3600,  # 세션 유지 시간 (초)
)

class InferenceRequest(BaseModel):
    inputs: List[List[float]]

@app.get("/")
async def read_root():
    return {"message": "MLflow FastAPI Model Serving Server is running 🚀"}

@app.get("/models/")
async def list_models():
    """
    등록된 모델 목록 반환
    """
    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(remote_url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {e}")

@app.get("/models/{model_name}/versions/")
async def list_model_versions(model_name: str):
    """
    특정 모델의 버전 목록 반환
    """
    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/{model_name}/versions/"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(remote_url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Error listing model versions: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing model versions: {e}")

@app.post("/models/predict/{model_name}/", include_in_schema=False)
async def predict_without_version(model_name: str, request: InferenceRequest):
    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/predict/{model_name}/"
    try:
        payload = request.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(remote_url, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/models/predict/{model_name}/{version}/", include_in_schema=False)
async def predict_with_version(model_name: str, version: str, request: InferenceRequest):
    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/predict/{model_name}/{version}/"
    try:
        payload = request.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(remote_url, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

app.include_router(captcha.router)
