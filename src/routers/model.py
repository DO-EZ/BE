import logging
import os
from typing import List

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

REMOTE_ML_SERVICE_URL = os.getenv("REMOTE_ML_SERVICE_URL")


class InferenceRequest(BaseModel):
    inputs: List[List[float]]


@router.get("/")
async def read_root():
    return {"message": "MLflow FastAPI Model Serving Server is running 🚀"}


@router.get("/models/")
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


@router.get("/models/{model_name}/versions/")
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
        raise HTTPException(
            status_code=500, detail=f"Error listing model versions: {e}"
        )


@router.post("/models/predict/{model_name}/", include_in_schema=False)
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


@router.post("/models/predict/{model_name}/{version}/", include_in_schema=False)
async def predict_with_version(
    model_name: str, version: str, request: InferenceRequest
):
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
