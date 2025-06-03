import logging
import os
import time
from typing import List

import httpx
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import Counter, Histogram
from pydantic import BaseModel

router = APIRouter(tags=["Model"])

REMOTE_ML_SERVICE_URL = os.getenv("REMOTE_ML_SERVICE_URL")

# ==============================
# Prometheus 메트릭 정의
# ==============================
# 엔드포인트별 기본 메트릭
MODEL_REQUEST_COUNT = Counter(
    "model_api_requests_total",
    "Model 관련 API 요청 수",
    ["endpoint", "method", "status"],
)
MODEL_REQUEST_LATENCY = Histogram(
    "model_api_request_duration_seconds",
    "Model 관련 API 요청 처리 시간(초)",
    ["endpoint", "method"],
)
MODEL_ERROR_COUNT = Counter(
    "model_api_errors_total", "Model 관련 API 에러 발생 수", ["endpoint", "method"]
)
# ==============================


class InferenceRequest(BaseModel):
    inputs: List[List[float]]


@router.get("/", summary="루트 디렉토리")
async def read_root(request: Request):
    endpoint = "read_root"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    try:
        result = {"message": "MLflow FastAPI Model Serving Server is running 🚀"}
        status = "200"
        return result
    finally:
        duration = time.monotonic() - start_time
        MODEL_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        MODEL_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()


@router.get("/models/", summary="등록된 모델 목록 반환")
async def list_models(request: Request):
    endpoint = "list_models"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(remote_url)
            response.raise_for_status()
            data = response.json()
            status = str(response.status_code)
            return data

    except Exception as e:
        status = "500"
        MODEL_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing models: {e}")

    finally:
        duration = time.monotonic() - start_time
        MODEL_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        MODEL_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()


@router.get("/models/{model_name}/versions/", summary="특정 모델의 버전 목록 반환")
async def list_model_versions(model_name: str, request: Request):
    endpoint = "list_model_versions"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/{model_name}/versions/"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(remote_url)
            response.raise_for_status()
            data = response.json()
            status = str(response.status_code)
            return data

    except Exception as e:
        status = "500"
        MODEL_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Error listing model versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error listing model versions: {e}"
        )

    finally:
        duration = time.monotonic() - start_time
        MODEL_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        MODEL_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()


@router.post(
    "/models/predict/{model_name}/",
    include_in_schema=False,
    summary="버전 미지정 모델 예측",
)
async def predict_without_version(
    model_name: str, request: Request, body: InferenceRequest
):
    endpoint = "predict_without_version"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/predict/{model_name}/"
    try:
        payload = body.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(remote_url, json=payload)
            response.raise_for_status()
            data = response.json()
            status = str(response.status_code)
            return data

    except Exception as e:
        status = "500"
        MODEL_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        duration = time.monotonic() - start_time
        MODEL_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        MODEL_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()


@router.post(
    "/models/predict/{model_name}/{version}/",
    include_in_schema=False,
    summary="버전 지정 모델 예측",
)
async def predict_with_version(
    model_name: str, version: str, request: Request, body: InferenceRequest
):
    endpoint = "predict_with_version"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    remote_url = f"{REMOTE_ML_SERVICE_URL}/models/predict/{model_name}/{version}/"
    try:
        payload = body.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(remote_url, json=payload)
            response.raise_for_status()
            data = response.json()
            status = str(response.status_code)
            return data

    except Exception as e:
        status = "500"
        MODEL_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        duration = time.monotonic() - start_time
        MODEL_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        MODEL_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()
