import logging
import time
from typing import List
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from src.utils.model_loader import predict_digit
from src.utils.image_processing import decode_image

router = APIRouter(tags=["Model"])

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

class CaptchaImageRequest(BaseModel):
    id: str
    image: str  # base64 인코딩된 이미지 문자열


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

# 내부에서 모델 로드 후 추론 수행
# 기존 prometheus 메트릭 유지
@router.post("/predict", summary="캡챠 이미지로 추론")
async def predict_image(request: Request, body: CaptchaImageRequest):
    endpoint = "predict_image"
    method = request.method
    start_time = time.monotonic()
    status = "200"

    try:
        tensor = decode_image(body.image, captcha_id=body.id)
        prediction = predict_digit(tensor)
        return {"id": body.id, "prediction": prediction}

    except Exception as e:
        status = "500"
        MODEL_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"추론 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")

    finally:
        duration = time.monotonic() - start_time
        MODEL_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        MODEL_REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
