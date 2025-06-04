import json
import logging
import os
import time

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram

from schemas.captcha import CaptchaRequest, CaptchaResponse
from utils.id_gen import generate_captcha_id
from utils.image_processing import decode_image

router = APIRouter(
    tags=["Captcha"],
)

REMOTE_ML_SERVICE_URL = os.getenv("REMOTE_ML_SERVICE_URL")
print(f"🔍 REMOTE_ML_SERVICE_URL 환경변수: {REMOTE_ML_SERVICE_URL}")

# 메모리 기반 문제 저장소 (임시용)
captcha_store = {}

# ==============================
# Prometheus 메트릭 정의
# ==============================
# 엔드포인트별 기본 메트릭
CAPTCHA_REQUEST_COUNT = Counter(
    "captcha_api_requests_total",
    "Captcha 관련 API 요청 수",
    ["endpoint", "method", "status"],
)
CAPTCHA_REQUEST_LATENCY = Histogram(
    "captcha_api_request_duration_seconds",
    "Captcha 관련 API 요청 처리 시간(초)",
    ["endpoint", "method"],
)
CAPTCHA_ERROR_COUNT = Counter(
    "captcha_api_errors_total", "Captcha 관련 API 에러 발생 수", ["endpoint", "method"]
)

# 캡차 생성/검증 관련
CAPTCHA_GENERATED = Counter("captcha_generated_total", "생성된 캡차 총 개수")
CAPTCHA_VERIFY_SUCCESS = Counter(
    "captcha_verify_success_total", "성공적으로 통과한 캡차 개수"
)
CAPTCHA_VERIFY_FAILURE = Counter(
    "captcha_verify_failure_total", "실패한 캡차 인증 개수"
)
CAPTCHA_INVALID_ID = Counter(
    "captcha_invalid_id_total", "유효하지 않은 캡차 ID로 요청한 횟수"
)

# 원격 ML 호출 관련
REMOTE_ML_LATENCY = Histogram(
    "remote_ml_request_duration_seconds",
    "원격 ML 서비스 호출 지연 시간(초)",
    ["model", "method"],
)
REMOTE_ML_ERRORS = Counter(
    "remote_ml_errors_total", "원격 ML 호출 실패 횟수", ["model", "method", "status"]
)
REMOTE_ML_PAYLOAD_SIZE = Gauge(
    "remote_ml_payload_bytes", "원격 ML 서비스로 전송된 페이로드 바이트 크기", ["model"]
)

# 요청/응답 크기
REQUEST_PAYLOAD_SIZE = Gauge(
    "api_request_payload_bytes", "API 요청 페이로드 크기(바이트)", ["endpoint"]
)
RESPONSE_SIZE_BYTES = Gauge(
    "api_response_size_bytes", "API 응답 바디 크기(바이트)", ["endpoint"]
)
# ==============================


@router.get("/captcha", summary="숫자 랜덤 생성")
def get_captcha(request: Request):
    endpoint = "get_captcha"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    try:
        from random import randint

        captcha_id = generate_captcha_id()
        expected = str(randint(0, 9))
        captcha_store[captcha_id] = expected

        CAPTCHA_GENERATED.inc()  # 생성 개수
        status = "200"
        return {"id": captcha_id, "expected": expected}

    except Exception as e:
        status = "500"
        CAPTCHA_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Captcha 생성 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 오류 발생")

    finally:
        duration = time.monotonic() - start_time
        CAPTCHA_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(
            duration
        )
        CAPTCHA_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()


@router.post("/predict", summary="유저 입력 이미지 추론")
async def predict(req: CaptchaRequest, request: Request):
    endpoint = "predict_captcha"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    expected = captcha_store.get(req.id)
    if not expected:
        status = "400"
        CAPTCHA_INVALID_ID.inc()
        CAPTCHA_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        # 잘못된 캡차 ID 요청이므로 400으로 응답, 카운터만 증가
        duration = time.monotonic() - start_time
        CAPTCHA_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(
            duration
        )
        CAPTCHA_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()
        raise HTTPException(status_code=400, detail="유효하지 않은 캡차 ID입니다.")

    try:

        image_input = decode_image(req.image, captcha_id=req.id)
        headers = {"Content-Type": "application/json"}
        payload = {"inputs": image_input.tolist()}

        # 요청 페이로드 크기 기록
        payload_size = len(json.dumps(payload).encode("utf-8"))
        REQUEST_PAYLOAD_SIZE.labels(endpoint=endpoint).set(payload_size)

        # 원격 ML 서비스의 HybridCNN 모델 예측 API 호출
        remote_url = f"{REMOTE_ML_SERVICE_URL}/invocations"
        async with httpx.AsyncClient() as client:
            ml_start = time.monotonic()
            response = await client.post(remote_url, json=payload, headers=headers)
            ml_duration = time.monotonic() - ml_start

            REMOTE_ML_LATENCY.labels(model="HybridCNN", method=method).observe(
                ml_duration
            )
            REMOTE_ML_PAYLOAD_SIZE.labels(model="HybridCNN").set(payload_size)

            response.raise_for_status()
            result = response.json()

            if response.status_code != 200:
                REMOTE_ML_ERRORS.labels(
                    model="HybridCNN", method=method, status=str(response.status_code)
                ).inc()

        logits = result["predictions"][0]
        predicted_digit = int(np.argmax(logits))
        passed = str(predicted_digit) == expected

        status = str(response.status_code)
        if passed:
            request.session["captcha_passed"] = True
            CAPTCHA_VERIFY_SUCCESS.inc()
            return CaptchaResponse(passed=True, message="✅ 통과")
        else:
            CAPTCHA_VERIFY_FAILURE.inc()
            return CaptchaResponse(
                passed=False, message="❌ 실패 (예측값: " + str(predicted_digit) + ")"
            )

    except HTTPException:
        # 이미 위에서 HTTPException을 발생시킬 때 처리했으므로, 여기로는 오지 않음
        raise

    except Exception as e:
        status = "500"
        CAPTCHA_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Captcha 예측 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 오류 발생")

    finally:
        duration = time.monotonic() - start_time
        CAPTCHA_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(
            duration
        )
        CAPTCHA_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()


@router.get("/check", summary="이미지 가부 결정")
def check_captcha(request: Request):
    endpoint = "check_captcha"
    method = request.method
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    try:
        passed_flag = request.session.get("captcha_passed", False)
        status = "200"
        if passed_flag:
            return {"access": "granted"}
        else:
            return {"access": "denied"}

    except Exception as e:
        status = "500"
        CAPTCHA_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"Captcha 체크 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 오류 발생")

    finally:
        duration = time.monotonic() - start_time
        CAPTCHA_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(
            duration
        )
        CAPTCHA_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()
