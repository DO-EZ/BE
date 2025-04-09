import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Request

from schemas.captcha import CaptchaRequest, CaptchaResponse
from utils.id_gen import generate_captcha_id
from utils.image_processing import decode_image

router = APIRouter()

REMOTE_ML_SERVICE_URL = os.getenv("REMOTE_ML_SERVICE_URL")

# 메모리 기반 문제 저장소 (임시용)
captcha_store = {}


@router.get("/captcha")
def get_captcha():
    from random import randint

    captcha_id = generate_captcha_id()
    expected = str(randint(0, 9))
    captcha_store[captcha_id] = expected
    return {"id": captcha_id, "expected": expected}


@router.post("/predict")
async def predict(req: CaptchaRequest, request: Request):
    expected = captcha_store.get(req.id)
    if not expected:
        raise HTTPException(status_code=400, detail="유효하지 않은 캡차 ID입니다.")

    try:
        image_input = decode_image(req.image, captcha_id=req.id)
        payload = {"inputs": image_input.tolist()}

        # 원격 ML 서비스의 HybridCNN 모델 예측 API 호출
        remote_url = f"{REMOTE_ML_SERVICE_URL}/models/predict/HybridCNN/"
        async with httpx.AsyncClient() as client:
            response = await client.post(remote_url, json=payload)
            response.raise_for_status()
            result = response.json()

        predicted_digit = result["predictions"][0]
        passed = str(predicted_digit) == expected

        if passed:
            request.session["captcha_passed"] = True
            return CaptchaResponse(passed=True, message="✅ 통과")
        else:
            return CaptchaResponse(
                passed=False, message="❌ 실패 (예측값: " + str(predicted_digit) + ")"
            )
    except Exception:
        logging.error("서버 오류 발생", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 오류 발생")


@router.get("/check")
def check_captcha(request: Request):
    if request.session.get("captcha_passed"):
        return {"access": "granted"}
    else:
        return {"access": "denied"}
