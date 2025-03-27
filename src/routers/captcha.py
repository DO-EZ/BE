from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from models.predict import mock_predict
from schemas.captcha import CaptchaRequest, CaptchaResponse
from utils.id_gen import generate_captcha_id

router = APIRouter()

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
def predict(req: CaptchaRequest, request: Request):
    expected = captcha_store.get(req.id)
    if not expected:
        return JSONResponse(
            status_code=400, content={"passed": False, "message": "캡차 ID 없음"}
        )

    passed = mock_predict(req.image, expected)
    if passed:
        request.session["captcha_passed"] = True
        return CaptchaResponse(passed=True, message="✅ 통과")
    else:
        return CaptchaResponse(passed=False, message="❌ 실패")


@router.get("/check")
def check_captcha(request: Request):
    if request.session.get("captcha_passed"):
        return {"access": "granted"}
    else:
        return {"access": "denied"}
