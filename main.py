from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schema.captcha import CaptchaRequest, CaptchaResponse
from model.predict import mock_predict
from utils.id_gen import generate_captcha_id

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용으로 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

captcha_store = {}

@app.get("/captcha")
def get_captcha():
    from random import randint
    expected = str(randint(0, 9))  
    captcha_id = generate_captcha_id()
    captcha_store[captcha_id] = expected
    return {"id": captcha_id, "expected": expected}

@app.post("/predict", response_model=CaptchaResponse)
def predict_captcha(request: CaptchaRequest):
    expected = captcha_store.get(request.id)
    if not expected:
        return CaptchaResponse(passed=False, message="캡차 ID 없음")

    result = mock_predict(request.image, expected)

    return CaptchaResponse(
        passed=result,
        message="성공" if result else "실패"
    )

@app.get("/health")
def health():
    return {"status": "ok"}
