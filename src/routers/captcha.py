from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from schemas.captcha import CaptchaRequest, CaptchaResponse
from utils.id_gen import generate_captcha_id
from PIL import Image
from torchvision import transforms
from models.models import SimpleCNN
import mlflow ,mlflow.pytorch
import base64
import io
import torch
import traceback

router = APIRouter()

mlflow.set_tracking_uri("http://localhost:5001") # mlflow 주소
model = mlflow.pytorch.load_model("models:/mnist-digit-captcha@production")

# 메모리 기반 문제 저장소 (임시용)
captcha_store = {}

def decode_image(image_base64: str) -> torch.Tensor:
    header, encoded = image_base64.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # shape: (1, 1, 28, 28)

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
        return JSONResponse(status_code=400, content={"passed": False, "message": "캡차 ID 없음"})

    try:
        image_tensor = decode_image(req.image)
        print(f"[이미지] shape: {image_tensor.shape}") #######
        prediction = model(image_tensor)
        predicted_digit = torch.argmax(prediction, dim=1).item()
        print(f"[예측값] {predicted_digit} / 정답: {expected}") #########
        
        passed = str(predicted_digit) == expected
        if passed:
            request.session["captcha_passed"] = True
            return CaptchaResponse(passed=True, message="✅ 통과")
        else:
            return CaptchaResponse(passed=False, message="❌ 실패 (예측값: " + str(predicted_digit) + ")")
    except Exception as e:
        return JSONResponse(status_code=500,content={"passed": False,"message": "서버 오류 발생","detail": traceback.format_exc()})

@router.get("/check")
def check_captcha(request: Request):
    if request.session.get("captcha_passed"):
        return {"access": "granted"}
    else:
        return {"access": "denied"}
