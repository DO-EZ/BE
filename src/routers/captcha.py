from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from schemas.captcha import CaptchaRequest, CaptchaResponse
from utils.id_gen import generate_captcha_id
from PIL import Image
from torchvision import transforms
from models.models import HybridCNN
import mlflow, mlflow.pytorch, base64, io, torch, os, traceback

router = APIRouter()

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

mlflow.set_tracking_uri("http://localhost:5001") # mlflow 주소

# 모델 불러오기 및 디바이스 이동
model = mlflow.pytorch.load_model("models:/HybridCNN@production")
model.to(device)
model.eval()

# 메모리 기반 문제 저장소 (임시용)
captcha_store = {}

# base64 이미지 데이터 중심 정렬 전처리
def center_image(image: Image.Image, padding: int = 20) -> Image.Image:
    import numpy as np
    from PIL import ImageOps

    img_array = np.array(image)
    if img_array.max() == 0:
        return image 

    img_array = (img_array < 200).astype(np.uint8)
    coords = np.argwhere(img_array)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image.crop((x0, y0, x1, y1))
    
    padded_image = ImageOps.expand(cropped, border=padding // 2, fill=255)
    max_dim = max(padded_image.size)
    squared_image = ImageOps.pad(padded_image, (max_dim, max_dim), color=255)

    return squared_image

#  base64 이미지 데이터 -> PyTorch 모델 입력용 텐서
def decode_image(image_base64: str, captcha_id: str = None) -> torch.Tensor:
    header, encoded = image_base64.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    original_image = Image.open(io.BytesIO(image_bytes)).convert("L")

    centered_image = center_image(original_image, padding=20)

    if captcha_id:
        save_dir = "saved_images"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"captcha_{captcha_id}.png"
        save_path = os.path.join(save_dir, filename)
        centered_image.save(save_path)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(centered_image).unsqueeze(0)  # shape: (1, 1, 28, 28)

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
        image_tensor = decode_image(req.image, captcha_id=req.id).to(device)
        prediction = model(image_tensor)
        predicted_digit = torch.argmax(prediction, dim=1).item()
        
        passed = str(predicted_digit) == expected
        if passed:
            request.session["captcha_passed"] = True
            return CaptchaResponse(passed=True, message="✅ 통과")
        else:
            return CaptchaResponse(passed=False, message="❌ 실패 (예측값: " + str(predicted_digit) + ")")
    except Exception as e:
        print("[서버 오류]", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "passed": False,
                "message": "서버 오류 발생",
                "detail": traceback.format_exc()
            }
        )

@router.get("/check")
def check_captcha(request: Request):
    if request.session.get("captcha_passed"):
        return {"access": "granted"}
    else:
        return {"access": "denied"}
