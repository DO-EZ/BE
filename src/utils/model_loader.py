# src/utils/model_loader.py

import mlflow.pytorch
import torch

# 서버 시작 시 1회만 로드
try:
    model = mlflow.pytorch.load_model("models:/HybridCNN/latest")
    model.eval()
except Exception as e:
    model = None
    print(f"[ERROR] 모델 로드 실패: {e}")

def predict_digit(input_tensor: torch.Tensor) -> int:
    if model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction
