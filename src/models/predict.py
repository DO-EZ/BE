import random


# 추후 모델 연결 가능하게 구성
def mock_predict(image_base64: str, expected: str) -> bool:
    # 지금은 50% 확률로 통과/실패 시뮬레이션
    return random.random() < 0.5
