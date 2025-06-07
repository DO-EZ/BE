import base64
import io
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from routers.captcha import captcha_store


def center_image(image: Image.Image, padding: int = 20) -> Image.Image:
    img_array = np.array(image)
    if img_array.max() == 0:
        return image

    binarized = (img_array < 200).astype(np.uint8)
    coords = np.argwhere(binarized)
    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image.crop((x0, y0, x1, y1))

    padded_image = ImageOps.expand(cropped, border=padding // 2, fill=255)
    max_dim = max(padded_image.size)
    squared_image = ImageOps.pad(padded_image, (max_dim, max_dim), color=255)
    return squared_image


def decode_image(
    image_base64: str, captcha_id: str = None, label: str = None
) -> torch.Tensor:
    try:
        # 기대하는 포맷: "data:image/png;base64,...."
        header, encoded = image_base64.split(",", 1)
    except ValueError:
        raise ValueError(
            "올바른 이미지 포맷이 아닙니다. header와 데이터 구분자(',')를 확인하세요."
        )

    try:
        image_bytes = base64.b64decode(encoded)
    except base64.binascii.Error:
        raise ValueError("base64 디코딩에 실패했습니다.")

    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as e:
        raise ValueError("이미지 처리 중 오류 발생: " + str(e))

    centered_image = center_image(original_image, padding=20)

    if captcha_id:
        save_dir = "static/images"
        os.makedirs(save_dir, exist_ok=True)
        label = captcha_store[captcha_id]
        filename = f"captcha_{captcha_id}_{label}.png"
        centered_image.save(os.path.join(save_dir, filename))

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tensor_image = transform(centered_image).unsqueeze(0)
    return tensor_image
