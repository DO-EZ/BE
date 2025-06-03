import io
import logging
import os
import time
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Gauge, Histogram

router = APIRouter(
    tags=["Image"],
)

SAVED_DIR = "static/images"  # CAPTCHA 이미지가 저장된 디렉터리

# ==============================
# Prometheus 메트릭 정의
# ==============================
# 엔드포인트별 기본 메트릭
IMAGE_REQUEST_COUNT = Counter(
    "image_api_requests_total",
    "Image 관련 API 요청 수",
    ["endpoint", "method", "status"],
)
IMAGE_REQUEST_LATENCY = Histogram(
    "image_api_request_duration_seconds",
    "Image 관련 API 요청 처리 시간(초)",
    ["endpoint", "method"],
)
IMAGE_ERROR_COUNT = Counter(
    "image_api_errors_total", "Image 관련 API 에러 발생 수", ["endpoint", "method"]
)
# 이미지 다운로드 관련
CAPTCHA_IMAGES_DOWNLOAD = Counter(
    "captcha_images_download_total", "이미지 다운로드(zip) 요청 수"
)
CAPTCHA_IMAGES_ZIP_SIZE = Gauge(
    "captcha_images_zip_bytes", "생성된 ZIP 파일 크기(바이트)"
)
# ==============================


@router.get("/images", summary="이미지 파일 다운로드")
async def download_captcha_images_zip():
    endpoint = "download_images"
    method = "GET"
    start_time = time.monotonic()
    status = "200"  # ← 기본값

    try:
        CAPTCHA_IMAGES_DOWNLOAD.inc()

        if not os.path.isdir(SAVED_DIR):
            # 빈 ZIP 생성
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, mode="w") as zf:
                pass
            zbuf.seek(0)
            zip_size = len(zbuf.getvalue())
            CAPTCHA_IMAGES_ZIP_SIZE.set(zip_size)
            status = "200"
            return StreamingResponse(
                zbuf,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=captcha_images.zip"
                },
            )

        # 실제 이미지 압축
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(SAVED_DIR):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in {".png", ".jpg", ".jpeg", ".gif"}:
                    continue
                full_path = os.path.join(SAVED_DIR, fname)
                zf.write(full_path, arcname=fname)
        zbuf.seek(0)

        zip_size = len(zbuf.getvalue())
        CAPTCHA_IMAGES_ZIP_SIZE.set(zip_size)
        status = "200"
        return StreamingResponse(
            zbuf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=captcha_images.zip"},
        )

    except Exception as e:
        status = "500"
        IMAGE_ERROR_COUNT.labels(endpoint=endpoint, method=method).inc()
        logging.error(f"이미지 다운로드 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 오류 발생")

    finally:
        duration = time.monotonic() - start_time
        IMAGE_REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        IMAGE_REQUEST_COUNT.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()
