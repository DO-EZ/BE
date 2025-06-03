import io
import os
import zipfile

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(
    tags=["Image"],
)

SAVED_DIR = "static/images"  # CAPTCHA 이미지가 저장된 디렉터리


@router.get("/images", summary="이미지 파일 다운로드")
async def download_captcha_images_zip():
    if not os.path.isdir(SAVED_DIR):
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w") as zf:
            pass
        zbuf.seek(0)
        return StreamingResponse(
            zbuf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=captcha_images.zip"},
        )

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(SAVED_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {".png", ".jpg", ".jpeg", ".gif"}:
                continue
            full_path = os.path.join(SAVED_DIR, fname)
            # ZIP 내부에서의 이름은 그대로 파일명으로
            zf.write(full_path, arcname=fname)
    zbuf.seek(0)

    return StreamingResponse(
        zbuf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=captcha_images.zip"},
    )
