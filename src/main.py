import os
from pathlib import Path

# 원격 ML 서비스 URL
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette_exporter import PrometheusMiddleware, handle_metrics

from routers import captcha, image_dataset, model

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
app = FastAPI()

print("REMOTE_ML_SERVICE_URL =", os.getenv("REMOTE_ML_SERVICE_URL"))

# CORS 설정 (프론트와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus 미들웨어
app.add_middleware(
    PrometheusMiddleware,
    app_name="captcha",
    prefix="captcha_",
)

# 세션 미들웨어 (캡차 통과 여부 등 저장용)
app.add_middleware(
    SessionMiddleware,
    secret_key="super-secret-key",
    max_age=3600,  # 세션 유지 시간 (초)
)

# Prometheus가 스크랩할 수 있도록 핸들러 연결
app.add_route("/metrics", handle_metrics)

# 라우터 등록
app.include_router(captcha.router)
app.include_router(model.router)
app.include_router(image_dataset.router)
