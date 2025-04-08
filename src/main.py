from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from routers import captcha

app = FastAPI()

# CORS 설정 (프론트와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 미들웨어 (캡차 통과 여부 등 저장용)
app.add_middleware(
    SessionMiddleware,
    secret_key="super-secret-key",
    max_age=3600,  # 세션 유지 시간 (초)
)

app.include_router(captcha.router)
