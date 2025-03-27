from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from routers import captcha

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key="super-secret-key",
    max_age=3600,  # 세션 유지 시간 (초)
)

app.include_router(captcha.router)
