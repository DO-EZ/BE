from pydantic import BaseModel


class CaptchaRequest(BaseModel):
    image: str  # base64 encoded image
    id: str  # captcha id


class CaptchaResponse(BaseModel):
    passed: bool
    message: str = ""
