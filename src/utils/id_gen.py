import uuid


def generate_captcha_id() -> str:
    return str(uuid.uuid4())
