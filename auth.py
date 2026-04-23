import jwt
import datetime
from fastapi import HTTPException, status
from config import get_security_setting

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=get_security_setting("jwt_expiration_minutes", 1440))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        get_security_setting("jwt_secret_key"),
        algorithm=get_security_setting("jwt_algorithm")
    )
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(
            token,
            get_security_setting("jwt_secret_key"),
            algorithms=[get_security_setting("jwt_algorithm")]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
