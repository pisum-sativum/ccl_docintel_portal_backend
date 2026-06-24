"""
JWT Authentication Engine for CCL DocIntel.
Provides token creation, verification, and FastAPI dependency injectors.
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ── Secrets & Config ────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ccl-docintel-super-secret-key-change-in-prod")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

# ── Password Hashing ─────────────────────────────────────────────────────────
# Using sha256_crypt for full compatibility with Anaconda's bcrypt environment
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ── JWT Token Creation ───────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ── FastAPI OAuth2 Scheme (reads Bearer token from Authorization header) ─────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ── Dependencies ─────────────────────────────────────────────────────────────
from fastapi import Request

def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    """
    Decodes the JWT and returns the payload dict.
    Raises HTTP 401 if token is missing or invalid.
    """
    if not token:
        token = request.query_params.get("token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_admin(current_user: dict = Depends(get_current_user)):
    """
    Dependency that additionally requires the user to have the 'admin' role.
    Raises HTTP 403 Forbidden for any non-admin user.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. This action requires Administrator privileges.",
        )
    return current_user
