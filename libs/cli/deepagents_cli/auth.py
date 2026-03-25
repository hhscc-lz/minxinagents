"""Token verification for single-sign-on integration.

登录系统生成 token 的规则：
    payload = "{phone}|{timestamp}"          # timestamp 为 Unix 秒
    sign    = HMAC-SHA256(payload, SECRET_KEY) 的前 16 位十六进制
    raw     = "{payload}.{sign}"
    token   = base64url(raw)                 # URL-safe base64，无 padding

示例（Python）：
    import base64, hashlib, hmac, time

    SECRET_KEY = "your-secret-key"
    phone      = "13800138000"
    payload    = f"{phone}|{int(time.time())}"
    sign       = hmac.new(SECRET_KEY.encode(), payload.encode(),
                          hashlib.sha256).hexdigest()[:16]
    token      = base64.urlsafe_b64encode(f"{payload}.{sign}".encode()).decode().rstrip("=")

环境变量：
    TOKEN_SECRET_KEY  共享密钥，双方保持一致
    TOKEN_MAX_AGE     token 有效期（秒），默认 300（5 分钟）
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time


_SECRET_KEY: str = os.environ.get("TOKEN_SECRET_KEY", "minxinagent-secret-key")
_MAX_AGE: int = int(os.environ.get("TOKEN_MAX_AGE", "300"))


class TokenError(Exception):
    """Token 验证失败的原因。"""


def verify_token(token: str) -> str:
    """验证 token 并返回手机号。

    Args:
        token: 登录系统生成的 URL-safe base64 token。

    Returns:
        验证通过的手机号（如 "13800138000"）。

    Raises:
        TokenError: token 不存在、格式错误、签名非法或已过期。
    """
    if not token:
        raise TokenError("未提供 token，请从系统入口进入")

    if not _SECRET_KEY:
        raise TokenError("服务器未配置 TOKEN_SECRET_KEY")

    # 1. base64 解码（兼容有无 padding）
    try:
        padding = 4 - len(token) % 4
        raw = base64.urlsafe_b64decode(token + "=" * (padding % 4)).decode()
    except Exception as e:
        raise TokenError("token 格式错误") from e

    # 2. 拆分 payload 和签名
    try:
        payload, sign = raw.rsplit(".", 1)
        phone, timestamp_str = payload.split("|")
    except ValueError as e:
        raise TokenError("token 结构非法") from e

    # 3. 验证签名（防伪造）
    expected = hmac.new(
        _SECRET_KEY.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()[:16]
    if not hmac.compare_digest(sign, expected):
        raise TokenError("token 签名非法")

    # 4. 验证时效（防重放）
    try:
        age = time.time() - int(timestamp_str)
    except ValueError as e:
        raise TokenError("token 时间戳非法") from e

    if age > _MAX_AGE:
        raise TokenError(f"token 已过期（有效期 {_MAX_AGE} 秒），请重新进入")

    if age < 0:
        raise TokenError("token 时间戳异常，请检查服务器时钟同步")

    return phone
