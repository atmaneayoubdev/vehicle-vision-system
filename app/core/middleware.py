from time import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.logging import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit_window: int, max_requests: int):
        super().__init__(app)
        self.rate_limit_window = rate_limit_window
        self.max_requests = max_requests
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        request_time = time()

        if client_ip in self.requests:
            # Filter out timestamps older than rate_limit_window
            self.requests[client_ip] = [timestamp for timestamp in self.requests[client_ip]
                                        if request_time - timestamp < self.rate_limit_window]

        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

        self.requests[client_ip].append(request_time)
        logger.info(
            f"Request allowed for IP: {client_ip} at time: {request_time}")
        response = await call_next(request)
        return response


class SecurityHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def setup_middleware(app):
    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_origins": ["*"],
        "allow_credentials": True,
    }
    app.add_middleware(CORSMiddleware, **cors_options)
    app.add_middleware(RateLimitMiddleware,
                       rate_limit_window=1, max_requests=2)
    app.add_middleware(SecurityHeaderMiddleware)
