"""Request ID tracking and structured logging middleware for FastAPI."""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api_access")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects and propagates a unique X-Request-ID header on all incoming requests."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract existing X-Request-ID or generate new unique identifier
        req_id = request.headers.get("X-Request-ID")
        if not req_id:
            req_id = f"req_{uuid.uuid4().hex[:12]}"

        request.state.request_id = req_id
        start_time = time.perf_counter()

        logger.info(
            f"[START] {request.method} {request.url.path} | RequestID={req_id} | Client={request.client.host if request.client else 'unknown'}"
        )

        try:
            response: Response = await call_next(request)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Process-Time"] = f"{elapsed_ms / 1000.0:.4f}s"

            logger.info(
                f"[FINISH] {request.method} {request.url.path} | Status={response.status_code} | "
                f"Duration={elapsed_ms}ms | RequestID={req_id}"
            )
            return response
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.error(
                f"[ERROR] {request.method} {request.url.path} | Exception={exc} | "
                f"Duration={elapsed_ms}ms | RequestID={req_id}"
            )
            raise
