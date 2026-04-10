from __future__ import annotations

from fastapi import FastAPI

from .anthropic import install_anthropic_routes
from .context import ProxyRouteContext
from .gemini import install_gemini_routes
from .openai import install_openai_routes


def install_proxy_api_routes(application: FastAPI) -> None:
    context = ProxyRouteContext.from_application(application)
    install_gemini_routes(context)
    install_openai_routes(context)
    install_anthropic_routes(context)
