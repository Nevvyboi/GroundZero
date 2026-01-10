"""
GroundZero API Module
=====================
FastAPI-based REST API for the GroundZero AI system.
"""

from .server import app, create_app, get_components, run_server
from .routes import router

__all__ = [
    'app',
    'create_app',
    'get_components',
    'run_server',
    'router'
]
