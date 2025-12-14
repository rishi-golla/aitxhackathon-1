"""
OSHA Vision Core Module.

Provides unified initialization and configuration for all DGX Spark optimizations.
"""

from src.core.initializer import (
    initialize_osha_vision,
    get_runtime_config,
    RuntimeConfig,
    OptimizationLevel
)

__all__ = [
    'initialize_osha_vision',
    'get_runtime_config',
    'RuntimeConfig',
    'OptimizationLevel'
]
