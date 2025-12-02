"""
Validation Module - Compliance Testing Framework

Provides validation tools for:
- DeepStack fidelity testing (vs Leduc reference)
- Value network calibration (ECE metrics)
- Perception accuracy testing
- Memory stability testing

Ensures championship-grade quality.
"""

from .pipeline_validator import PipelineValidator, run_full_validation
from .fidelity_test import DeepStackFidelityTest
from .calibration import CalibrationTester

__all__ = [
    'PipelineValidator',
    'run_full_validation',
    'DeepStackFidelityTest',
    'CalibrationTester'
]
