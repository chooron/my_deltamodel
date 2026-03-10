"""
DiffBlendV1 模型测试套件 - __init__.py
"""

from .test_diff_blend_v1 import (
    test_model_instantiation,
    test_forward_pass,
    test_gradient_flow,
    test_gradient_with_snow,
    test_water_balance,
    test_weight_methods,
    test_nmul_configurations,
    run_all_tests,
)

__all__ = [
    'test_model_instantiation',
    'test_forward_pass',
    'test_gradient_flow',
    'test_gradient_with_snow',
    'test_water_balance',
    'test_weight_methods',
    'test_nmul_configurations',
    'run_all_tests',
]
