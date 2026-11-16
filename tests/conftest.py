# Copyright (c) OpenMMLab. All rights reserved.


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: requires a CUDA device')
    config.addinivalue_line('markers', 'slow: real-weight or long-running case, excluded from the default suite')
