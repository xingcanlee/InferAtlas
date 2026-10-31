def pytest_configure(config):
    config.addinivalue_line(
        "markers", "extract: real CPU/meta extraction using the external torch/vLLM environment"
    )
