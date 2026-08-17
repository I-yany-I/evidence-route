"""pytest 公共 fixtures。

约定：测试不访问真实网络（LLM/搜索全部 mock），
需要临时目录用 tmp_path，需要样例数据放 tests/fixtures/。
"""
