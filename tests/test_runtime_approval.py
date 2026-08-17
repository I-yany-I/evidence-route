"""runtime/approval.py 的单元测试。

覆盖 auto_approve 直通 / CLI y / Y / n（monkeypatch builtins.input）。
"""

import asyncio

from agent_collab.runtime import Approver


def test_auto_approve_returns_true():
    approver = Approver(auto_approve=True)
    assert asyncio.run(approver.request("python_repl", {})) is True


def test_cli_yes_returns_true(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    approver = Approver(auto_approve=False)
    assert asyncio.run(approver.request("python_repl", {})) is True


def test_cli_uppercase_y_returns_true(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "Y")
    approver = Approver()
    assert asyncio.run(approver.request("t", {})) is True


def test_cli_no_returns_false(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    approver = Approver()
    assert asyncio.run(approver.request("t", {})) is False
