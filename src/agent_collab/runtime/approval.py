"""人工审批器：CLI 交互或自动放行。"""

from __future__ import annotations


class Approver:
    """审批器。auto_approve=True 直接放行（演示/评测模式）；否则 CLI 询问 y/n。

    audit 可选：提供时把每次审批决策记录为 ``approval`` 审计事件。
    """

    def __init__(self, auto_approve: bool = False, audit: object | None = None):
        self.auto_approve = auto_approve
        self.audit = audit

    async def request(self, tool_name: str, args: dict) -> bool:
        """审批一次工具调用；返回是否放行。"""
        if self.auto_approve:
            return True
        answer = input(f"允许执行工具 {tool_name}？(y/n) ")
        approved = answer.strip().lower() == "y"
        if self.audit is not None:
            self.audit.record("approval", "user", {
                "tool": tool_name, "arguments": args, "approved": approved,
            })
        return approved
