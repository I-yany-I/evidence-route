"""Gradio 界面：声明输入 + 模式选择 + 终答/审计消息流/facts 表格。

启动：``python -m agent_collab.ui.app``
"""

from __future__ import annotations

import gradio as gr

from ..demo.fact_check_demo import run_fact_check_async


def _render_conversation(events: list[dict]) -> str:
    """把 audit.replay() 渲染成对话流（Markdown）。"""
    lines: list[str] = []
    for e in events:
        kind = e.get("kind", "")
        actor = e.get("actor", "")
        detail = e.get("detail", {}) or {}
        if kind == "pattern_start":
            lines.append(f"▶️ **模式开始**：{detail.get('query', '')}")
        elif kind == "message":
            lines.append(f"📨 **{actor}** → {detail.get('to', '')}：{detail.get('task', '')}")
        elif kind == "agent_start":
            lines.append(f"🤖 **{actor}** 开始：{detail.get('task', '')}")
        elif kind == "tool_call":
            lines.append(f"🔧 **{actor}** 调用 `{detail.get('tool', '')}`：`{detail.get('arguments', '')}`")
        elif kind == "tool_result":
            content = str(detail.get('content', ''))
            lines.append(f"📄 **{actor}** 工具结果：{content[:200]}")
        elif kind == "agent_result":
            lines.append(f"✅ **{actor}**：{detail.get('content', '')}")
        elif kind == "decision":
            lines.append(f"⚖️ **{actor}** 裁决：{detail}")
        elif kind == "pattern_end":
            lines.append(f"🏁 **终答**：{detail.get('answer', '')}")
    return "\n\n".join(lines) if lines else "（暂无审计事件）"


async def _run(query: str, pattern: str, offline: bool) -> tuple[str, str, list]:
    """运行一次核查，返回 (终答, 消息流 Markdown, facts 表格行)。"""
    result = await run_fact_check_async(query, pattern=pattern or None, offline=bool(offline))
    conversation = _render_conversation(result.audit.replay())
    facts_rows = [
        [f.claim, f.verdict, f.evidence, ", ".join(f.sources), f.by]
        for f in result.facts
    ]
    return result.answer, conversation, facts_rows


def build_app() -> gr.Blocks:
    """构造 Gradio Blocks 应用。"""
    with gr.Blocks(title="AgentCollab 事实核查") as app:
        gr.Markdown("# AgentCollab 事实核查演示")
        gr.Markdown("输入待核声明，选择协作模式，观察多智能体核查过程与审计消息流。")
        with gr.Row():
            query = gr.Textbox(
                label="待核声明", lines=2,
                placeholder="例如：OpenAI 于 2024 年 5 月发布多模态大模型 GPT-4o……",
            )
            pattern = gr.Dropdown(
                choices=["parallel", "pipeline", "supervisor", "debate"],
                value="parallel", label="协作模式",
            )
        with gr.Row():
            offline = gr.Checkbox(label="离线模式（样例数据检索，不联网）", value=True)
            run_btn = gr.Button("运行核查", variant="primary")
        with gr.Accordion("终答", open=True):
            answer = gr.Markdown()
        with gr.Accordion("审计消息流", open=True):
            conversation = gr.Markdown()
        with gr.Accordion("事实库", open=False):
            facts = gr.Dataframe(
                headers=["声明", "判定", "证据", "来源", "写入方"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
            )
        run_btn.click(
            fn=_run,
            inputs=[query, pattern, offline],
            outputs=[answer, conversation, facts],
        )
    return app


demo = build_app()

if __name__ == "__main__":
    demo.launch()
