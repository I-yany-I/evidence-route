"""patterns 组：四种协作模式（pipeline/parallel/supervisor/debate）。"""

from .base import AgentResult, CollaborationPattern, build_team, run_pattern
from .debate import DebatePattern
from .parallel import ParallelPattern
from .pipeline import PipelinePattern
from .supervisor import SupervisorPattern

__all__ = [
    "AgentResult",
    "CollaborationPattern",
    "build_team",
    "run_pattern",
    "PipelinePattern",
    "ParallelPattern",
    "SupervisorPattern",
    "DebatePattern",
]
