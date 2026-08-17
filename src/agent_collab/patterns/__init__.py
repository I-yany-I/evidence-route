"""patterns 组：五种协作模式（pipeline/parallel/supervisor/debate/factcheck）。"""

from .base import AgentResult, CollaborationPattern, build_team, run_pattern
from .debate import DebatePattern
from .factcheck import FactCheckPattern
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
    "FactCheckPattern",
]
