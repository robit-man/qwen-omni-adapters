"""Lossless virtual context for bounded-context language models.

The package keeps source evidence and derived memory deliberately separate.
Raw text is append-only and recoverable; everything placed in a transformer
window is a disposable, provenance-carrying working set.
"""

from qwen_omni_adapters.virtual_memory.aggregation import (
    FrequencyAggregationBuilder,
    FrequencyAggregationResult,
    requested_frequency_count,
)
from qwen_omni_adapters.virtual_memory.chunking import ChunkDraft, StructureAwareChunker
from qwen_omni_adapters.virtual_memory.controller import (
    ControllerConfig,
    RecursiveMemoryController,
)
from qwen_omni_adapters.virtual_memory.embedding import HashingEmbedder, validate_embedding
from qwen_omni_adapters.virtual_memory.engine import (
    PreparedTurn,
    VirtualContextEngine,
    select_relevant_memories,
)
from qwen_omni_adapters.virtual_memory.extractor import (
    ExtractedCandidate,
    StructuredMemoryExtractor,
)
from qwen_omni_adapters.virtual_memory.hierarchy import MemoryHierarchy, ResidentPage
from qwen_omni_adapters.virtual_memory.models import (
    MEMORY_POLICIES,
    ControllerAction,
    EvidenceChunk,
    MemoryClass,
    MemoryPolicy,
    MemoryRecord,
    RetrievalHit,
    WorkingContext,
)
from qwen_omni_adapters.virtual_memory.packer import ContextBudget, WorkingContextPacker
from qwen_omni_adapters.virtual_memory.recurrent import (
    RecurrentConfig,
    RecurrentMemoryBuilder,
    RecurrentResult,
    RecurrentWriteRequest,
)
from qwen_omni_adapters.virtual_memory.retrieval import HybridRetriever, QueryPlan
from qwen_omni_adapters.virtual_memory.store import ImmutableEvidenceStore
from qwen_omni_adapters.virtual_memory.telemetry import MemoryOperation, TraceCollector
from qwen_omni_adapters.virtual_memory.tokenization import LlamaCppTokenCounter

__all__ = [
    "ChunkDraft",
    "ContextBudget",
    "ControllerAction",
    "ControllerConfig",
    "EvidenceChunk",
    "ExtractedCandidate",
    "FrequencyAggregationBuilder",
    "FrequencyAggregationResult",
    "HybridRetriever",
    "HashingEmbedder",
    "ImmutableEvidenceStore",
    "MEMORY_POLICIES",
    "MemoryClass",
    "MemoryHierarchy",
    "LlamaCppTokenCounter",
    "MemoryOperation",
    "MemoryPolicy",
    "MemoryRecord",
    "PreparedTurn",
    "QueryPlan",
    "RecurrentConfig",
    "RecurrentMemoryBuilder",
    "RecurrentResult",
    "RecurrentWriteRequest",
    "RecursiveMemoryController",
    "RetrievalHit",
    "ResidentPage",
    "StructureAwareChunker",
    "StructuredMemoryExtractor",
    "TraceCollector",
    "VirtualContextEngine",
    "WorkingContext",
    "WorkingContextPacker",
    "validate_embedding",
    "select_relevant_memories",
    "requested_frequency_count",
]
