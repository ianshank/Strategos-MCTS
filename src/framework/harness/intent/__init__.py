"""Intent capture: normalisers, SPEC parsers, the schema v2 validator, and SDD tooling."""

from src.framework.harness.intent.normalizer import DefaultIntentNormalizer
from src.framework.harness.intent.spec_loader import SpecCriterion, SpecLoader, SpecParseError
from src.framework.harness.intent.spec_scaffold import (
    SPEC_ID_PATTERN,
    SpecScaffoldError,
    modules_overlap,
    scaffold_spec,
)
from src.framework.harness.intent.spec_trace import TraceResult, VerifiedFlip, evaluate_trace, run_trace
from src.framework.harness.intent.spec_validator import (
    SPEC_STATUSES,
    SpecValidator,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "DefaultIntentNormalizer",
    "SPEC_ID_PATTERN",
    "SPEC_STATUSES",
    "SpecCriterion",
    "SpecLoader",
    "SpecParseError",
    "SpecScaffoldError",
    "SpecValidator",
    "TraceResult",
    "ValidationIssue",
    "ValidationReport",
    "VerifiedFlip",
    "evaluate_trace",
    "modules_overlap",
    "run_trace",
    "scaffold_spec",
]
