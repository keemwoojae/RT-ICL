from .config import DEFAULT_CONFIG, DEFAULT_DATASETS, RTICLConfig
from .data import Dataset, load_dataset, split_compounds
from .descriptors import compound_from_smiles
from .pipeline import (
    PipelineResult, prepare_cross_validation_requests, prepare_inference_requests,
    run_cross_validation, run_inference,
)
from .prompt import PromptPayload, build_prompt
from .providers import create_provider

__all__ = [
    "DEFAULT_CONFIG", "DEFAULT_DATASETS", "RTICLConfig", "Dataset",
    "load_dataset", "split_compounds", "compound_from_smiles", "PipelineResult",
    "prepare_cross_validation_requests", "prepare_inference_requests",
    "run_cross_validation", "run_inference", "load_compound_table",
    "prepare_compounds", "preprocess_dataset", "PromptPayload", "build_prompt",
    "create_provider",
]


def __getattr__(name: str):
    # Leave preprocessing unloaded until requested so its CLI can run with -m.
    if name in {"load_compound_table", "prepare_compounds", "preprocess_dataset"}:
        from . import preprocessing

        value = getattr(preprocessing, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
