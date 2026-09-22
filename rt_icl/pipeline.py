import asyncio
import hashlib
import json
import operator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sklearn.model_selection import KFold

from .config import RTICLConfig, DEFAULT_CONFIG
from .evaluation import build_metrics, check_result_conflicts, save_results
from .prompt import COMPOUND_KEYS, PromptPayload, build_prompt
from .providers import Provider, ProviderResponse
from .retrieval import select_tanimoto_examples
from .utils import as_finite_float, parse_rt


@dataclass(frozen=True)
class PreparedRequest:
    """One query, its selected demonstrations, and its immutable prompt payload."""

    index: int
    source_index: int
    query: dict[str, Any]
    examples: tuple[dict[str, Any], ...]
    prompt: PromptPayload
    actual_rt: float | None = None
    fold: int | None = None
    reference_rt_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class PipelineResult:
    """In-memory result plus the optional directory in which it was saved."""

    config: dict[str, Any]
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    output_dir: Path | None = None


def prepare_inference_requests(
    references: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    lc_condition: Mapping[str, Any],
    *,
    config: RTICLConfig = DEFAULT_CONFIG,
) -> list[PreparedRequest]:
    reference_rows = _normalize_compounds(references, require_rt=True)
    query_rows = _normalize_compounds(queries, require_rt=False)
    if not reference_rows:
        raise ValueError("At least one labeled reference compound is required")
    return _prepare_requests(
        reference_rows,
        query_rows,
        lc_condition,
        config=config,
        fold=None,
        source_indices=range(len(query_rows)),
    )


def prepare_cross_validation_requests(
    compounds: Sequence[Mapping[str, Any]],
    lc_condition: Mapping[str, Any],
    *,
    config: RTICLConfig = DEFAULT_CONFIG,
    seed: int | None = None,
    n_folds: int | None = None,
    folds: Iterable[int] | None = None,
) -> list[PreparedRequest]:
    rows = _normalize_compounds(compounds, require_rt=True)
    if not rows:
        raise ValueError("Cannot evaluate an empty compound dataset")
    resolved_folds = n_folds if n_folds is not None else config.n_folds
    if resolved_folds != config.n_folds:
        raise ValueError(
            f"RT-ICL evaluation is fixed to {config.n_folds} folds, got {resolved_folds}"
        )
    resolved_seed = seed if seed is not None else config.seeds[0]
    if resolved_seed not in config.seeds:
        raise ValueError(
            f"RT-ICL evaluation seed must be one of {config.seeds}, got {resolved_seed}"
        )
    requested_folds = (
        list(range(resolved_folds)) if folds is None else sorted({operator.index(value) for value in folds})
    )
    invalid = [fold for fold in requested_folds if not 0 <= fold < resolved_folds]
    if invalid:
        raise ValueError(
            f"Fold index out of range for n_folds={resolved_folds}: {invalid}"
        )

    splitter = KFold(n_splits=resolved_folds, shuffle=True, random_state=resolved_seed)
    prepared: list[PreparedRequest] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(rows)):
        if fold not in requested_folds:
            continue
        references = [rows[int(index)] for index in train_indices]
        queries = [rows[int(index)] for index in test_indices]
        prepared.extend(
            _prepare_requests(
                references,
                queries,
                lc_condition,
                config=config,
                fold=fold,
                source_indices=(int(index) for index in test_indices),
            )
        )
    return prepared


async def execute_requests(
    requests: Sequence[PreparedRequest],
    provider: Provider,
    *,
    max_concurrent: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    max_concurrent = operator.index(max_concurrent)
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")
    ensure_available = getattr(provider, "ensure_available", None)
    if callable(ensure_available):
        ensure_available()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(request: PreparedRequest) -> dict[str, Any]:
        try:
            async with semaphore:
                response = await provider.generate(request.prompt)
        except Exception as exc:
            response = ProviderResponse(text=None, error=f"{type(exc).__name__}: {exc}")

        parsed = parse_rt(response.text if response.completed else None)
        predicted_rt = parsed.value
        actual_rt = request.actual_rt
        return {
            "fold": request.fold,
            "index": request.index,
            "source_index": request.source_index,
            "SMILES": str(request.query["SMILES"]).strip(),
            "actual_rt": actual_rt,
            "pred_rt": predicted_rt,
            "abs_error": (
                abs(predicted_rt - actual_rt)
                if predicted_rt is not None and actual_rt is not None else None
            ),
            "parse_status": (
                "incomplete" if not response.completed
                else "api_error" if response.text is None else parsed.status
            ),
            "response": response.text,
            "error": response.error,
            "prompt_sha256": request.prompt.sha256,
            "reference_rt_range": (
                list(request.reference_rt_range) if request.reference_rt_range is not None else None
            ),
            "examples": [
                {
                    "SMILES": str(example["SMILES"]).strip(),
                    "RT": as_finite_float(example.get("RT")),
                    **({"Tanimoto": example["Tanimoto"]} if "Tanimoto" in example else {}),
                }
                for example in request.examples
            ],
            "usage": response.usage,
            "response_metadata": response.metadata,
        }

    started = datetime.now(timezone.utc)
    predictions = await asyncio.gather(*(run_one(request) for request in requests))
    ended = datetime.now(timezone.utc)
    token_usage = {
        key: sum(int(row["usage"].get(key, 0) or 0) for row in predictions)
        for key in ("input_tokens", "output_tokens", "thinking_tokens")
    }
    timing = {
        "started_at": started.isoformat(),
        "finished_at": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
    }
    return predictions, token_usage, timing


async def run_inference(
    references: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    lc_condition: Mapping[str, Any],
    provider: Provider,
    *,
    config: RTICLConfig = DEFAULT_CONFIG,
    max_concurrent: int = 40,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    if output_dir is not None:
        check_result_conflicts(output_dir)
    requests = prepare_inference_requests(references, queries, lc_condition, config=config)
    return await _run(
        requests, provider, config=config,
        data={"references": list(references), "queries": list(queries), "lc_condition": lc_condition},
        run_info={"mode": "inference", "reference_count": len(references), "query_count": len(queries)},
        max_concurrent=max_concurrent, output_dir=output_dir,
    )


async def run_cross_validation(
    compounds: Sequence[Mapping[str, Any]],
    lc_condition: Mapping[str, Any],
    provider: Provider,
    *,
    dataset_id: str | None = None,
    config: RTICLConfig = DEFAULT_CONFIG,
    seed: int | None = None,
    n_folds: int | None = None,
    max_concurrent: int = 40,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    if output_dir is not None:
        check_result_conflicts(output_dir)
    resolved_seed = seed if seed is not None else config.seeds[0]
    resolved_folds = n_folds if n_folds is not None else config.n_folds
    requests = prepare_cross_validation_requests(
        compounds, lc_condition, config=config, seed=resolved_seed, n_folds=resolved_folds,
    )
    return await _run(
        requests, provider, config=config,
        data={"compounds": list(compounds), "lc_condition": lc_condition},
        run_info={
            "mode": "cross_validation",
            **({"dataset_id": dataset_id} if dataset_id is not None else {}),
            "compound_count": len(compounds),
            "split": {
                "method": "KFold", "shuffle": True,
                "n_folds": resolved_folds, "seed": resolved_seed,
            },
        },
        max_concurrent=max_concurrent, output_dir=output_dir,
    )


async def _run(
    requests: Sequence[PreparedRequest],
    provider: Provider,
    *,
    config: RTICLConfig,
    data: Mapping[str, Any],
    run_info: dict[str, Any],
    max_concurrent: int,
    output_dir: str | Path | None,
) -> PipelineResult:
    provider_config = provider.metadata()
    for provider_key, config_key in (
        ("max_output_tokens", "max_output_tokens"),
        ("thinking_level", "thinking_level"),
        ("reasoning_effort", "thinking_level"),
    ):
        if provider_key not in provider_config:
            continue
        actual = provider_config[provider_key]
        expected = getattr(config, config_key)
        if provider_key == "reasoning_effort" and actual is None and expected == "low":
            continue
        if actual != expected:
            raise ValueError(
                f"config.{config_key}={expected!r} conflicts with "
                f"provider {provider_key}={actual!r}; "
                "pass the same config to create_provider() and run_inference()/run_cross_validation()"
            )
    run_config = {
        **run_info,
        "rt_icl_config": asdict(config),
        "provider": provider_config,
        "max_concurrent": operator.index(max_concurrent),
        "prompt_sha256": hash_prepared_prompts(requests),
        "data_sha256": hash_data(data),
    }
    predictions, usage, timing = await execute_requests(requests, provider, max_concurrent=max_concurrent)
    metrics = build_metrics(
        predictions, include_folds=run_info["mode"] == "cross_validation",
        token_usage=usage, timing=timing,
    )
    saved_path = None
    if output_dir is not None:
        saved_path = save_results(
            output_dir, config=run_config, metrics=metrics,
            predictions=predictions,
        )
    return PipelineResult(run_config, metrics, predictions, saved_path)


def hash_data(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_prepared_prompts(requests: Sequence[PreparedRequest]) -> str:
    manifest = [
        {
            "fold": request.fold,
            "source_index": request.source_index,
            "prompt_sha256": request.prompt.sha256,
        }
        for request in requests
    ]
    return hash_data(manifest)


def _prepare_requests(
    references: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
    lc_condition: Mapping[str, Any],
    *,
    config: RTICLConfig,
    fold: int | None,
    source_indices: Iterable[int],
) -> list[PreparedRequest]:
    prepared: list[PreparedRequest] = []
    for local_index, (original_query, source_index) in enumerate(
        zip(queries, source_indices, strict=True)
    ):
        query = {key: value for key, value in original_query.items() if str(key).casefold() != "rt"}
        # Filter once so retrieval and RT bounds use the same eligible pool.
        query_smiles = str(query["SMILES"]).strip()
        candidates = [
            reference
            for reference in references
            if str(reference["SMILES"]).strip() != query_smiles
        ]
        if len(candidates) < config.n_shot:
            raise ValueError(
                f"Query {source_index} has only {len(candidates)} eligible references; "
                f"RT-ICL requires {config.n_shot}"
            )
        examples = select_tanimoto_examples(
            query, candidates, k=config.n_shot,
            fp_radius=config.fp_radius, fp_nbits=config.fp_nbits,
        )
        # Observed RT range of the retrieval pool (train fold in CV), never the query.
        candidate_rts = [as_finite_float(candidate["RT"]) for candidate in candidates]
        reference_rt_range = (min(candidate_rts), max(candidate_rts))
        payload = build_prompt(
            query,
            examples,
            dict(lc_condition),
            reference_rt_range=reference_rt_range,
        )
        prepared.append(
            PreparedRequest(
                index=local_index,
                source_index=source_index,
                query=query,
                examples=tuple(dict(example) for example in examples),
                prompt=payload,
                actual_rt=as_finite_float(original_query.get("RT")),
                fold=fold,
                reference_rt_range=reference_rt_range,
            )
        )
    return prepared


def _normalize_compounds(
    compounds: Sequence[Mapping[str, Any]],
    *,
    require_rt: bool,
) -> list[dict[str, Any]]:
    normalized = []
    label = "Reference" if require_rt else "Query"
    for index, compound in enumerate(compounds):
        row = dict(compound)
        for key in ("SMILES", "RT"):
            if key not in row and key.lower() in row:
                row[key] = row.pop(key.lower())
        missing = [key for key in COMPOUND_KEYS if key != "Name" and row.get(key) is None]
        if missing:
            raise ValueError(f"{label} compound {index} is missing RT-ICL prompt fields: {', '.join(missing)}")
        if not str(row["SMILES"]).strip():
            raise ValueError(f"{label} compound {index} must contain a non-empty SMILES field")
        if require_rt and as_finite_float(row.get("RT")) is None:
            raise ValueError(f"Reference compound {index} has no finite RT value")
        normalized.append(row)
    return normalized


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "PipelineResult",
    "PreparedRequest",
    "execute_requests",
    "hash_data",
    "hash_prepared_prompts",
    "prepare_cross_validation_requests",
    "prepare_inference_requests",
    "run_cross_validation",
    "run_inference",
]
