from dataclasses import dataclass


DEFAULT_DATASETS = (
    "0004", "0007", "0027", "0178", "0182",
    "0231", "0234", "0235", "0283", "0317",
)


@dataclass(frozen=True)
class RTICLConfig:

    n_shot: int = 10
    fp_radius: int = 3
    fp_nbits: int = 4096
    n_folds: int = 10
    seeds: tuple[int, ...] = (0, 2, 4)
    thinking_level: str = "low"
    max_output_tokens: int = 8192


DEFAULT_CONFIG = RTICLConfig()
