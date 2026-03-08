"""
SpineIDS Clinical ML - Utility Functions
=========================================
General-purpose helpers used across the pipeline:

    - Patient ID extraction and validation
    - Logging setup
    - Global random-seed initialisation
    - File I/O (pickle, JSON, NumPy)
    - Timing context manager
    - Display formatting helpers
"""

import json
import logging
import os
import pickle
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# =============================================================================
# Logging
# =============================================================================

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a named logger.

    Args:
        log_file: Optional path for a file handler.  Directory is created
                  automatically if it does not exist.
        level:    Logging level (default: INFO).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger('SpineIDS_Clinical_ML')
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Reproducibility
# =============================================================================

def set_random_seed(seed: int = 42) -> None:
    """
    Set the random seed for Python, NumPy, and (optionally) PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# =============================================================================
# Patient ID Utilities
# =============================================================================

def extract_real_pid(global_id: str) -> str:
    """
    Extract a normalised RealPID from a raw Global_ID string.

    The function handles both centre-prefixed IDs (e.g. ``Center1_STB_0001``)
    and bare IDs (e.g. ``STB_0001``).  The numeric component is zero-padded
    to four digits.

    Args:
        global_id: Raw identifier string.

    Returns:
        Normalised RealPID in the form ``{STB|BS|PS}_{0000}``.

    Raises:
        ValueError: If no valid disease-code / numeric pair is found.

    Examples:
        >>> extract_real_pid("Center1_STB_0001")
        'STB_0001'
        >>> extract_real_pid("Center2_BS_23")
        'BS_0023'
    """
    match = re.search(r'(STB|BS|PS)_(\d+)', global_id, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}_{match.group(2).zfill(4)}"
    raise ValueError(f"Cannot extract a valid RealPID from '{global_id}'")


def standardize_real_pid(real_pid: str) -> str:
    """
    Normalise a RealPID to ``{STB|BS|PS}_{0000}`` format.

    Args:
        real_pid: Input identifier.

    Returns:
        Standardised RealPID.

    Raises:
        ValueError: If the input does not match the expected pattern.
    """
    match = re.match(r'^(STB|BS|PS)_(\d+)$', real_pid, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}_{match.group(2).zfill(4)}"
    raise ValueError(f"Invalid RealPID format: '{real_pid}'")


def validate_real_pid_format(real_pid: str) -> bool:
    """
    Return True if *real_pid* conforms to the ``{STB|BS|PS}_{0000}`` schema.
    """
    return bool(re.match(r'^(STB|BS|PS)_\d{4}$', real_pid))


def label_to_name(label: int) -> str:
    """Map an integer class index to its disease abbreviation."""
    return {0: 'STB', 1: 'BS', 2: 'PS'}.get(label, 'Unknown')


def name_to_label(name: str) -> int:
    """Map a disease abbreviation to its integer class index."""
    return {'STB': 0, 'BS': 1, 'PS': 2}.get(name.upper(), -1)


# =============================================================================
# File I/O
# =============================================================================

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Serialise *obj* to a pickle file, creating parent directories as needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Deserialise and return the object stored at *filepath*."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Serialise *obj* to a JSON file.

    NumPy scalars, arrays, and ``pathlib.Path`` objects are converted
    automatically.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    def _convert(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return o

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, default=_convert, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load and return the JSON object stored at *filepath*."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_numpy(arr: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save a NumPy array to a ``.npy`` file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, arr)


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load and return a NumPy array from a ``.npy`` file."""
    return np.load(filepath)


# =============================================================================
# Display Helpers
# =============================================================================

def get_timestamp() -> str:
    """Return the current datetime as a compact string (``YYYYMMDD_HHMMSS``)."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_metrics_for_display(
    metrics: Dict[str, float],
    precision: int = 4,
) -> str:
    """Format a metrics dict as an indented, human-readable string."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{precision}f}")
        else:
            lines.append(f"  {key}: {value}")
    return '\n'.join(lines)


def format_metrics_with_ci(
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    precision: int = 4,
) -> str:
    """
    Return a formatted string combining a point estimate with its 95 % CI.

    Example output: ``0.8732 (95% CI: 0.8456-0.8987)``
    """
    return (
        f"{point_estimate:.{precision}f} "
        f"(95% CI: {ci_lower:.{precision}f}-{ci_upper:.{precision}f})"
    )


def create_config_snapshot(config_dict: Dict, output_path: Path) -> None:
    """Persist a timestamped configuration snapshot to *output_path*."""
    save_json({'timestamp': get_timestamp(), 'config': config_dict}, output_path)


def print_section_header(title: str, width: int = 60) -> None:
    """Print a prominent section separator to stdout."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subsection_header(title: str, width: int = 60) -> None:
    """Print a subsection separator to stdout."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


# =============================================================================
# Timing
# =============================================================================

class Timer:
    """Context manager for measuring and reporting elapsed time."""

    def __init__(self, name: str = "Operation"):
        self.name       = name
        self.start_time = None
        self.end_time   = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        elapsed = self.end_time - self.start_time
        print(f"[{self.name}] elapsed: {elapsed}")

    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
