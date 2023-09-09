import json
import typing as tp
from pathlib import Path


def pytest_addoption(parser) -> None:
    parser.addoption("--performance", action="store_true", help="run the performance test")


def get_mock_data(lang: tp.Optional[str] = "en") -> tp.Tuple[tp.List[str], tp.List[str]]:
    """
    Returns
    -------
    tuple of lists of str
        Hypotheses and references.
    """
    if lang not in ["ru", "en"]:
        raise ValueError("Support only 'ru' and 'en' languages")
    metrics_path = Path(__file__).parent / f"data/metrics_{lang}_input.json"
    with open(metrics_path, "r") as f:
        aligned_pairs = json.load(f)
    hyps = [item["hyp"] for item in aligned_pairs]
    refs = [item["ref"] for item in aligned_pairs]
    return hyps, refs
