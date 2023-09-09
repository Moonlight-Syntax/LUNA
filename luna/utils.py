import typing as tp


def validate_batch(hyps: tp.List[str], refs: tp.List[str]) -> None:
    if not refs:
        # Reference-free metrics
        return
    if not hyps:
        raise ValueError("Hypotheses list is empty.")
    if len(hyps) != len(refs):
        raise ValueError("Hypotheses and references lists have different lengths.")
