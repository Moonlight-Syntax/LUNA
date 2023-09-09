from luna.sources.rouge_we_utils import (
    _counter_overlap,
    _ngram_count,
    _ngram_counts,
    _safe_f1,
    pre_process_summary,
)


def rouge_n(hyp, references, n, alpha):
    """
    Compute the ROUGE-N score of a hypothesis with respect to one or more references, for
    a given value of `n`. Alpha is used to calculate resulting F-beta score. Here alpha = 1 / (1 + beta^2)
    """

    hyp = pre_process_summary(hyp, n)
    references = [pre_process_summary(model, n) for model in references]

    matches = 0
    recall_total = 0
    hyp_counter = _ngram_counts(hyp, n)
    for reference in references:
        ref_counter = _ngram_counts(reference, n)
        matches += _counter_overlap(hyp_counter, ref_counter)
        recall_total += _ngram_count(reference, n)
    precision_total = len(references) * _ngram_count(hyp, n)
    return _safe_f1(matches, recall_total, precision_total, alpha)
