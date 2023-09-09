import typing as tp

import torch
from blanc import BlancHelp, BlancTune

from luna.base import Metrics


class BlancMetrics(Metrics):
    """
    Implementation of the metric introduced in the paper
    'Fill in the BLANC: Human-free quality estimation of document summaries', Vasilyev et.al.

    Parameters
    ----------
    model_name: str, default = None
        Should be a huggingface BertForMaskedLM model.
    device: str | torch.device, default = "cpu"
        Device where to run the model.
    inference_batch_size: int, default = 1
        Batch size for inference the model.
    finetune_batch_size: int, default = 1
        Batch size for finetuning the model. Relevant only for BLANC-tune.
    finetune_epochs: int, default = 10
        Number of finetuning epochs. Relevant only for BLANC-tune.
    random_seed: int, default = 0
        Random seed. Relevant only for BLANC-tune.
        random_seed = 0 means no fixed seed.
    type: str, default = "help"
        Can be either "help" or "tune". Depending on that the metric will be either
        BLANC-tune or BLANC-help. More on the difference see in the paper.
    show_progress_bar: bool, default = True
        Whether to show progress bar or not.
    Notes
    -----
    This is a reference-free summarization metric, the evaluate_example and
    evaluate_batch methods get as input hypotheses (summaries) that
    we want to evaluate, and documents that were summarized (inputs).

    Implementation is taken from https://github.com/PrimerAI/blanc
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: tp.Union[str, torch.device] = "cpu",
        inference_batch_size: int = 1,
        finetune_batch_size: int = 1,
        finetune_epochs: int = 10,
        random_seed: int = 0,
        type: str = "help",
        show_progress_bar: bool = True,
    ):
        if type == "help":
            self.metric = BlancHelp(
                model_name=model_name,
                device=device,
                inference_batch_size=inference_batch_size,
                show_progress_bar=show_progress_bar,
            )
        elif type == "tune":
            self.metric = BlancTune(
                model_name=model_name,
                device=device,
                inference_batch_size=inference_batch_size,
                finetune_batch_size=finetune_batch_size,
                finetune_epochs=finetune_epochs,
                random_seed=random_seed,
                show_progress_bar=show_progress_bar,
            )
        else:
            raise ValueError(f"Type can be only help or tune. Got type = {type}")

    def evaluate_example(self, hyp: str, inp: str) -> float:
        return self.metric.eval_summaries_for_docs(docs=[inp], doc_summaries=[[hyp]])[0][0]

    def evaluate_batch(self, hyps: tp.List[str], inps: tp.List[str]) -> tp.List[float]:
        return [
            score[0] for score in self.metric.eval_summaries_for_docs(docs=inps, doc_summaries=[[hyp] for hyp in hyps])
        ]

    def __repr__(self) -> str:
        return "BLANC"
