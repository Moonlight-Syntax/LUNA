"""
The implementation references to the repository:
https://github.com/ThomasScialom/summa-qa
"""


import typing as tp
import warnings

import spacy
import torch
from sklearn import metrics
from transformers import BertForQuestionAnswering, BertTokenizer

from luna.base import Metrics


class Const:
    LANG_TO_MODEL = {"ru": "ru_core_news_md", "en": "en_core_web_sm"}


class QuestionGenerator:
    """
    Question generator.
    One question is a sentence with masked token.
    To answer the question predict the masked token.

    Attributes
    ----------
    model: spacy model object
        Loads a model from a spacy library. Pass spacy_model_name argument.
    """

    MASKED_TOKEN = "[MASK]"

    def __init__(self, lang: tp.Optional[str] = "ru") -> None:
        spacy_path = Const.LANG_TO_MODEL[lang]
        self.model = spacy.load(spacy_path)

    def generate(self, paragraph: str) -> tp.Tuple[tp.List[str], tp.List[str]]:
        """
        Generator handler.

        Parameters
        ----------
        paragraph: str
            Input paragraph.

        Returns
        -------
        list of str
            List of generated questions.
        list of str
            List of corresponding answers.
        """
        masked_questions = []
        answers = []

        for sent in self.model(paragraph).sents:
            sent_subtree = sent.subtree
            ent_start = sent.start_char
            for ent in sent_subtree:
                # for ent in sent.ents:
                if ent.is_alpha:
                    id_start = ent_start - sent.start_char
                    id_end = ent_start - sent.start_char + len(ent.text)
                    masked_question = sent.text[:id_start] + self.MASKED_TOKEN + sent.text[id_end:]
                    masked_questions.append(masked_question)
                    answers.append(ent.text)
                ent_start += len(ent.text) + 1
        return masked_questions, answers


class QuestionPredictor:
    _EN_CONFIG = {"tokenizer": "bert-base-uncased", "QA": "bert-large-uncased-whole-word-masking-finetuned-squad"}
    _RU_CONFIG = {"tokenizer": "./sbersquad_rubert_large", "QA": "./sbersquad_rubert_large"}
    SEP_TOKEN = "[SEP]"

    def __init__(self, tokenizer_name: tp.Optional[str] = None, qa_name: tp.Optional[str] = None) -> None:
        tokenizer_name = tokenizer_name or self._RU_CONFIG["tokenizer"]
        qa_name = qa_name or self._RU_CONFIG["QA"]

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertForQuestionAnswering.from_pretrained(qa_name)
        self.SEP_id = self.tokenizer.encode(self.SEP_TOKEN)[0]

    def predict(self, question: str, text: str) -> tp.Tuple[str, float]:
        """
        LM question answering model inference method.

        The method logic is taken from the reference repo as is:
        https://github.com/ThomasScialom/summa-qa/blob/39bdaeafc922dbd704bdc4b4af9e587516b831cb/summaqa/qa_models.py#L6

        Parameters
        ----------
        question: str
            Question sentence or paragraph.
        text: str
            Answer reference. Both parameters are put into the question-answering model.
            E.g.: BertForQuestionAnswering.

        Returns
        -------
        str
            Predicted summary.
        float
            Confidence score.
        """

        # input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_text = question + " [SEP] " + text
        input_ids = self.tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(self.SEP_id) else 1 for i in range(len(input_ids))]
        output = self.model(torch.tensor([input_ids]))
        start_scores, end_scores = output.start_logits, output.end_logits

        start_scores = torch.functional.F.softmax(start_scores, -1) * torch.Tensor(token_type_ids)
        end_scores = torch.functional.F.softmax(end_scores, -1) * torch.Tensor(token_type_ids)

        start_values, start_indices = start_scores.topk(1)
        end_values, end_indices = end_scores.topk(1)

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        asw = " ".join(all_tokens[start_indices[0][0] : end_indices[0][0] + 1])
        prob = start_values[0][0] * end_values[0][0]

        return asw, prob.item()


class SummaQAMetrics(Metrics):
    """
    Answers Unite! Unsupervised Metrics for Reinforced Summarization Models.
    Reference: https://arxiv.org/abs/1909.01610

    Attributes
    ----------
    model: QuestionPredictor, optional
        Prediction (or evaluation) model.
    qap_tokenizer: str, optional
        Name of the tokenizer to download from the Huggingface hub.
    qap_model: str, optional
        NAme of the prediction model to download from the Huggingface hub.
    device: str, optional
        Device to be used for inference. By default, try to choose an available GPU.
    score_to_return: str, optional
        For measurements we use probabilities and F-Score. Choose "prob" or "fscore" respectively.

    Note
    ----
    All attributes are optional, since there are models presets for both RU and EN domains.

    """

    def __init__(
        self,
        model: tp.Optional[tp.Any] = None,
        qap_tokenizer: tp.Optional[str] = None,
        qap_model: tp.Optional[str] = None,
        lang: tp.Optional[str] = "ru",
        device: tp.Optional[str] = None,
        score_to_return: tp.Optional[str] = "fscore",
    ) -> None:
        # TODO: support both russian and english versions (currently only russian)
        self.lang = lang
        self.question_generator = QuestionGenerator(lang=lang)

        # Initializing the Question predictor
        if model and (qap_tokenizer or qap_model):
            raise ValueError("qap_tokenizer or qap_model are available only with default QuestionPredictor")
        self.model = model or QuestionPredictor(tokenizer_name=qap_tokenizer, qa_name=qap_model)

        # Other attributes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if score_to_return not in ["fscore", "prob"]:
            raise ValueError("An attribute score_to_return should satisfy to a restricted list of values")
        self.score_to_return = score_to_return
        self._corpus_evaluation = False
        self._average = "macro"

    def _compute(self, questions, true_asws, evaluated_text) -> tp.Dict[str, float]:
        if not questions:
            return {"prob": 0, "fscore": 0}
        score_prob, score_f = 0, 0
        for question, true_asw in zip(questions, true_asws):
            asw_pred, prob = self.model.predict(question, evaluated_text)
            score_prob += prob

            # case tokenizer is applied here
            result = self.model.tokenizer.encode(true_asw) if true_asw else []
            asw_pred_tokenized = self.model.tokenizer.encode(asw_pred) if asw_pred else []
            while len(result) != len(asw_pred_tokenized):
                if len(result) < len(asw_pred_tokenized):
                    asw_pred_tokenized.pop()
                else:
                    asw_pred_tokenized.append(-1)

            if asw_pred_tokenized:
                score_f += metrics.f1_score(result, asw_pred_tokenized, average=self._average)

        return {"prob": score_prob / len(questions), "fscore": score_f / len(questions)}

    def evaluate_example(self, hyp: str, ref: str | None) -> float:
        raise RuntimeError("Separate examples evaluation is not supported for corpus-level metrics")

    def evaluate_batch(self, hyps: tp.List[str], refs: tp.Optional[tp.List[str]] = None) -> tp.List[float]:
        """
        In this approach we use a question generator as pseudo labels source.
        Further, these pseudo labels are used for a comparison with the output retrieved from the question predictor.
        For details, check out the diagram in the reference repo.
        """
        if not self._corpus_evaluation:
            warnings.warn("Batch processing is considered as processing the textual corpus")

        if not refs:
            raise ValueError("SummaQA metric requires a list of references (summaries) as an input")
        scores = {"prob": 0.0, "fscore": 0.0}
        for hyp, ref in zip(hyps, refs):
            masked_questions, masked_question_asws = self.question_generator.generate(hyp)
            gen_score = self._compute(masked_questions, masked_question_asws, ref)
            scores["prob"] += gen_score["prob"]
            scores["fscore"] += gen_score["fscore"]
        scores["prob"] = scores["prob"] / len(hyps)
        scores["fscore"] = scores["fscore"] / len(refs)
        return scores[self.score_to_return]

    def evaluate_corpus(self, hyps: tp.List[str], refs: tp.Optional[tp.List[str]] = None) -> tp.List[float]:
        """
        A copy of the previous method to incorporate the meaning of this metric (it's not batch-level, but corpus-level).
        """
        self._corpus_evaluation = True
        metrics = self.evaluate_batch(hyps, refs)
        self._corpus_evaluation = False
        return metrics
