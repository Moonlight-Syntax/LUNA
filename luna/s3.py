import os
import typing as tp

from luna.base import Metrics
from luna.sources.s3_utils.s3_utils import S3, load_model, load_model_from_file


class S3Metrics(Metrics):
    """
    Implementation of the metric introduced in the paper
    'Learning to Score System Summaries for Better Content Selection Evaluation.', Maxime Peyrard et.al.

    Parameters
    ----------
    mode: str, default = "pyr"
        What type of manual scores was metric trained to approximate. Can be either "pyr", which stands for
        "Pyramid" or "resp", which stands for "responsiveness".
    emb_path: Optional[str], default = None
        Used for rouge-we metric.
        Path to file with word embeddings. If set to None, will download files from
        https://drive.google.com/uc?id=1NGAoXi_QzpXl-gAon2UPwpX_PnxYupjn which is a copy of the file from
        http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2
        The file will be downloaded to the local directory "embeddings/" (created if not exists),
        and will have a name "deps.words"
    model_path: Optional[str], default = None
        Path to .pkl file with weights of a model. If set to None, will download files from
        https://drive.google.com/file/d/19sdnH0e5YOtZBYi3kNQ-n0J8FRwuQhB2/view?usp=share_link for mode "pyr"
        or from https://drive.google.com/file/d/1qD-XfIiSocUi9QUR0ne4sn-vgmQSm3dV/view?usp=sharing for mode "resp"
        and save them into the directory "models" under respective names "pyr.pkl" or "resp.pkl".
        Both files are python3 .pkl files with the same weights as were stored in the official paper repository.
        Do not change this parameter if you want to reproduce metric from the paper.
        Do not change this parameter if you are not absolutely sure that you need to do it, and you are not absolutely
        sure that your model is stored in the same format as models from the official repository.
    Notes
    -----
    Implementation is taken from https://github.com/UKPLab/emnlp-ws-2017-s3
    This is a version that uses only 6 best features: ROUGE-1, ROUGE-2, ROUGE-WE-1, ROUGE-WE-2, JS-1,
    JS-2, and achieves best performance in the paper.
    """

    MODEL_URLS = dict(
        pyr="https://drive.google.com/uc?id=19sdnH0e5YOtZBYi3kNQ-n0J8FRwuQhB2",
        resp="https://drive.google.com/uc?id=1qD-XfIiSocUi9QUR0ne4sn-vgmQSm3dV",
    )

    def __init__(
        self,
        mode: str = "pyr",
        emb_path: tp.Optional[str] = None,
        model_path: tp.Optional[str] = None,
    ):
        if mode not in ["pyr", "resp"]:
            raise ValueError(f"Invalid mode. Mode can be either pyr or resp. Got mode {mode}")
        self.mode = mode
        self.emb_path = emb_path

        if model_path is None:
            dirname = os.path.dirname(__file__)
            if not os.path.exists(os.path.join(dirname, "models")):
                os.mkdir(os.path.join(dirname, "models"))
            save_path = os.path.join(dirname, f"models/{mode}.pkl")
            self.model = load_model(save_path, self.MODEL_URLS[mode])
        else:
            self.model = load_model_from_file(model_path)

    def evaluate_example(self, hyp: str, ref: str) -> float:
        return S3([ref], hyp, self.emb_path, self.model)

    def __repr__(self) -> str:
        return "S3"
