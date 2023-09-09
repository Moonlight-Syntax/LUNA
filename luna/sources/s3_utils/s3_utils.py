# Copyright 2016
# Ubiquitous Knowledge Processing (UKP) Lab
# Technische Universitat Darmstadt
# Licensed under the Apache License, Version 2.0 (the «License»)
# Changes made to the source code include:
#   - reorganizing file structure
#   - change supplementary metrics to ones we implemented in our toolkit
#   - convert to python 3


import logging
import os
import pickle

import gdown
import numpy as np

from luna.rouge_we import RougeWeMetrics
from luna.sources.s3_utils.js_utils import JS_eval
from luna.sources.s3_utils.rouge_n_utils import rouge_n


def extract_feature(reference, summary_text, emb_path):
    features = dict()

    # Get ROUGE-1, ROUGE-2 recall
    features["ROUGE_1_R"] = rouge_n(summary_text, [reference], 1, 0.0)
    features["ROUGE_2_R"] = rouge_n(summary_text, [reference], 2, 0.0)

    # Get JS. JS_eval supports multireference
    features["JS_eval_1"] = JS_eval(summary_text, [reference], 1)
    features["JS_eval_2"] = JS_eval(summary_text, [reference], 2)

    # Get ROUGE-1-WE, ROUGE-2-WE recall
    features["ROUGE_1_R_WE"] = RougeWeMetrics(emb_path=emb_path, n_gram=1, alpha=0.0).evaluate_example(
        summary_text, reference
    )
    features["ROUGE_2_R_WE"] = RougeWeMetrics(emb_path=emb_path, n_gram=2, alpha=0.0).evaluate_example(
        summary_text, reference
    )

    return features


def load_model_from_web(save_path, url):
    logging.info("Downloading model...")
    gdown.download(url, save_path, quiet=False)


def load_model(save_path, url):
    if not os.path.exists(save_path):
        load_model_from_web(save_path, url)
    return load_model_from_file(save_path)


def load_model_from_file(filename):
    with open(filename, "rb") as f:
        return pickle.loads(f.read())


def S3(reference, system_summary, emb_path, model):
    # Extract features
    instance = extract_feature(reference, system_summary, emb_path)
    features = sorted([f for f in instance.keys()])

    feature_vector = []
    for feat in features:
        feature_vector.append(instance[feat])

    # Apply model
    X = np.array([feature_vector]).astype(np.float64)
    score = model.predict(X)[0]

    return score
