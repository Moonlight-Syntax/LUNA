# Taken from https://github.com/AIPHES/emnlp19-moverscore

from __future__ import absolute_import, division, print_function

import logging
import os
import string
import zipfile
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool

import numpy as np
import requests
import torch
from pyemd import emd
from torch import nn
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def get_model_and_tokenizer_from_transformers(model_name="distilbert-base-uncased", device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    model.eval()
    model.to(device)
    return model, tokenizer


def truncate(tokens, model_max_length):
    if len(tokens) > model_max_length - 2:
        tokens = tokens[0 : (model_max_length - 2)]
    return tokens


def process(tokenizer, model_max_length, a):
    a = ["[CLS]"] + truncate(tokenizer.tokenize(a), model_max_length) + ["[SEP]"]
    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    model_max_length = tokenizer.model_max_length
    process_partial = partial(process, tokenizer, model_max_length)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def bert_encode_transformers(model, model_name, x, attention_mask):
    model.eval()
    with torch.no_grad():
        result = model(x, attention_mask=attention_mask)
    return result.hidden_states 


def collate_idf(arr, tokenizer, idf_dict, pad="[PAD]", device="cuda:0"):
    tokens = [
        ["[CLS]"]
        + truncate(
            tokenizer.tokenize(a), tokenizer.model_max_length
        )
        + ["[SEP]"]
        for a in arr
    ]
    arr = [tokenizer.convert_tokens_to_ids(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.convert_tokens_to_ids([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens


def get_bert_embedding(
    all_sens, model, tokenizer, idf_dict, model_name=None, batch_size=-1, device="cuda:0"
):
    assert model_name, "Please provide model_name argument for the model from huggingface transformers"

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens, tokenizer, idf_dict, device=device)

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode_transformers(
                model, model_name, padded_sens[i : i + batch_size], attention_mask=mask[i : i + batch_size]
            )

            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 0.00001)


def load_ngram(ids, embedding, idf, n, o, device):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(device)
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf


def word_mover_score(
    refs,
    hyps,
    idf_dict_ref,
    idf_dict_hyp,
    model_name,
    model,
    tokenizer,
    stop_words=[],
    n_gram=1,
    remove_subwords=True,
    batch_size=256,
    device="cuda:0",
):
    preds = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start : batch_start + batch_size]
        batch_hyps = hyps[batch_start : batch_start + batch_size]

        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(
            batch_refs, model, tokenizer, idf_dict_ref, model_name=model_name, device=device
        )
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(
            batch_hyps, model, tokenizer, idf_dict_hyp, model_name=model_name, device=device
        )

        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
        hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

        ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
        hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

        ref_embedding_avg = ref_embedding[-5:].mean(0)
        hyp_embedding_avg = hyp_embedding[-5:].mean(0)

        ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
        hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

        for i in range(len(ref_tokens)):
            if remove_subwords:
                ref_ids = [
                    k
                    for k, w in enumerate(ref_tokens[i])
                    if w not in set(string.punctuation) and "##" not in w and w not in stop_words
                ]
                hyp_ids = [
                    k
                    for k, w in enumerate(hyp_tokens[i])
                    if w not in set(string.punctuation) and "##" not in w and w not in stop_words
                ]
            else:
                ref_ids = [
                    k for k, w in enumerate(ref_tokens[i]) if w not in set(string.punctuation) and w not in stop_words
                ]
                hyp_ids = [
                    k for k, w in enumerate(hyp_tokens[i]) if w not in set(string.punctuation) and w not in stop_words
                ]

            ref_embedding_i, ref_idf_i = load_ngram(ref_ids, ref_embedding[i], ref_idf[i], n_gram, 1, device)
            hyp_embedding_i, hyp_idf_i = load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], n_gram, 1, device)

            raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

            distance_matrix = pairwise_distances(raw, raw)

            c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
            c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

            c1[: len(ref_idf_i)] = ref_idf_i
            c2[-len(hyp_idf_i) :] = hyp_idf_i

            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
            preds.append(score)
    return preds
