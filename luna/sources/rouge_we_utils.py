# pylint: disable=C0103,W0621

import bz2
import collections
import logging
import os

import gdown
import numpy as np
import six
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from scipy import spatial

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

tokenizer = RegexpTokenizer(r"\w+")
stopset = frozenset(stopwords.words("english"))
stemmer = SnowballStemmer("english")


###################################################
###             Pre-Processing
###################################################


def get_all_content_words(sentences, stem=True, tokenize=True):
    all_words = []
    if tokenize:
        for s in sentences:
            if stem:
                all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])
            else:
                all_words.extend(tokenizer.tokenize(s))
    else:
        if isinstance(sentences, list):
            all_words = sentences[0].split()
        else:
            all_words = sentences.split()

    normalized_content_words = list(map(normalize_word, all_words))
    return normalized_content_words


def pre_process_summary(summary, stem=True, tokenize=True):
    summary_ngrams = get_all_content_words(summary, stem=stem, tokenize=tokenize)
    return summary_ngrams


def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)


def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))


def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)


def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result


def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def _safe_f1(matches, recall_total, precision_total, alpha, return_all=False):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        if return_all:
            return precision_score, recall_score, (precision_score * recall_score) / denom
        else:
            return (precision_score * recall_score) / denom
    else:
        if return_all:
            return precision_score, recall_score, 0.0
        else:
            return 0.0


def _has_embedding(ngram, embs):
    for w in ngram:
        if not w in embs:
            return False
    return True


def _get_embedding(ngram, embs):
    res = []
    for w in ngram:
        res.append(embs[w])
    return np.sum(np.array(res), 0)


def _find_closest(ngram, counter, embs):
    # If there is nothin to match, nothing is matched
    if len(counter) == 0:
        return "", 0, 0

    # If we do not have embedding for it, we try lexical matching
    if not _has_embedding(ngram, embs):
        if ngram in counter:
            return ngram, counter[ngram], 1
        else:
            return "", 0, 0

    ranking_list = []
    ngram_emb = _get_embedding(ngram, embs)
    for k, v in six.iteritems(counter):
        # First check if there is an exact match
        if k == ngram:
            ranking_list.append((k, v, 1.0))
            continue

        # if no exact match and no embeddings: no match
        if not _has_embedding(k, embs):
            ranking_list.append((k, v, 0.0))
            continue

        # Soft matching based on embeddings similarity
        k_emb = _get_embedding(k, embs)
        ranking_list.append((k, v, 1 - spatial.distance.cosine(k_emb, ngram_emb)))

    # Sort ranking list according to sim
    ranked_list = sorted(ranking_list, key=lambda tup: tup[2], reverse=True)

    # Extract top item
    return ranked_list[0]


def _soft_overlap(peer_counter, model_counter, embs):
    THRESHOLD = 0.8
    result = 0
    for k, v in six.iteritems(peer_counter):
        closest, count, sim = _find_closest(k, model_counter, embs)
        if sim < THRESHOLD:
            continue
        if count <= v:
            del model_counter[closest]
            result += count
        else:
            model_counter[closest] -= v
            result += v

    return result


def rouge_n_we(peer, models, embs, n, alpha=0.5, return_all=False, stem=True, tokenize=True):
    """
    Compute the ROUGE-N-WE score of a peer with respect to one or more models, for
    a given value of `n`.
    """

    if len(models) == 1 and isinstance(models[0], str):
        models = [models]
    peer = pre_process_summary(peer, stem=stem, tokenize=tokenize)
    models = [pre_process_summary(model, stem=stem, tokenize=tokenize) for model in models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _soft_overlap(peer_counter, model_counter, embs)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha, return_all)


#  convert to unicode and convert to lower case
def normalize_word(word):
    return word.lower()


def _convert_to_numpy(vector):
    return np.array([float(x) for x in vector])


def load_embeddings_from_web(save_path):
    """ "
    Download embeddings from gdrive to save_path file.
    Embeddings taken from http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2
    """
    url = "https://drive.google.com/uc?id=1NGAoXi_QzpXl-gAon2UPwpX_PnxYupjn"
    output = save_path + ".bz2"

    logging.info("Downloading embeddings...")
    gdown.download(url, output, quiet=False)

    logging.info("Decompressing embeddings...")
    with open(output, "rb") as file:
        decompressed = bz2.decompress(file.read())

    logging.info("Saving embeddings...")
    with open(save_path, "wb") as outputf:
        outputf.write(decompressed)


def load_embeddings(filepath=None):
    if filepath is None:
        dirname = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(dirname, "embeddings")):
            os.mkdir(os.path.join(dirname, "embeddings"))
        filepath = os.path.join(dirname, "embeddings/deps.words")
        if not os.path.exists(os.path.join(".", filepath)):
            load_embeddings_from_web(filepath)
    dict_embedding = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip().split(" ")
            key = line[0]
            vector = line[1::]
            dict_embedding[key.lower()] = _convert_to_numpy(vector)
    return dict_embedding
