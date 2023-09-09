# Copyright 2016
# Ubiquitous Knowledge Processing (UKP) Lab
# Technische Universitat Darmstadt
# Licensed under the Apache License, Version 2.0 (the «License»)
# Changes made to the source code include:
#   - reorganized file structure
# 	- removed conversion to unicode
# 	- changed to python 3


import math

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

###################################################
###				Pre-Processing
###################################################


def normalize_word(word):
    return word.lower()


def is_ngram_content(ngram, stopset):
    for gram in ngram:
        if not (gram in stopset):
            return True
    return False


def get_all_content_words(sentences, N):
    all_words = []
    tokenizer = RegexpTokenizer(r"\w+")
    stemmer = SnowballStemmer("english")
    for s in sentences:
        all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])

    stopset = frozenset(stopwords.words("english"))
    if N == 1:
        content_words = [w for w in all_words if w not in stopset]
    else:
        content_words = all_words

    normalized_content_words = list(map(normalize_word, content_words))
    if N > 1:
        return [gram for gram in ngrams(normalized_content_words, N) if is_ngram_content(gram, stopset)]
    return normalized_content_words


def compute_word_freq(words):
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    return word_freq


def compute_tf(sentences, N=1):
    content_words = get_all_content_words(sentences, N)  ## stemmed
    content_words_count = len(content_words)
    content_words_freq = compute_word_freq(content_words)

    content_word_tf = dict((w, f / float(content_words_count)) for w, f in content_words_freq.items())
    return content_word_tf


def pre_process_summary(summary, ngrams):
    return compute_tf(summary, ngrams)


###################################################
###				Metrics
###################################################


def KL_Divergence(summary_freq, doc_freq):
    sum_val = 0
    for w, f in summary_freq.items():
        if w in doc_freq:
            sum_val += f * math.log(f / float(doc_freq[w]))

    if np.isnan(sum_val):
        raise Exception("KL_Divergence returns NaN")

    return sum_val


def compute_average_freq(l_freq_1, l_freq_2):
    average_freq = {}
    keys = set(l_freq_1.keys()) | set(l_freq_2.keys())

    for k in keys:
        s_1 = l_freq_1.get(k, 0)
        s_2 = l_freq_2.get(k, 0)
        average_freq[k] = (s_1 + s_2) / 2.0

    return average_freq


def JS_Divergence(doc_freq, summary_freq):
    average_freq = compute_average_freq(summary_freq, doc_freq)
    js = (KL_Divergence(summary_freq, average_freq) + KL_Divergence(doc_freq, average_freq)) / 2.0

    if np.isnan(js):
        raise Exception("JS_Divergence returns NaN")

    return js


def JS_eval(summary, references, n):
    sum_rep = pre_process_summary(summary, n)
    refs_reps = [pre_process_summary(ref, n) for ref in references]

    avg = 0.0
    for ref_rep in refs_reps:
        avg += JS_Divergence(ref_rep, sum_rep)

    return avg / float(len(references))
