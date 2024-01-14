import re
import os
import pathlib
import unicodedata

from typing import List, Dict, Set, Tuple, Optional

import numpy as np

from datasets import load_metric
from nltk.tokenize.toktok import ToktokTokenizer
from nltk import word_tokenize as _word_tokenize, sent_tokenize

norm_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])

def normalize(s0: str) -> str:
    s = s0.strip()
    s = unicodedata.normalize("NFKC", s)  # passa la ŀl a l·l
    s = s.translate(norm_table)  # normalitza algunes '
    s = s.replace("L•L", "L·L").replace("L•l", "L·l")
    s = s.replace("l•l", "l·l").replace("l•L", "l·L")
    s = s.replace("L.L", "L·L").replace("L.l", "L·l")
    s = s.replace("l.l", "l·l").replace("l.L", "l·L")
    return s

def lcs(S1, S2):
    m = len(S1)
    n = len(S2)

    L = [[0 for x in range(n+1)] for x in range(m+1)]

    # Building the mtrix in bottom-up way
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif S1[i-1] == S2[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    index = L[m][n]

    lcs_algo = [""] * (index+1)
    lcs_algo[index] = ""

    i = m
    j = n
    while i > 0 and j > 0:

        if S1[i-1] == S2[j-1]:
            lcs_algo[index-1] = S1[i-1]
            i -= 1
            j -= 1
            index -= 1

        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1

    return len(lcs_algo[:-1])

def word_tokenize_old(text: str) -> List[str]:
    tokenizer = ToktokTokenizer()

    x = normalize(text)
    x = re.sub(r"[^0-9a-záéíóúàèòäëïöüçñ ·'\-]",' ', x.lower())
    x = tokenizer.tokenize(x)

    return [w.strip() for w in x if len(w.strip())]


def word_tokenize(text: str) -> List[str]:
    text = normalize(text.lower())
    #text = text.replace("-", "")
    #text = text.replace("'", " ")
    #text = re.sub(r" [ ]+", ' ', text)
    return [
        w for w in _word_tokenize(text, language="spanish")
    ]


def get_extractive_phrases(
    a: List[str], s: List[str]
) -> List[Tuple[int, int, List[str]]]:
    positions: Dict[str, List[int]] = {}

    for i, word in enumerate(a):
        if word not in positions:
            positions[word] = []

        positions[word].append(i)

    F_: Set[str] = set()
    F: List[Tuple[int, int, List[str]]] = []
    i, j = 0, 0

    while i < len(s):
        f: Optional[Tuple[int, int, List[str]]] = None

        if s[i] in positions:
            cfp = 0
            mlen = len(positions[s[i]])

            while cfp < mlen:
                u = i
                v = positions[s[i]][cfp]

                while u < len(s) and v < len(a) and s[u] == a[v]:
                    u += 1
                    v += 1

                if f is None or len(f[2]) <= (u-i-1):
                    f = (positions[s[i]][cfp], i, s[i:u])

                while cfp < mlen and positions[s[i]][cfp] <= v:
                    cfp += 1

        if f is not None:
            i += max(len(f[2]), 1)
        else:
            i += 1

        if f is not None and len(f[2]) > 0:
            key = " ".join(f[2])

            if key not in F_:
                F.append(f)
                F_.add(key)

    return F


def get_ngrams(words: List[str], n: int) -> Set[str]:
    result = set()

    for i in range(len(words) - n + 1):
        result.add(",".join(words[i:i + n]))

    return result


def calc_coverage(
        s: List[str], ext_phrases: List[Tuple[int, int, List[str]]]
) -> float:
    if len(s) == 0:
        return .0

    sum_lens = sum([len(k) for (_,_,k) in ext_phrases])
    return (1. / len(s)) * sum_lens


def calc_density(
        s: List[str], ext_phrases: List[Tuple[int, int, List[str]]]
) -> float:
    if len(s) == 0:
        return .0

    sum_sqr_lens = sum([len(k)**2 for (_,_,k) in ext_phrases])
    return (1. / len(s)) * sum_sqr_lens


def calc_disorder_lcs(
    s: List[str], ext_phrases: List[Tuple[int, int, List[str]]]
) -> float:
    if len(ext_phrases) == 0:
        return 1.0

    a_ordered = sorted(ext_phrases, key=lambda x: x[0])
    s_ordered = sorted(ext_phrases, key=lambda x: x[1])

    a_text = []
    s_text = []

    for i in range(len(a_ordered)):
        a_text.extend(a_ordered[i][2])
        s_text.extend(s_ordered[i][2])

    mlen = lcs(a_text, s_text)

    return 1 - (mlen / len(s))


def calc_disorder_pos(
    s: List[str], ext_phrases: List[Tuple[int, int, List[str]]]
) -> float:
    if len(s) == 0:
        return 0.0

    if len(ext_phrases) == 0:
        return 1.0

    a_ordered = [sent[2] for sent in sorted(ext_phrases, key=lambda x: x[0])]
    s_ordered = [sent[2] for sent in sorted(ext_phrases, key=lambda x: x[1])]

#    a_text = []
#    s_text = []

#    for i in range(len(a_ordered)):
#        a_text.extend(a_ordered[i][2])
#        s_text.extend(s_ordered[i][2])

    a_tuples = sorted(list(zip(a_ordered, range(len(a_ordered)))))
    s_tuples = sorted(list(zip(s_ordered, range(len(s_ordered)))))

    disorder = 0.0
    lenext = 0.0

    for i in range(len(a_tuples)):
        d = abs(a_tuples[i][1] - s_tuples[i][1]) * (len(a_tuples[i][0]) ** 2)
        disorder += d
        lenext += len(a_tuples[i][0])

    disorder /= (len(a_tuples)**2)
    abstract = float(len(s) - lenext)

    if abstract > 0:
        return 1.0/abstract + lenext/len(s) * disorder

    return disorder


def calc_content_reordering(
    s: List[str], ext_phrases: List[Tuple[int, int, List[str]]]
) -> float:
    def inversions(s):
        count = 0
        for i in range(len(s)-1):
            for j in range(i+1,len(s)):
                if s[i] > s[j]:
                    count +=  1
                    break
        return count

    if len(s) == 0:
        return 0.0

    if len(ext_phrases) == 0:
        return 1.0

    indices = [sent[1] for sent in sorted(ext_phrases, key=lambda x: x[0])]
    lenext = float(sum([len(sent[2]) for sent in ext_phrases]))

    abstract = float(len(s) - lenext)
    creorder = float(inversions(indices)/len(ext_phrases))

    if abstract > 0:
        return 1.0/abstract + float(lenext/len(s)) * creorder

    return creorder


def calc_abstractivity(
        s: List[str], ext_phrases: List[Tuple[int, int, List[str]]], p: int
) -> float:
    if len(s) == 0:
        return .0

    sum_sqr_lens = sum([len(k)**p for (_,_,k) in ext_phrases])
    return 1 - (1. / (len(s)**p)) * sum_sqr_lens


def calc_compression_ratio(a: list, s: list) -> float:
    if len(s) == 0:
        return .0

    return float(len(a)) / len(s)


def calc_novel_ngrams(article: List[str], summary: List[str], n: int = 2):
    article_ngrams = get_ngrams(article, n)
    summary_ngrams = get_ngrams(summary, n)

    if len(summary_ngrams) == 0:
        return .0

    return len(summary_ngrams.difference(article_ngrams)) / len(summary_ngrams)


def get_prefix(filename: str) -> str:
    base = os.path.basename(filename)
    prefix = os.path.splitext(base)[0].split("_")[0]

    return prefix


def normalize_vects(x: List[float], range_value: Tuple[float, float]):
    res = np.array(x)

    return (res - range_value[0]) / (range_value[1] - range_value[0])
