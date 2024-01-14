from typing import List, Dict, Union
from multiprocessing import Pool
from functools import partial

from datasets import load_metric
from nltk import sent_tokenize

from .utils import calc_novel_ngrams, calc_coverage, calc_density,\
    calc_compression_ratio, word_tokenize, get_extractive_phrases,\
    calc_abstractivity, calc_content_reordering


def _calculate_extractive(pair: tuple, ps: List[int]):
    if len(pair) > 0 and isinstance(pair[0], str):
        text = word_tokenize(pair[0])
        summary = word_tokenize(pair[1])
    else:
        text, summary = pair

    ext_phrases = get_extractive_phrases(text, summary)

    r = {
        "coverage": calc_coverage(summary, ext_phrases),
        "density": calc_density(summary, ext_phrases),
        "compression": calc_compression_ratio(text, summary),
        "content-reordering": calc_content_reordering(summary, ext_phrases)
    }

    for p in ps:
        r["abstr-p-%d" % p] = calc_abstractivity(
            summary, ext_phrases, p
        )

    return r


def _calculate_nngs(pair: tuple, ns: List[int]):
    if len(pair) > 0 and isinstance(pair[0], str):
        text = word_tokenize(pair[0])
        summary = word_tokenize(pair[1])
    else:
        text, summary = pair

    r: Dict[str, float] = {}

    for n in ns:
        r["nng-%d" % n] = calc_novel_ngrams(text, summary, n)

    return r


def _calculate_words_stats(summary: List[str]) -> dict:
    nw = 0.0

    for sentence in summary:
        nw += len(word_tokenize(sentence))

    result = {
        "words": nw,
        "sentences": len(summary),
        "words-sentence": nw / len(summary)
    }

    return result


def summary_extractivity(
    texts: Union[List[str], List[List[str]]],
    summaries: Union[List[str], List[List[str]]], ps: List
    [int] = [1, 2]
) -> List[dict]:
    pool = Pool(16)

    with Pool(16) as pool:
        calc = partial(_calculate_extractive, ps=ps)
        result = pool.map(calc, zip(texts, summaries))

    return list(result)


def novel_ngrams(
    articles: Union[List[str], List[List[str]]],
    summaries: Union[List[str], List[List[str]]],
    ns: List [int] = [2,3,4]
) -> List[dict]:


    with Pool(16) as pool:
        calc_scores = partial(_calculate_nngs, ns=ns)
        result = pool.map(calc_scores, zip(articles, summaries))

    return list(result)


def words_sentences(summaries: List[List[str]]) -> List[dict]:
    with Pool(16) as pool:
        return list(pool.map(_calculate_words_stats, summaries))


def rouge(
    predictions: Union[List[str], List[List[str]]],
    references: Union[List[str], List[List[str]]]
) -> List[dict]:
    rouge_metric = load_metric("rouge")
    if len(predictions) > 0 and isinstance(predictions[0], str):
        preds = [
            "\n".join(sent_tokenize(pred, language="spanish")).strip() \
            for pred in predictions
        ]
        refs = [
            "\n".join(sent_tokenize(ref, language="spanish")).strip() \
            for ref in references
        ]
    else:
        preds = ["\n".join(pred).strip() for pred in predictions]
        refs = ["\n".join(ref).strip() for ref in references]

    rouge_metric.add_batch(predictions=preds, references=refs)

    samples_scores = rouge_metric.compute(use_stemmer=False, use_agregator=False)

    result: List = []

    for i in range(len(preds)):
        res = {}

        for k in samples_scores.keys():
            res[k] = samples_scores[k][i].fmeasure

        result.append(res)

    return result


def bertscore(
    predictions: Union[List[str], List[List[str]]],
    references: Union[List[str], List[List[str]]], lang: str = "es"
) -> List[dict]:
    if len(predictions) > 0 and isinstance(predictions[0], str):
        preds = [
            "\n".join(sent_tokenize(pred, language="spanish")).strip() \
            for pred in predictions
        ]
        refs = [
            "\n".join(sent_tokenize(ref, language="spanish")).strip() \
            for ref in references
        ]
    else:
        preds = ["\n".join(pred).strip() for pred in predictions]
        refs = ["\n".join(ref).strip() for ref in references]

    bert_metric = load_metric("bertscore")
    bert_metric.add_batch(predictions=preds, references=refs)

    scores = bert_metric.compute(lang=lang, batch_size=4)

    result: List[dict] = []

    for i in range(len(preds)):
        result.append({"bert": scores["f1"][i]})

    return result
