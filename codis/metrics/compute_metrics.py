from typing import List, Dict, Optional, Tuple
from enum import Enum
from multiprocessing import Pool
from functools import partial

from datasets import load_metric
from rouge_score.scoring import BootstrapAggregator
from nltk import sent_tokenize as _sent_tokenize

from . import (
    summary_extractivity, novel_ngrams, rouge, bertscore, words_sentences
)
from .utils import word_tokenize


class Metrics(Enum):
    EXTRACTIVE = 0
    NNG = 1
    ROUGE = 2
    BERT = 3
    WS = 4
    ALL = 5


def update_dicts(completes: List[dict], partials: List[dict]):
    assert len(completes) == len(partials)

    for i in range(len(completes)):
        completes[i].update(partials[i])


def compute_metrics(
    predictions: List[str], texts: Optional[List[str]] = None,
    references: Optional[List[str]] = None, nng: List[int] = [2,3,4],
    ps: List[int] = [1, 2], lang: str = "es",
    metrics: List[Metrics] = [Metrics.ALL]
) -> List[Dict[str, float]]:

    result: List[dict] =  []
    for i in range(len(predictions)):
        result.append({})

    if Metrics.ALL in metrics:
        metrics = [
            Metrics.EXTRACTIVE,
            Metrics.NNG,
            Metrics.ROUGE,
            Metrics.WS,
            Metrics.BERT
        ]

    article_words: List[List[str]] = []
    prediction_words: List[List[str]] = []

    reference_sentences: List[List[str]] = []
    prediction_sentences: List[List[str]] = []

    with Pool(16) as pool:
        if Metrics.EXTRACTIVE in metrics or Metrics.NNG in metrics:
            if texts is None:
                raise ValueError("'texts' is required and is None type")

            article_words = list(pool.map(word_tokenize, texts))
            prediction_words = list(pool.map(word_tokenize, predictions))

        if Metrics.ROUGE in metrics or Metrics.BERT in metrics:
            if references is None:
                raise ValueError("'references' is required and is None type")

            sent_tokenize = partial(_sent_tokenize, language="spanish")

            reference_sentences = list(pool.map(sent_tokenize, references))
            prediction_sentences = list(pool.map(sent_tokenize, predictions))

    if Metrics.EXTRACTIVE in metrics:
        assert texts is not None
        update_dicts(result,
            summary_extractivity(article_words, prediction_words, ps)
        )

    if Metrics.NNG in metrics:
        assert texts is not None
        update_dicts(result,
            novel_ngrams(article_words, prediction_words, nng)
        )

    if Metrics.ROUGE in metrics:
        assert references is not None
        update_dicts(result,
            rouge(prediction_sentences, reference_sentences)
        )

    if Metrics.BERT in metrics:
        assert references is not None
        update_dicts(result,
            bertscore(prediction_sentences, reference_sentences, lang)
        )

    if Metrics.WS in metrics:
        update_dicts(result,
            words_sentences(prediction_sentences)
        )

    return result


def compute_aggregated_metrics(
    predictions: List[str], texts: Optional[List[str]] = None,
    references: Optional[List[str]] = None, nng: List[int] = [2,3,4],
    ps: List[int] = [1, 2], lang: str = "es",
    metrics: List[Metrics] = [Metrics.ALL]
) -> Dict[str, float]:
    aggregator = BootstrapAggregator()

    scores = compute_metrics(
        predictions=predictions, texts=texts, references=references,
        nng=nng, ps=ps, lang=lang, metrics=metrics
    )

    for score in scores:
        aggregator.add_scores(score)

    result = aggregator.aggregate()
    keys = result.keys()

    for k in keys:
        v = result[k]
        result[k] = (v.low, v.mid, v.high)

    return result

def compute_combined_metrics(
    predictions: List[str], texts: Optional[List[str]] = None,
    references: Optional[List[str]] = None, nng: List[int] = [2,3,4],
    ps: List[int] = [1, 2], lang: str = "es",
    metrics: List[Metrics] = [Metrics.ALL]
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    aggregator = BootstrapAggregator()

    scores = compute_metrics(
        predictions=predictions, texts=texts, references=references,
        nng=nng, ps=ps, lang=lang, metrics=metrics
    )

    for score in scores:
        aggregator.add_scores(score)

    result = aggregator.aggregate()
    keys = result.keys()

    for k in keys:
        v = result[k]
        result[k] = (v.low, v.mid, v.high)

    return scores, result
