"""Generate BERT-TF-IDF sentence embeddings for event templates.

Adapted from CSCLog/utils/sentence_embding.py.
"""
import json
import math
import operator
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel


STOPWORDS = {"in", "on", "with", "by", "for", "at", "about", "under", "of", "to", "from"}


def _get_keys(sentence: str) -> List[str]:
    """Tokenize a log template into content words (no stopwords)."""
    line = sentence.lower()
    line = re.sub(r"[^\w\u4e00-\u9fff]+", " ", line)
    return [w for w in line.split() if w not in STOPWORDS]


def _compute_tfidf(list_words: List[List[str]]) -> List[Tuple[str, float]]:
    """Compute TF-IDF over a corpus of tokenized documents."""
    doc_frequency: Dict[str, int] = defaultdict(int)
    for word_list in list_words:
        for w in word_list:
            doc_frequency[w] += 1

    total = sum(doc_frequency.values())
    word_tf = {w: c / total for w, c in doc_frequency.items()}

    doc_num = len(list_words)
    word_doc: Dict[str, int] = defaultdict(int)
    for w in doc_frequency:
        for doc in list_words:
            if w in doc:
                word_doc[w] += 1

    word_tfidf = {
        w: word_tf[w] * math.log(doc_num / (word_doc[w] + 1))
        for w in doc_frequency
    }
    return sorted(word_tfidf.items(), key=operator.itemgetter(1), reverse=True)


def _word_embeddings(
    keys: List[Tuple[str, float]],
    tokenizer,
    model: BertModel,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute BERT embedding * TF-IDF weight for each (word, weight) pair."""
    encode_keys = {}
    model.eval()
    with torch.no_grad():
        for word, weight in keys:
            encoded = tokenizer(word, return_tensors="pt").to(device)
            output = model(**encoded)
            # Mean-pool token embeddings, scale by TF-IDF weight
            vec = output[0].mean(dim=1).squeeze(0)
            encode_keys[word] = vec * weight
    return encode_keys


def _sentence_embeddings(
    sentences: Dict[str, List[str]],
    word_vecs: Dict[str, torch.Tensor],
) -> Dict[str, List[float]]:
    """Sum word vectors to build a sentence-level embedding per EventId."""
    first = next(iter(word_vecs.values()))
    result = {}
    for event_id, words in sentences.items():
        vec = torch.zeros_like(first)
        for w in words:
            if w in word_vecs:
                vec = vec + word_vecs[w]
        result[event_id] = vec.tolist()
    return result


def build_embeddings(
    templates_csv: str,
    bert_model_path: str,
    output_json: str,
    device_str: str = "cpu",
) -> Dict[str, List[float]]:
    """Full pipeline: templates CSV → sentence embeddings JSON.

    Args:
        templates_csv:    Path to <name>_templates.csv with EventId, EventTemplate columns.
        bert_model_path:  Directory containing BERT checkpoint.
        output_json:      Where to write the JSON mapping {EventId: [float, ...]}.
        device_str:       'cuda' or 'cpu'.

    Returns:
        Dict mapping EventId → embedding vector.
    """
    device = torch.device(device_str)
    datas = pd.read_csv(templates_csv)

    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    bert = BertModel.from_pretrained(bert_model_path).to(device)

    sentences: Dict[str, List[str]] = {
        row["EventId"]: _get_keys(row["EventTemplate"])
        for _, row in datas.iterrows()
    }

    all_words = [words for words in sentences.values()]
    tfidf_keys = _compute_tfidf(all_words)

    word_vecs = _word_embeddings(tfidf_keys, tokenizer, bert, device)
    emb = _sentence_embeddings(sentences, word_vecs)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(emb, f)

    print(f"Embeddings written to {output_json} ({len(emb)} templates, dim={len(next(iter(emb.values())))})")
    return emb
