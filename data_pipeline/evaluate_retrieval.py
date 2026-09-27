"""Offline retrieval evaluation against a curated JSON relevance set."""

import argparse
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from recommender import rerank_candidates
from settings import settings

CASES_PATH = Path("evaluation/retrieval_cases.json")
OUTPUT_PATH = Path("evaluation/retrieval_report.json")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"\s+", " ", text).strip()


def is_relevant(metadata: dict, case: dict) -> bool:
    names = {normalize(value) for value in case.get("relevant_beer_names", [])}
    styles = {normalize(value) for value in case.get("relevant_styles", [])}
    return normalize(metadata.get("beer_name", "")) in names or normalize(metadata.get("beer_style", "")) in styles


def score_case(case: dict, results: list[dict], target_count: int, k: int) -> dict:
    if target_count == 0:
        raise ValueError(f"Caso '{case.get('id', '?')}' não corresponde a cervejas no catálogo.")
    ranked = [is_relevant(item, case) for item in results[:k]]
    hits = sum(ranked)
    positions = [index + 1 for index, relevant in enumerate(ranked) if relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank, relevant in enumerate(ranked, 1) if relevant)
    ideal_hits = min(target_count, k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "id": case["id"],
        "query": case["query"],
        "relevant_results": hits,
        "precision_at_k": hits / k,
        "recall_at_k": hits / target_count,
        "hit_rate_at_k": float(hits > 0),
        "reciprocal_rank": 1 / positions[0] if positions else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
        "retrieved": [
            {"beer_name": item.get("beer_name"), "beer_style": item.get("beer_style"), "relevant": relevant}
            for item, relevant in zip(results[:k], ranked)
        ],
    }


def evaluate(cases_path: Path = CASES_PATH, output_path: Path = OUTPUT_PATH, k: int = 5) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("O arquivo de avaliação deve conter uma lista não vazia de casos.")
    for case in cases:
        if not case.get("id") or not case.get("query"):
            raise ValueError("Cada caso precisa de 'id' e 'query'.")

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection = client.get_collection(name=settings.collection_name, embedding_function=embedder)
    if collection.count() == 0:
        raise ValueError("A coleção está vazia. Execute a ingestão antes da avaliação.")
    catalog = collection.get(include=["metadatas"])["metadatas"]

    scored = []
    for case in cases:
        result = collection.query(
            query_texts=[case["query"]],
            n_results=min(max(k, settings.retrieval_candidates), collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ranked = rerank_candidates(
            result["documents"][0], result["metadatas"][0], result["distances"][0], k
        )
        metadata = [item["metadata"] for item in ranked]
        relevant_catalog_count = sum(is_relevant(item, case) for item in catalog)
        scored.append(score_case(case, metadata, relevant_catalog_count, k))
    metric_names = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "reciprocal_rank", "ndcg_at_k"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection.name,
        "collection_size": collection.count(),
        "k": k,
        "cases_count": len(scored),
        "metrics": {name: sum(item[name] for item in scored) / len(scored) for name in metric_names},
        "cases": scored,
        "note": "Rótulos por estilo medem correspondência de categoria (proxy); prefira cervejas relevantes revisadas por humanos para avaliar relevância semântica.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Relatório salvo em {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k precisa ser maior que zero")
    evaluate(args.cases, args.output, args.k)
