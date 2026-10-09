"""Offline retrieval evaluation against a curated JSON relevance set."""

import argparse
import csv
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


def beer_key(metadata: dict) -> tuple[str, str]:
    return normalize(metadata.get("beer_name", "")), normalize(metadata.get("beer_style", ""))


def is_relevant(metadata: dict, case: dict, human_labels: dict | None = None) -> bool:
    if human_labels is not None:
        return human_labels.get(beer_key(metadata), False)
    names = {normalize(value) for value in case.get("relevant_beer_names", [])}
    styles = {normalize(value) for value in case.get("relevant_styles", [])}
    return normalize(metadata.get("beer_name", "")) in names or normalize(metadata.get("beer_style", "")) in styles


def score_case(
    case: dict,
    results: list[dict],
    target_count: int,
    k: int,
    human_labels: dict | None = None,
) -> dict:
    if target_count == 0:
        raise ValueError(f"Caso '{case.get('id', '?')}' não corresponde a cervejas no catálogo.")
    if human_labels is not None:
        missing = [item for item in results[:k] if beer_key(item) not in human_labels]
        if missing:
            raise ValueError(
                f"Revisão humana do caso '{case['id']}' não contém todos os resultados do top {k}."
            )
    ranked = [is_relevant(item, case, human_labels) for item in results[:k]]
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
            {
                "beer_name": item.get("beer_name"),
                "beer_style": item.get("beer_style"),
                "abv": item.get("abv") if item.get("abv_known", True) else "",
                "review_overall": item.get("review_overall", ""),
                "review_count": item.get("review_count", ""),
                "relevant": relevant,
            }
            for item, relevant in zip(results[:k], ranked)
        ],
    }


def aggregate_metrics(scored_cases: list[dict]) -> dict:
    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
    ]
    return {
        name: sum(item[name] for item in scored_cases) / len(scored_cases)
        for name in metric_names
    }


def read_human_judgments(path: Path, cases: list[dict]) -> dict[str, dict[tuple[str, str], bool]]:
    labels: dict[str, dict[tuple[str, str], bool]] = {}
    answers = {"1": True, "0": False, "yes": True, "no": False, "sim": True, "não": False}
    with path.open(encoding="utf-8-sig", newline="") as review_file:
        reader = csv.DictReader(review_file)
        required_columns = {"case_id", "beer_name", "beer_style", "relevance"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "A planilha de revisão precisa das colunas case_id, beer_name, beer_style e relevance."
            )
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            key = beer_key(row)
            answer = (row.get("relevance") or "").strip().casefold()
            if not case_id or not key[0] or not key[1]:
                continue
            if not answer:
                continue
            if answer not in answers:
                raise ValueError(
                    f"Relevância inválida no caso '{case_id}' para '{key[0]}': use 1/0, sim/não ou yes/no."
                )
            case_labels = labels.setdefault(case_id, {})
            if key in case_labels:
                raise ValueError(f"Cerveja duplicada na revisão do caso '{case_id}': {key[0]}.")
            case_labels[key] = answers[answer]

    missing_cases = [case["id"] for case in cases if not labels.get(case["id"])]
    if missing_cases:
        raise ValueError(
            "A revisão humana ainda não tem rótulos preenchidos para: " + ", ".join(missing_cases)
        )
    return labels


def write_human_review_template(
    path: Path, cases: list[dict], baseline_scored: list[dict], reranked_scored: list[dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as review_file:
        fields = [
            "case_id",
            "query",
            "beer_name",
            "beer_style",
            "abv",
            "review_overall",
            "review_count",
            "baseline_rank",
            "reranked_rank",
            "relevance",
            "notes",
        ]
        writer = csv.DictWriter(review_file, fieldnames=fields)
        writer.writeheader()
        for case, baseline, reranked in zip(cases, baseline_scored, reranked_scored):
            candidates: dict[tuple[str, str], dict] = {}
            for system, scores in (("baseline", baseline), ("reranked", reranked)):
                for rank, item in enumerate(scores["retrieved"], start=1):
                    key = beer_key(item)
                    candidate = candidates.setdefault(
                        key,
                        {
                            "case_id": case["id"],
                            "query": case["query"],
                            "beer_name": item.get("beer_name", ""),
                            "beer_style": item.get("beer_style", ""),
                            "abv": item.get("abv", ""),
                            "review_overall": item.get("review_overall", ""),
                            "review_count": item.get("review_count", ""),
                            "baseline_rank": "",
                            "reranked_rank": "",
                            "relevance": "",
                            "notes": "",
                        },
                    )
                    candidate[f"{system}_rank"] = rank
            writer.writerows(candidates.values())


def evaluate(
    cases_path: Path = CASES_PATH,
    output_path: Path = OUTPUT_PATH,
    k: int = 5,
    review_template_path: Path | None = None,
    human_judgments_path: Path | None = None,
) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("O arquivo de avaliação deve conter uma lista não vazia de casos.")
    for case in cases:
        if not case.get("id") or not case.get("query"):
            raise ValueError("Cada caso precisa de 'id' e 'query'.")
    human_labels = read_human_judgments(human_judgments_path, cases) if human_judgments_path else None

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection = client.get_collection(name=settings.collection_name, embedding_function=embedder)
    if collection.count() == 0:
        raise ValueError("A coleção está vazia. Execute a ingestão antes da avaliação.")
    catalog = collection.get(include=["metadatas"])["metadatas"]

    baseline_scored = []
    reranked_scored = []
    for case in cases:
        result = collection.query(
            query_texts=[case["query"]],
            n_results=min(max(k, settings.retrieval_candidates), collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        vector_results = result["metadatas"][0]
        case_labels = human_labels.get(case["id"]) if human_labels is not None else None
        if case_labels is not None:
            relevant_catalog_count = sum(case_labels.values())
        else:
            relevant_catalog_count = sum(is_relevant(item, case) for item in catalog)
        baseline_scored.append(
            score_case(case, vector_results[:k], relevant_catalog_count, k, case_labels)
        )

        ranked = rerank_candidates(
            result["documents"][0], result["metadatas"][0], result["distances"][0], k
        )
        metadata = [item["metadata"] for item in ranked]
        reranked_scored.append(
            score_case(case, metadata, relevant_catalog_count, k, case_labels)
        )

    if review_template_path:
        write_human_review_template(review_template_path, cases, baseline_scored, reranked_scored)

    baseline_metrics = aggregate_metrics(baseline_scored)
    reranked_metrics = aggregate_metrics(reranked_scored)
    metric_deltas = {
        name: reranked_metrics[name] - value
        for name, value in baseline_metrics.items()
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection.name,
        "collection_size": collection.count(),
        "k": k,
        "cases_count": len(reranked_scored),
        "evaluation_mode": "human_pooled_judgments" if human_labels is not None else "style_proxy",
        "metrics": reranked_metrics,
        "baseline": {
            "name": "vector_similarity",
            "metrics": baseline_metrics,
        },
        "reranked": {
            "name": "vector_similarity_plus_business_signals",
            "metrics": reranked_metrics,
        },
        "reranking_delta": metric_deltas,
        "cases": [
            {
                "id": reranked["id"],
                "query": reranked["query"],
                "baseline": baseline,
                "reranked": reranked,
            }
            for baseline, reranked in zip(baseline_scored, reranked_scored)
        ],
        "note": (
            "Rótulos humanos cobrem o pool dos resultados dos dois sistemas; recall usa apenas os itens desse pool."
            if human_labels is not None
            else "Rótulos por estilo medem correspondência de categoria (proxy); prefira cervejas relevantes revisadas por humanos para avaliar relevância semântica."
        ),
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
    parser.add_argument("--review-template", type=Path)
    parser.add_argument("--human-judgments", type=Path)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k precisa ser maior que zero")
    if args.review_template and args.human_judgments:
        parser.error("use --review-template ou --human-judgments, não ambos na mesma execução")
    evaluate(args.cases, args.output, args.k, args.review_template, args.human_judgments)
