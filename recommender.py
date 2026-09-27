"""Shared retrieval, lightweight ranking, and grounded answer generation."""

import logging
import math
import re
import time
import unicodedata

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama

from settings import settings

logger = logging.getLogger(__name__)


def connect_collection():
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    return client.get_collection(name=settings.collection_name, embedding_function=embedder)


def available_styles(collection) -> list[str]:
    records = collection.get(include=["metadatas"])["metadatas"]
    return sorted({str(row["beer_style"]) for row in records if row and row.get("beer_style")}, key=len, reverse=True)


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9%]+", " ", value).strip()


def _style_constraints(query: str, styles: list[str]) -> list[str]:
    normalized_query = _normalize(query)
    matches = {style for style in styles if _normalize(style) in normalized_query}

    aliases = {
        "double ipa": lambda style: "ipa" in _normalize(style) and ("double" in _normalize(style) or "imperial" in _normalize(style)),
        "imperial ipa": lambda style: "ipa" in _normalize(style) and "imperial" in _normalize(style),
        "ipa": lambda style: "ipa" in _normalize(style),
        "stout": lambda style: "stout" in _normalize(style),
        "porter": lambda style: "porter" in _normalize(style),
        "pilsner": lambda style: "pilsner" in _normalize(style),
        "lager": lambda style: "lager" in _normalize(style),
        "sour": lambda style: any(word in _normalize(style) for word in ("sour", "wild ale", "berliner weissbier")),
        "wheat": lambda style: "wheat" in _normalize(style) or "hefeweizen" in _normalize(style),
    }
    for phrase, predicate in aliases.items():
        if re.search(rf"\b{re.escape(phrase)}\b", normalized_query):
            if not any(phrase in _normalize(style) for style in matches):
                matches.update(style for style in styles if predicate(style))
    return sorted(matches)


def extract_structured_filters(query: str, styles: list[str]) -> tuple[dict | None, list[str]]:
    normalized = _normalize(query)
    conditions: list[dict] = []
    labels: list[str] = []
    selected_styles = _style_constraints(query, styles)
    if selected_styles:
        conditions.append({"beer_style": {"$in": selected_styles}})
        labels.append("Estilo: " + ", ".join(selected_styles))

    number = r"(\d+(?:[.,]\d+)?)"
    above_patterns = [
        rf"(?:abv|alcohol|alcool|graduacao)\s*(?:(?:de|of)\s*)?(?:acima de|mais de|maior que|no minimo|at least|over|above|greater than)\s*{number}",
        rf"{number}\s*%?\s*(?:abv|alcohol|alcool)?\s*(?:ou mais|no minimo|at least|or more|minimum|more than)",
        rf"(?:acima de|more than|above)\s*{number}\s*%?\s*(?:abv|alcohol|alcool)?",
    ]
    below_patterns = [
        rf"(?:abv|alcohol|alcool|graduacao)\s*(?:(?:de|of)\s*)?(?:abaixo de|menos de|menor que|no maximo|at most|under|below)\s*{number}",
        rf"{number}\s*%?\s*(?:abv|alcohol|alcool)?\s*(?:ou menos|no maximo|at most|or less|maximum)",
        rf"(?:abaixo de|less than|below|under)\s*{number}\s*%?\s*(?:abv|alcohol|alcool)?",
    ]
    operator = re.search(r"\b(?:abv|alcohol|alcool)?\s*(>=|>)\s*(\d+(?:[.,]\d+)?)", query.casefold())
    lower_operator = re.search(r"\b(?:abv|alcohol|alcool)?\s*(<=|<)\s*(\d+(?:[.,]\d+)?)", query.casefold())
    above = operator or next((match for pattern in above_patterns if (match := re.search(pattern, normalized))), None)
    below = lower_operator or next((match for pattern in below_patterns if (match := re.search(pattern, normalized))), None)
    if above and (not below or above.start() <= below.start()):
        text = above.group(2) if operator and above is operator else above.group(1)
        abv = float(text.replace(",", "."))
        conditions.extend([{"abv_known": {"$eq": True}}, {"abv": {"$gte": abv}}])
        labels.append(f"ABV minimo {abv:g}%")
    elif below:
        text = below.group(2) if lower_operator and below is lower_operator else below.group(1)
        abv = float(text.replace(",", "."))
        conditions.extend([{"abv_known": {"$eq": True}}, {"abv": {"$lte": abv}}])
        labels.append(f"ABV maximo {abv:g}%")
    elif any(term in normalized for term in ("abv elevado", "alto abv", "high abv", "high alcohol", "strong beer", "cerveja forte")):
        conditions.extend([{"abv_known": {"$eq": True}}, {"abv": {"$gte": 8.0}}])
        labels.append("ABV minimo 8% (criterio inferido)")
    elif any(term in normalized for term in ("low abv", "baixo abv", "baixo teor alcoolico", "low alcohol")):
        conditions.extend([{"abv_known": {"$eq": True}}, {"abv": {"$lte": 5.0}}])
        labels.append("ABV maximo 5% (criterio inferido)")

    if any(term in normalized for term in ("bem avaliada", "bem avaliado", "alta nota", "well rated", "highly rated", "high rated", "top rated")):
        conditions.extend([
            {"review_overall": {"$gte": settings.high_rating_threshold}},
            {"review_count": {"$gte": settings.minimum_rating_reviews}},
        ])
        labels.append(f"Nota geral minima {settings.high_rating_threshold:.1f}/5 em ao menos {settings.minimum_rating_reviews} avaliacoes")
    if any(term in normalized for term in ("alta nota de paladar", "high palate rating", "high palate score")):
        conditions.append({"review_palate": {"$gte": settings.high_rating_threshold}})
        labels.append(f"Nota media de paladar minima {settings.high_rating_threshold:.1f}/5")

    if not conditions:
        return None, labels
    return (conditions[0] if len(conditions) == 1 else {"$and": conditions}), labels


def _rank_result(distance: float, metadata: dict, max_reviews: int) -> float:
    semantic = max(0.0, 1.0 - float(distance or 0.0) / 2.0)
    count = max(1, int(metadata.get("review_count", 1) or 1))
    raw_rating = max(0.0, min(5.0, float(metadata.get("review_overall", 0.0) or 0.0)))
    # Shrink small-sample averages toward the catalog mean to reduce one-review winners.
    rating = ((raw_rating * count) + (3.5 * 20)) / (count + 20) / 5.0
    popularity = math.log1p(count) / math.log1p(max(1, max_reviews))
    return 0.82 * semantic + 0.15 * rating + 0.03 * popularity


def rerank_candidates(documents: list[str], metadatas: list[dict], distances: list[float], k: int) -> list[dict]:
    max_reviews = max((int(meta.get("review_count", 1) or 1) for meta in metadatas), default=1)
    items = [
        {
            "document": document,
            "metadata": metadata,
            "distance": distance,
            "ranking_score": _rank_result(distance, metadata, max_reviews),
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
    items.sort(key=lambda item: item["ranking_score"], reverse=True)
    return items[:k]


def retrieve_beers(collection, query: str, styles: list[str], k: int | None = None) -> dict:
    started_at = time.perf_counter()
    k = k or settings.retrieval_k
    collection_size = collection.count()
    if not collection_size:
        return {"items": [], "filters": [], "filter_applied": None}
    where, labels = extract_structured_filters(query, styles)
    count = min(collection_size, max(k, settings.retrieval_candidates))
    raw = collection.query(
        query_texts=[query],
        n_results=count,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]
    items = rerank_candidates(documents, metadatas, distances, k)
    logger.info("retrieval_complete results=%d structured_filter=%s elapsed_seconds=%.3f", len(items), bool(where), time.perf_counter() - started_at)
    return {"items": items, "filters": labels, "filter_applied": where}


def build_prompt(query: str, items: list[dict]) -> str:
    context = "\n".join(item["document"] for item in items)
    return f"""Você é um sommelier de cervejas artesanais. Responda em português do Brasil.
Recomende somente cervejas presentes no contexto. Use os dados como fonte factual: não invente ABV, notas, avaliações ou características sensoriais.
Use uma lista curta, explique por que cada opção atende ao pedido e informe quando o ABV ou a quantidade de avaliações for desconhecida.

CONTEXTO DO CATÁLOGO:
{context}

PEDIDO:
{query}

RECOMENDAÇÃO:"""


def generate_answer(query: str, items: list[dict], temperature: float = 0.3) -> str:
    started_at = time.perf_counter()
    llm = Ollama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=temperature)
    answer = llm.invoke(build_prompt(query, items))
    logger.info("generation_complete model=%s elapsed_seconds=%.3f", settings.ollama_model, time.perf_counter() - started_at)
    return answer
