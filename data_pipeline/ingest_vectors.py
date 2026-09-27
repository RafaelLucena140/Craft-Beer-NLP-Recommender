"""Build a replacement Chroma collection, then swap it into service."""

import hashlib
import math
import os
import uuid
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

DATA_PATH = Path(os.getenv("BEER_DATA_PATH", "data/raw/beers_cleaned.csv"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "craft_beers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 256
REQUIRED_COLUMNS = {"beer_name", "beer_style", "beer_abv", "review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"}
SCORE_COLUMNS = ["review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"]


def _stable_id(row: pd.Series) -> str:
    identity = "|".join(str(row.get(column, "")) for column in ("brewery_name", "beer_name", "beer_style"))
    return hashlib.sha256(identity.casefold().encode("utf-8")).hexdigest()


def _number(value) -> float:
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else 0.0


def _record(row: pd.Series):
    abv_known = pd.notna(row["beer_abv"]) and math.isfinite(float(row["beer_abv"]))
    abv = float(row["beer_abv"]) if abv_known else 0.0
    score_values = {col: _number(row[col]) for col in SCORE_COLUMNS}
    brewery = row.get("brewery_name")
    brewery = str(brewery).strip() if pd.notna(brewery) else "Desconhecida"
    name, style = str(row["beer_name"]).strip(), str(row["beer_style"]).strip()
    document = (
        f"Beer: {name}. Brewery: {brewery}. Style: {style}. "
        f"ABV: {abv if abv_known else 'unknown'}. "
        + ", ".join(f"{col.removeprefix('review_').title()}: {value:g}" for col, value in score_values.items())
        + "."
    )
    metadata = {
        "beer_name": name,
        "beer_style": style,
        "brewery_name": brewery,
        "abv": abv,
        "abv_known": bool(abv_known),
        **score_values,
    }
    return _stable_id(row), document, metadata


def ingest_data(data_path: Path = DATA_PATH) -> int:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset limpo não encontrado: {data_path}. Execute data_pipeline/clean_data.py primeiro.")
    df = pd.read_csv(data_path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset limpo vazio; a coleção atual foi mantida.")
    for column in ["beer_name", "beer_style", *SCORE_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce") if column in SCORE_COLUMNS else df[column].astype("string").str.strip()
    df["beer_abv"] = pd.to_numeric(df["beer_abv"], errors="coerce")
    df = df.dropna(subset=["beer_name", "beer_style", *SCORE_COLUMNS])
    if df.empty:
        raise ValueError("Nenhuma linha válida para indexação; a coleção atual foi mantida.")

    print(f"Conectando ao ChromaDB em {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    token = uuid.uuid4().hex[:12]
    staging_name = f"beer_stage_{token}"
    backup_name = f"beer_backup_{token}"
    staging = client.create_collection(name=staging_name, embedding_function=embedder)
    old_renamed = False
    try:
        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start : start + BATCH_SIZE]
            records = [_record(row) for _, row in batch.iterrows()]
            staging.add(
                ids=[record[0] for record in records],
                documents=[record[1] for record in records],
                metadatas=[record[2] for record in records],
            )
            print(f"Indexadas {min(start + len(batch), len(df))}/{len(df)}")
        if staging.count() != len(df):
            raise RuntimeError(f"Validação da staging falhou: {staging.count()} de {len(df)} registros.")

        try:
            current = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            current = None
        if current is not None:
            current.modify(name=backup_name)
            old_renamed = True
        staging.modify(name=COLLECTION_NAME)
        if old_renamed:
            client.delete_collection(name=backup_name)
        print(f"Ingestão concluída: {len(df)} registros em '{COLLECTION_NAME}'.")
        return len(df)
    except Exception:
        # Retain the previous index if validation or the collection swap fails.
        try:
            client.delete_collection(name=staging_name)
        except Exception:
            pass
        if old_renamed:
            try:
                client.get_collection(name=COLLECTION_NAME)
            except Exception:
                client.get_collection(name=backup_name).modify(name=COLLECTION_NAME)
        raise


if __name__ == "__main__":
    ingest_data()
