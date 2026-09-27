"""Aggregate raw BeerAdvocate reviews into one reproducible row per beer."""

from pathlib import Path

import pandas as pd

from settings import settings

INPUT_PATH = settings.raw_data_path
OUTPUT_PATH = settings.prepared_data_path
SAMPLE_SIZE = settings.sample_size
SAMPLE_SEED = settings.sample_seed
SCORE_COLUMNS = ["review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"]
IDENTITY_COLUMNS = ["brewery_name", "beer_name", "beer_style"]
REQUIRED_COLUMNS = {"beer_name", "beer_style", "beer_abv", *SCORE_COLUMNS}


def prepare_data(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset bruto não encontrado: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    raw_rows = len(df)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {', '.join(missing)}")

    optional = [column for column in IDENTITY_COLUMNS if column in df.columns]
    has_beer_id = "beer_beerid" in df.columns
    identity = ["beer_beerid"] if has_beer_id else optional
    columns = sorted(REQUIRED_COLUMNS | set(optional) | set(identity))
    df = df.loc[:, columns].copy()
    for column in ["beer_name", "beer_style", *optional]:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    for column in ["beer_abv", *SCORE_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["beer_name", "beer_style", "review_overall"])
    if df.empty:
        raise ValueError("Nenhuma avaliação com identidade de cerveja e nota geral válida foi encontrada.")

    if has_beer_id:
        fallback_key = df[optional].fillna("").astype(str).agg("|".join, axis=1)
        beer_id = df["beer_beerid"].astype("string")
        df["_beer_key"] = beer_id.where(beer_id.notna() & beer_id.ne(""), "fallback:" + fallback_key)
        group_columns = ["_beer_key"]
    else:
        group_columns = identity.copy()
    aggregations = {column: (column, "mean") for column in SCORE_COLUMNS}
    aggregations["beer_abv"] = ("beer_abv", "median")
    aggregations["review_count"] = ("review_overall", "count")
    aggregations["review_overall_std"] = ("review_overall", "std")
    if "beer_beerid" in df.columns:
        aggregations["beer_beerid"] = ("beer_beerid", "first")
        aggregations["beer_name"] = ("beer_name", "first")
        aggregations["beer_style"] = ("beer_style", "first")
        if "brewery_name" in df.columns:
            aggregations["brewery_name"] = ("brewery_name", "first")
    grouped = df.groupby(group_columns, dropna=False, sort=False).agg(**aggregations).reset_index()
    if "_beer_key" in grouped.columns:
        grouped = grouped.drop(columns="_beer_key")
    grouped["review_overall_std"] = grouped["review_overall_std"].fillna(0.0)

    if len(grouped) > SAMPLE_SIZE:
        # Sample aggregated beers, never individual reviews, with a fixed seed.
        grouped = grouped.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
    grouped = grouped.sort_values(["beer_name", "beer_style"], kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        grouped.to_csv(temporary_path, index=False, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Avaliações brutas: {raw_rows}")
    print(f"Cervejas agregadas: {len(grouped)} (amostra máxima {SAMPLE_SIZE}, seed {SAMPLE_SEED})")
    print(f"ABV ausente: {int(grouped['beer_abv'].isna().sum())}; avaliações médias por cerveja: {grouped['review_count'].mean():.1f}")
    print(f"Arquivo salvo em {output_path}")
    return grouped


if __name__ == "__main__":
    prepare_data()
