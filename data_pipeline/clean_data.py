"""Validate and prepare the BeerAdvocate CSV for reproducible ingestion."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/raw/beers_dataset.csv")
OUTPUT_PATH = Path("data/raw/beers_cleaned.csv")
SAMPLE_SIZE = 5_000
SAMPLE_SEED = 42
REQUIRED_COLUMNS = {
    "beer_name", "beer_style", "beer_abv", "review_overall",
    "review_aroma", "review_appearance", "review_palate", "review_taste",
}
OPTIONAL_COLUMNS = {"brewery_name"}
SCORE_COLUMNS = ["review_overall", "review_aroma", "review_appearance", "review_palate", "review_taste"]


def prepare_data(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset bruto não encontrado: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {', '.join(missing)}")

    columns = sorted(REQUIRED_COLUMNS | (OPTIONAL_COLUMNS & set(df.columns)))
    df = df.loc[:, columns].copy()
    for col in ["beer_name", "beer_style", *OPTIONAL_COLUMNS]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace("", pd.NA)
    for col in ["beer_abv", *SCORE_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with usable identity and ratings; an unknown ABV is valid and stays null.
    df = df.dropna(subset=["beer_name", "beer_style", *SCORE_COLUMNS])
    df = df.drop_duplicates(subset=["beer_name", "beer_style"], keep="first")
    if len(df) > SAMPLE_SIZE:
        # Fixed seed makes the capped catalog repeatable and avoids source-order bias.
        df = df.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
    df = df.sort_values(["beer_name", "beer_style"], kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        df.to_csv(temporary_path, index=False, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Linhas lidas: {len(pd.read_csv(input_path, usecols=['beer_name']))}")
    print(f"Cervejas preparadas: {len(df)} (amostra máxima {SAMPLE_SIZE}, seed {SAMPLE_SEED})")
    print(f"ABV ausente: {int(df['beer_abv'].isna().sum())}; salvo em {output_path}")
    return df


if __name__ == "__main__":
    prepare_data()
