# Human relevance review

The 50 queries in `retrieval_cases.json` currently use beer styles as proxy labels. To replace that proxy with human judgments, review the pooled candidate list from both retrieval systems.

## Create the review sheet

Start ChromaDB, then run this command from the repository root:

```powershell
.\venv\Scripts\python.exe -m data_pipeline.evaluate_retrieval `
  --cases evaluation/retrieval_cases.json `
  --output evaluation/retrieval_report.json `
  --review-template evaluation/human_review_template.csv `
  -k 5
```

The CSV contains the unique union of the top five results from vector similarity and the current reranker for each query. Rank columns show which system returned each beer and at what position. ABV, average rating, and review count are included when available.

## Label the candidates

Open `evaluation/human_review_template.csv` in Excel or another spreadsheet editor. For every row, fill in:

- `relevance`: `1` if the beer satisfies the query's main request and its explicit constraints; `0` if it does not.
- `notes`: briefly record why a result is relevant or irrelevant, especially for ambiguous cases.

Judge each beer against the full query, not just its style. For example, a beer in the requested style is not relevant if it violates an explicit ABV limit. Do not infer aroma or flavor details from ABV or ratings alone; use reliable product or review information when those details matter. If evidence is insufficient, note that and resolve the judgment before calculating metrics.

Save the completed sheet as `evaluation/human_review.csv`, keeping the column names unchanged and preserving every candidate row. The evaluator rejects missing judgments for top-five results.

## Evaluate the human judgments

```powershell
.\venv\Scripts\python.exe -m data_pipeline.evaluate_retrieval `
  --cases evaluation/retrieval_cases.json `
  --human-judgments evaluation/human_review.csv `
  --output evaluation/retrieval_report_human.json `
  -k 5
```

The report marks `evaluation_mode` as `human_pooled_judgments`. Precision@5, hit rate, reciprocal rank, and nDCG@5 use the reviewed top-five results. Recall uses the relevant beers in the judged pool, which is the union of both systems' top-five candidates; it does not claim recall over all 5,000 catalog items. Keep the completed judgment sheet with the report so the result can be audited.
