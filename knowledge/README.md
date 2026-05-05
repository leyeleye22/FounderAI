# Knowledge folder

This folder stores the retrieval corpus used by FounderAI.

Recommended sub-folders:

- `problem_statement/`
- `interviews/`
- `bmc/`
- `competitors/`
- `market_size/`
- `roi/`
- `sprints/`

Each document should capture:

- field goal
- rules
- good examples
- bad examples
- challenge prompts
- formulas if needed

## File format

Use JSON files containing one object or a list of objects with:

- `document_id`
- `module`
- `field`
- `title`
- `content`
- `tags`
- `language`
- `source_type`

## Ingestion

Run:

`python scripts/ingest_knowledge.py`

This ingests the local `knowledge/` corpus into the persistent Chroma store.
