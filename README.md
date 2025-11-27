# databend-aiserver

[Databend](https://github.com/databendlabs/databend) AI Server extends any data warehouse with
AI-ready UDFs, seamlessly fusing object storage, embeddings, and SQL pipelines.

## UDFs (prefix `ai_`)
- `list_files(stage, limit)` – UDTF that emits one row per object in external stages.
- `embed_1024(text)` – 1024-dim embeddings (batch-friendly, default model qwen).
- `parse_document(stage, path)` – parse a document via Docling and return Markdown layout as VARIANT.

## Quickstart
```bash
uv sync
uv run databend-aiserver --port 8815
```

## Sample SQL
```sql
CREATE CONNECTION my_s3_connection
  STORAGE_TYPE = 's3'
  ACCESS_KEY_ID = '<your-access-key-id>'
  SECRET_ACCESS_KEY = '<your-secret-access-key>';

CREATE STAGE docs_stage
  URL='s3://load/files/'
  CONNECTION = (CONNECTION_NAME = 'my_s3_connection');

SELECT * FROM ai_list_files(@docs_stage, 50);
SELECT ai_embed_1024(doc_body) FROM docs_tbl;
SELECT ai_parse_document(@docs_stage, 'reports/q1.pdf');
```

`ai_list_files` returns columns: `stage`, `relative_path`, `path`, `is_dir`,
`size`, `mode`, `content_type`, `etag`, and `truncated` (true when the optional limit is hit).

## Tests
```bash
# Full suite (CI should run this)
uv run pytest

# Quicker local loop (skips tests marked slow)
uv run pytest -m "not slow"
```

---

Built by the Databend team — engineers who redefine what's possible with data.
