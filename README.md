# databend-aiserver

[Databend](https://github.com/databendlabs/databend) AI Server extends any data warehouse with
AI-ready UDFs, seamlessly fusing object storage, embeddings, and SQL pipelines.

## UDFs (prefix `aiserver_`)
- `list_stage_files(stage, limit)` – enumerate objects from external stages.
- `read_pdf(stage, path)` – extract PDF text.
- `read_docx(stage, path)` – extract DOCX text.
- `vector_embed_text_1024(model, text)` – 1024-dim embeddings (batch-friendly).

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

SELECT * FROM aiserver_list_stage_files(@docs_stage, 50);
SELECT aiserver_read_pdf(@docs_stage, 'reports/q1.pdf');
SELECT aiserver_read_docx(@docs_stage, 'reports/q1.docx');
SELECT aiserver_vector_embed_text_1024('qwen', doc_body) FROM docs_tbl;
```

## Tests
```bash
uv run pytest
uv run pytest -m "not slow"
```

---

Built by the Databend team — engineers who redefine what's possible with data.
