# databend-aiserver

[Databend](https://github.com/databendlabs/databend) AI Server extends any data warehouse with AI-ready UDFs, seamlessly fusing object storage, embeddings, and SQL pipelines.

## Quickstart

```bash
uv sync
uv run databend-aiserver --port 8815
```

## AI Functions

| Function | Signature | Purpose | Output |
| :--- | :--- | :--- | :--- |
| **ai_list_files** | `(stage_location, max_files)` | List objects in a stage for inspection/sampling. | Table with file details (`path`, `size`, etc.) |
| **ai_embed_1024** | `(text)` | Generate 1024-dim embeddings (default: Qwen). | `VECTOR(1024)` |
| **ai_parse_document** | `(stage_location, path)` | Parse docs (PDF, DOCX, Images, etc.) to Markdown. | `VARIANT` (pages, metadata) |

## Usage

### 1. Register Functions in Databend

```sql
CREATE OR REPLACE FUNCTION ai_list_files(stage_location STAGE_LOCATION, max_files INT)
RETURNS TABLE (stage_name VARCHAR, relative_path VARCHAR, path VARCHAR, is_dir BOOLEAN, size BIGINT, mode VARCHAR, content_type VARCHAR, etag VARCHAR, truncated BOOLEAN)
LANGUAGE PYTHON HANDLER = 'ai_list_files' ADDRESS = '<your-ai-server-address>';

CREATE OR REPLACE FUNCTION ai_embed_1024(VARCHAR)
RETURNS VECTOR(1024)
LANGUAGE PYTHON HANDLER = 'ai_embed_1024' ADDRESS = '<your-ai-server-address>';

CREATE OR REPLACE FUNCTION ai_parse_document(stage_location STAGE_LOCATION, path VARCHAR)
RETURNS VARIANT
LANGUAGE PYTHON HANDLER = 'ai_parse_document' ADDRESS = '<your-ai-server-address>';
```

### 2. Run Queries

```sql
-- Setup Stage
CREATE CONNECTION my_s3_conn STORAGE_TYPE = 's3' ACCESS_KEY_ID = '...' SECRET_ACCESS_KEY = '...';
CREATE STAGE docs_stage URL='s3://load/files/' CONNECTION = (CONNECTION_NAME = 'my_s3_conn');

-- Execute AI Functions
SELECT * FROM ai_list_files(@docs_stage, 50);
SELECT ai_embed_1024(doc_body) FROM docs_tbl;
SELECT ai_parse_document(@docs_stage, 'reports/q1.pdf');
```

## Development

```bash
# Run full test suite
uv run pytest
```

---

Built by the [Databend](https://github.com/databendlabs/databend) team.
