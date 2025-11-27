# databend-aiserver

[Databend](https://github.com/databendlabs/databend) AI Server extends any data warehouse with
AI-ready UDFs, seamlessly fusing object storage, embeddings, and SQL pipelines.

## UDFs (prefix `ai_`)

### ai_list_files
- **Purpose:** List objects in a stage for quick inspection or sampling of large buckets.
- **Signature:** `ai_list_files(stage STAGE_LOCATION, limit INT)` (UDTF)
- **Columns:** `stage, relative_path, path, is_dir, size, mode, content_type, etag, truncated`.
- **Logging:** Emits start log with stage/limit and a completion log with entry count, truncated flag, prefix, and duration—helps spot slow scans.
- **Example output row:** `{stage:"docs_stage", relative_path:"reports/", path:"reports/", is_dir:true, size:null, truncated:false}`
- **Create in Databend:**
  ```sql
  CREATE FUNCTION ai_list_files(stage STAGE_LOCATION, limit INT)
  RETURNS TABLE (
      stage         VARCHAR,
      relative_path VARCHAR,
      path          VARCHAR,
      is_dir        BOOLEAN,
      size          BIGINT NULL,
      mode          VARCHAR NULL,
      content_type  VARCHAR NULL,
      etag          VARCHAR NULL,
      truncated     BOOLEAN
  );
  ```

### ai_embed_1024
- **Purpose:** Generate 1024-dim embeddings (default alias `qwen` → `Qwen/Qwen3-Embedding-0.6B`) for text or batches of text.
- **Signature:** `ai_embed_1024(text STRING)` (accepts scalar or array input)
- **Runtime:** Device auto-selected at server start (CPU/GPU/MPS) via runtime detection; logs start (batch size, device kind) and completion latency/device.
- **Output:** `ARRAY(FLOAT NULL)` length 1024 per input row.
- **Example output:** `[0.0123, -0.0456, ..., 0.0031]` (1024 floats)
- **Create in Databend:**
  ```sql
  CREATE FUNCTION ai_embed_1024(text VARCHAR)
  RETURNS ARRAY(FLOAT NULL);
  ```

### ai_parse_document
- **Purpose:** Parse documents to Markdown layout plus per-page content using the Docling backend.
- **Signature:** `ai_parse_document(stage STAGE_LOCATION, path STRING)`
- **Supported formats:** PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, VTT, and common images (PNG, TIFF, JPEG, ...).
- **Runtime:** Device chosen via runtime detection; logs start (path, device kind) and completion (pages, fallback, duration).
- **Output (VARIANT):**
  - `pages`: array of `{index, content}`
  - `metadata`: `{pageCount, chunkingFallback}`
  - `errorInformation`: nullable
- **Example output:**
  ```json
  {
    "pages": [
      {"index": 0, "content": "# Report\nIntroduction..."},
      {"index": 1, "content": "Findings..."}
    ],
    "metadata": {"pageCount": 2, "chunkingFallback": false},
    "errorInformation": null
  }
  ```
- **Create in Databend:**
  ```sql
  CREATE FUNCTION ai_parse_document(stage STAGE_LOCATION, path VARCHAR)
  RETURNS VARIANT;
  ```

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


## Tests
```bash
# Full suite (CI should run this)
uv run pytest

# Quicker local loop (skips tests marked slow)
uv run pytest -m "not slow"
```

---

Built by the Databend team — engineers who redefine what's possible with data.
