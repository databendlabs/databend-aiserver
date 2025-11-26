# Copyright 2025 Databend Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection of UDF implementations exposed by the AI server."""

from .stage import ai_list_files
from .files import ai_read_docx, ai_read_pdf
from .embeddings import ai_embed_1024

__all__ = [
    "ai_list_files",
    "ai_read_pdf",
    "ai_read_docx",
    "ai_embed_1024",
]
