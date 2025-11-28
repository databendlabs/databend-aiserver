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

"""Unit tests for opendal API usage to catch API changes."""

import pytest
from opendal import Operator


def test_opendal_entry_has_path_attribute():
    """Verify Entry objects have path attribute."""
    op = Operator("memory")
    op.write("test.txt", b"hello")
    
    entries = list(op.list(""))
    assert len(entries) > 0
    
    entry = entries[0]
    assert hasattr(entry, "path")
    assert entry.path == "test.txt"


def test_opendal_entry_has_metadata_attribute():
    """Verify Entry objects have metadata attribute."""
    op = Operator("memory")
    op.write("test.txt", b"hello")
    
    entries = list(op.list(""))
    entry = entries[0]
    
    # Entry SHOULD have metadata attribute in newer opendal
    assert hasattr(entry, "metadata")


def test_opendal_stat_returns_metadata():
    """Verify stat() returns Metadata object."""
    op = Operator("memory")
    op.write("test.txt", b"hello world")
    
    metadata = op.stat("test.txt")
    
    # Verify Metadata has expected attributes
    assert hasattr(metadata, "content_length")
    assert hasattr(metadata, "content_type")
    assert hasattr(metadata, "etag")
    assert hasattr(metadata, "mode")
    
    # Verify values
    assert metadata.content_length == 11  # len("hello world")


def test_opendal_metadata_has_is_dir_method():
    """Verify Metadata has is_dir() method."""
    op = Operator("memory")
    op.write("test.txt", b"hello")
    
    metadata = op.stat("test.txt")
    
    # Metadata SHOULD have is_dir() method in newer opendal
    assert hasattr(metadata, "is_dir")


def test_opendal_directory_detection_via_path():
    """Verify directories are detected by path ending with /."""
    op = Operator("memory")
    op.create_dir("mydir/")
    op.write("mydir/file.txt", b"content")
    
    entries = list(op.list(""))
    
    # Find directory entry
    dir_entries = [e for e in entries if e.path.endswith("/")]
    assert len(dir_entries) > 0
    
    # Directories end with /
    for entry in dir_entries:
        assert entry.path.endswith("/")
