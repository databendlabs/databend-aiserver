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

from databend_aiserver.stages.operator import _build_s3_options


def test_s3_options_default_to_anonymous_without_creds():
    opts = _build_s3_options({"type": "s3", "bucket": "public-bucket"})

    assert opts["allow_anonymous"] == "true"
    assert opts["disable_config_load"] == "true"
    assert opts["disable_ec2_metadata"] == "true"
    assert opts["region"] == "us-east-1"


def test_s3_options_allow_anonymous_even_with_creds():
    opts = _build_s3_options(
        {
            "type": "s3",
            "bucket": "private-bucket",
            "access_key_id": "ak",
            "secret_access_key": "sk",
        }
    )

    assert opts["allow_anonymous"] == "true"
    assert "disable_config_load" not in opts
    assert "disable_ec2_metadata" not in opts


def test_s3_options_respect_explicit_flags():
    opts = _build_s3_options(
        {
            "type": "s3",
            "bucket": "bucket",
            "allow_anonymous": False,
            "disable_credential_loader": False,
        }
    )

    assert opts["allow_anonymous"] == "false"
    assert opts["disable_credential_loader"] == "false"
