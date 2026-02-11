# Copyright 2025 ModelBest Inc. and/or its affiliates
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

"""SQLBench apigen environment wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from MUA_environments.base.environment import BaseEnvironment
from MUA_environments.base.tool_registry import ToolRegistry

from .data_loader import SQLBenchApigenDataLoader


class SQLBenchApigenEnvironment(BaseEnvironment):
    """Environment that exposes TauBench airline SQL via sqlexe/getdbschema."""

    @property
    def environment_name(self) -> str:
        return "sqlbench_apigen"

    @property
    def environment_type(self) -> str:
        return "sqlbench_apigen"

    def get_data_loader(self) -> SQLBenchApigenDataLoader:
        return SQLBenchApigenDataLoader(self.config)

    def get_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry(self.config)
        self._register_tools(registry)
        return registry

    def _register_tools(self, registry: ToolRegistry) -> None:
        from verl.tools.sqlbench_apigen.sqlexe import SQLExeTool
        from verl.tools.sqlbench_apigen.get_dbschema import GetDBSchemaTool

        shared = self.get_shared_data()
        db_path = Path(shared["db_path"]).resolve()
        schema_path = Path(shared["schema_path"]).resolve()
        db_name = shared.get("db_name")

        sql_config: Dict[str, Any] = {
            "db_root": str(db_path.parent),
            "timeout": self.config.get("timeout", 5),
            "db_name": db_name,
        }
        schema_config: Dict[str, Any] = {
            "data_root": str(schema_path.parent),
            "timeout": self.config.get("schema_timeout", self.config.get("timeout", 5)),
            "strip_inserts": self.config.get("strip_inserts", True),
        }

        registry.register_tool_class("sqlexe", SQLExeTool, sql_config)
        registry.register_tool_class("getdbschema", GetDBSchemaTool, schema_config)

    def get_shared_data(self) -> Dict[str, Any]:
        return super().get_shared_data()
