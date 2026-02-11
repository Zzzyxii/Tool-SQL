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

from typing import Any, Dict

from MUA_environments.base.environment import BaseEnvironment
from MUA_environments.base.tool_registry import ToolRegistry
from .data_loader import SQLBenchCoSQLDataLoader


class SQLBenchCoSQLEnvironment(BaseEnvironment):
    """COSQL multi-database environment.

    Provides shared `db_root` information. Individual tool calls specify the
    concrete `db_name` they want to operate on.
    """

    @property
    def environment_name(self) -> str:  # runtime identifier
        return "sqlbench_cosql"

    @property
    def environment_type(self) -> str:  # mapping key used in factory / manager
        return "cosql"

    def get_data_loader(self) -> SQLBenchCoSQLDataLoader:
        return SQLBenchCoSQLDataLoader(self.config)

    def get_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry(self.config)
        self._register_cosql_tools(registry)
        return registry

    def _register_cosql_tools(self, registry: ToolRegistry) -> None:
        # Import cosql-specific tools (names must match tool_config and dataset actions)
        from verl.tools.sqlbench_cosql.get_dbschema import GetDBSchemaTool
        from verl.tools.sqlbench_cosql.sqlexe import SQLExeTool

        shared = self.get_shared_data()
        tool_config: Dict[str, Any] = {"db_root": shared.get("db_root")}
        schema_config: Dict[str, Any] = {"data_root": shared.get("db_root")}

        tools_to_register = [
            ("sqlexe", SQLExeTool, tool_config),
            ("getdbschema", GetDBSchemaTool, schema_config),
        ]
        for tool_name, tool_class, config in tools_to_register:
            registry.register_tool_class(tool_name, tool_class, config)
