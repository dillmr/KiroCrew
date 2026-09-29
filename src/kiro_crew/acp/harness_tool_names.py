"""Name tables from kiro-cli's tool names to goose's and opencode's own.

A spec hook's tool matcher is written in kiro-cli's names (``execute_bash``,
``fs_write``). KAS states its own id for a call and
:func:`kiro_crew.acp.kas_permissions.kas_tool_match_names` maps it back. goose and
opencode state no such id on a permission request, but each names the tool on the
``tool_call`` frame that comes first:

* goose in ``_meta.goose.toolCall.toolName`` (``shell`` for its builtin shell);
* opencode in the frame's ``title``, which on that first frame is the tool's own
  name (``bash``); later updates reuse the title for the command.

The permission event carries that name as its ``harness_tool_id``, joined to the
backend by :data:`HARNESS_TOOL_ID_SEPARATOR` (``goose#shell``). A KAS id never
contains that character (``_dispatch._HARNESS_TOOL_ID_RE``), so a KAS id can never
be read through these tables, and a goose id never through KAS's.

Every id below is a tool the live harness offered: goose 1.50.1's tool list as its
request log sent it to the model, and opencode 1.18.30's as ``opencode debug agent
build`` resolved it. A tool with no row (goose's ``analyze``, opencode's
``todowrite``) is met only by a matcher that names it as written.
"""

from __future__ import annotations

from kiro_crew.acp.kas_permissions import KIRO_TOOL_ALIASES
from kiro_crew.acp_backends import ACP_BACKEND_GOOSE, ACP_BACKEND_OPENCODE

#: Joins a backend to the tool name its frame stated. Outside the characters a KAS
#: ``toolId`` may carry, so the two id spaces cannot meet.
HARNESS_TOOL_ID_SEPARATOR = "#"

#: kiro-cli tool name -> the goose builtin tools that do the same job.
GOOSE_TOOL_IDS_BY_KIRO_TOOL: dict[str, tuple[str, ...]] = {
    "execute_bash": ("shell",),
    "fs_read": ("tree", "read_image"),
    "fs_write": ("write", "edit"),
    "use_subagent": ("delegate",),
}

#: kiro-cli tool name -> the opencode tools that do the same job.
OPENCODE_TOOL_IDS_BY_KIRO_TOOL: dict[str, tuple[str, ...]] = {
    "execute_bash": ("bash",),
    "fs_read": ("read",),
    "fs_write": ("write", "edit"),
    "grep": ("grep",),
    "glob": ("glob",),
    "web_fetch": ("webfetch",),
    "web_search": ("websearch",),
    "use_subagent": ("task",),
}

#: The backends whose permission event carries a qualified harness tool id.
HARNESS_TOOL_TABLES: dict[str, dict[str, tuple[str, ...]]] = {
    ACP_BACKEND_GOOSE: GOOSE_TOOL_IDS_BY_KIRO_TOOL,
    ACP_BACKEND_OPENCODE: OPENCODE_TOOL_IDS_BY_KIRO_TOOL,
}

#: Every harness id in these tables, for the spec reader's "names no tool" warning.
HARNESS_TOOL_MATCH_VOCABULARY: frozenset[str] = frozenset(
    tool_id for table in HARNESS_TOOL_TABLES.values() for ids in table.values() for tool_id in ids
)


def qualified_harness_tool_id(backend: str, tool_name: str) -> str:
    """``backend#tool_name`` for a backend with a table, else ``""``."""
    if backend not in HARNESS_TOOL_TABLES or not tool_name:
        return ""
    return f"{backend}{HARNESS_TOOL_ID_SEPARATOR}{tool_name}"


def split_harness_tool_id(tool_id: str) -> tuple[str, str] | None:
    """``(backend, tool_name)`` of a qualified id, or ``None`` for any other id."""
    backend, sep, name = tool_id.partition(HARNESS_TOOL_ID_SEPARATOR)
    if not sep or not name or backend not in HARNESS_TOOL_TABLES:
        return None
    return backend, name


def harness_tool_match_names(tool_id: str) -> tuple[str, ...] | None:
    """The names a tool matcher meets for a qualified id, or ``None`` if unqualified.

    Built as :func:`kiro_crew.acp.kas_permissions.kas_tool_match_names` builds KAS's:
    the kiro-cli name whose row reaches the tool first, then the harness's own name,
    then kiro-cli's aliases for that name. So ``execute_bash``, ``shell`` and
    ``bash`` all meet an opencode shell call, and the first name is the one a
    kiro-cli hook would be told the tool is called.
    """
    parts = split_harness_tool_id(tool_id)
    if parts is None:
        return None
    backend, name = parts
    table = HARNESS_TOOL_TABLES[backend]
    kiro_names = sorted(kiro for kiro, ids in table.items() if name in ids)
    aliases = sorted(alias for alias, kiro in KIRO_TOOL_ALIASES.items() if kiro in kiro_names)
    return tuple(dict.fromkeys([*kiro_names, name, *aliases]))
