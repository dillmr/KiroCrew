"""Agent spec hooks on goose and opencode: Crew's turn loop runs them.

Neither harness receives the spec's ``hooks`` block (its ``session/new`` element set
has no field for it), so Crew fires it, as it does for KAS. Three things have to
hold for that to be true rather than claimed:

* the backend is a member of ``ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS``, so the chat
  turn loop and the subagent / task-runner gate read the spec at all;
* a PreToolUse matcher in kiro-cli's names (``execute_bash``) meets the harness's
  own tool, which needs the permission event to say which tool it is -- read here
  from the committed LIVE frames of each harness, not from a hand-built frame;
* the hook can BLOCK that call, on the permission request.

claude and codex stay out: each approves some calls inside the harness without
asking, so a PreToolUse hook would be skipped on exactly those.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kiro_crew.config.paths as paths_mod
import kiro_crew.hooks as hooks_mod
from kiro_crew.acp import _dispatch
from kiro_crew.acp.types import EVENT_PERMISSION_REQUEST, JsonRpcMessage
from kiro_crew.agent_sdk import spec_hooks
from kiro_crew.agent_sdk.backends import (
    ACP_BACKEND_CLAUDE,
    ACP_BACKEND_CODEX,
    ACP_BACKEND_GOOSE,
    ACP_BACKEND_OPENCODE,
    ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS,
)
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.dashboard import chat_runner
from kiro_crew.hooks import (
    HOOK_EVENT_AGENT_SPAWN,
    HOOK_EVENT_POST_TOOL_USE,
    HOOK_EVENT_PRE_TOOL_USE,
    HOOK_EVENT_STOP,
    HOOK_EVENT_USER_PROMPT_SUBMIT,
    ScriptHookResult,
    ScriptHookStore,
)
from kiro_crew.providers.mirrors import mirror_for
from kiro_crew.providers.mirrors.base import Concern, Disposition

_FRAMES = Path(__file__).parent / "fixtures" / "acp_frames"

#: backend -> (the live capture holding a shell call and its permission request,
#: the harness's own name for that shell tool).
_SHELL_CAPTURE = {
    ACP_BACKEND_GOOSE: ("goose/turn-live.jsonl", "shell"),
    ACP_BACKEND_OPENCODE: ("opencode/permission-request-live.jsonl", "bash"),
}

_MIRRORS = sorted(_SHELL_CAPTURE)


@pytest.fixture(autouse=True)
def _fresh_cache():
    spec_hooks._cache.clear()
    yield
    spec_hooks._cache.clear()


@pytest.fixture
def agents_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    monkeypatch.setattr(paths_mod, "kiro_agents_dir", lambda: d)
    return d


def _write_spec(agents_dir: Path, name: str, hooks: dict) -> None:
    spec = {"name": name, "prompt": "p", "hooks": hooks}
    (agents_dir / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")


def _provider(backend: str, cwd: str = "/w") -> SimpleNamespace:
    return SimpleNamespace(capabilities=capabilities_for(backend), cwd=cwd)


def _permission_event(backend: str, *, as_backend: str | None = None):
    """Replay a live capture through the real parsers; return its permission event.

    The ``tool_call`` frames go through ``parse_session_update`` with the caches a
    session handle owns, then the ``session/request_permission`` frame through
    ``build_permission_event``, which is the path every mirror session takes.
    """
    capture, _ = _SHELL_CAPTURE[backend]
    caches: dict = {
        "tool_input_cache": {},
        "shell_cache": {},
        "raw_params_cache": {},
        "mcp_server_name_cache": {},
        "tool_name_cache": {},
        "tool_input_redacted_cache": {},
        "diff_path_cache": {},
        "harness_tool_name_cache": {},
    }
    for line in (_FRAMES / capture).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        method = frame.get("method")
        params = frame.get("params") or {}
        if method == "session/update":
            _dispatch.parse_session_update(params.get("update") or {}, **caches)
        elif method == "session/request_permission":
            msg = JsonRpcMessage(id=frame["id"], method=method, params=params)
            event, _ = _dispatch.build_permission_event(
                msg,
                tool_input_cache=caches["tool_input_cache"],
                tool_input_redacted_cache=caches["tool_input_redacted_cache"],
                shell_cache=caches["shell_cache"],
                raw_params_cache=caches["raw_params_cache"],
                mcp_server_name_cache=caches["mcp_server_name_cache"],
                tool_name_cache=caches["tool_name_cache"],
                diff_path_cache=caches["diff_path_cache"],
                harness_tool_name_cache=caches["harness_tool_name_cache"],
                harness_backend=as_backend if as_backend is not None else backend,
            )
            assert event is not None and event.kind == EVENT_PERMISSION_REQUEST
            return event
    raise AssertionError(f"no permission request in {capture}")


def _fake_runs(monkeypatch, exit_code: int) -> list:
    ran: list = []

    async def fake_run(hook, context="", hook_event=None, cwd=None):
        ran.append((hook.event, hook.command, (hook_event or {}).get("tool_name"), cwd))
        return ScriptHookResult(
            hook_id=hook.id, hook_name=hook.name, event=hook.event, exit_code=exit_code
        )

    monkeypatch.setattr(hooks_mod, "run_script_hook", fake_run)
    return ran


# ── membership: who Crew fires spec hooks for ──


@pytest.mark.parametrize("backend", _MIRRORS)
def test_crew_fires_the_spec_hooks_of_goose_and_opencode(backend):
    assert backend in ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS
    assert capabilities_for(backend).crew_fires_spec_hooks is True


@pytest.mark.parametrize("backend", [ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX])
def test_claude_and_codex_stay_out_until_their_auto_approved_calls_ask(backend):
    # Each approves some calls inside the harness, which then sends no permission
    # request, so a PreToolUse hook would be skipped on those calls.
    assert backend not in ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS
    hooks = mirror_for(backend).rulings()[Concern.HOOKS]
    assert hooks.disposition is Disposition.NO_CHANNEL


@pytest.mark.parametrize("backend", _MIRRORS)
def test_the_mirror_rules_hooks_translated(backend):
    hooks = mirror_for(backend).rulings()[Concern.HOOKS]
    assert hooks.disposition is Disposition.TRANSLATED
    assert "harness_tool_names" in hooks.reason


# ── the permission event names the harness's own tool ──


@pytest.mark.parametrize("backend", _MIRRORS)
def test_a_live_shell_permission_request_carries_the_harness_tool(backend):
    _, native = _SHELL_CAPTURE[backend]
    event = _permission_event(backend)
    assert event.harness_tool_id == f"{backend}#{native}"
    assert spec_hooks.spec_hook_tool_names(event.harness_tool_id)[0] == "execute_bash"


@pytest.mark.parametrize("backend", _MIRRORS)
def test_a_backend_with_no_table_gets_no_harness_tool_id(backend):
    # The same frames on a backend with no name table build the event as before.
    assert _permission_event(backend, as_backend=ACP_BACKEND_CLAUDE).harness_tool_id == ""


def test_opencodes_later_titles_name_the_command_not_the_tool():
    # Only the first tool_call frame's title is the tool; the in_progress update
    # that follows reuses the title for the command and must not replace it.
    update = {"sessionUpdate": "tool_call", "toolCallId": "c1", "title": "bash", "kind": "execute"}
    names: dict = {}
    _dispatch.parse_session_update(update, harness_tool_name_cache=names)
    refine = {**update, "sessionUpdate": "tool_call_update", "title": "rm -rf build"}
    _dispatch.parse_session_update(refine, harness_tool_name_cache=names)
    assert names == {"c1": "bash"}
    # A prose title is not a name at all.
    prose = {**update, "toolCallId": "c2", "title": "shell · echo hi"}
    _dispatch.parse_session_update(prose, harness_tool_name_cache=names)
    assert names["c2"] == ""


# ── the hook fires and blocks on that permission request ──


@pytest.mark.parametrize("backend", _MIRRORS)
@pytest.mark.parametrize("matcher", ["execute_bash", "shell"])
def test_a_pre_tool_use_spec_hook_blocks_the_live_shell_call(
    backend, matcher, agents_dir, tmp_path, monkeypatch
):
    _write_spec(agents_dir, "a1", {"preToolUse": [{"matcher": matcher, "command": "deny.sh"}]})
    ran = _fake_runs(monkeypatch, exit_code=2)
    turn = asyncio.run(spec_hooks.turn_spec_hooks(_provider(backend), "a1"))
    assert turn.gated and not turn.unreadable and len(turn.hooks) == 1
    event = _permission_event(backend)
    reason = asyncio.run(
        hooks_mod.permission_pre_tool_block(
            ScriptHookStore(config_dir=tmp_path),
            turn.hooks,
            turn.cwd,
            event.title,
            event.tool_input,
            tool_identity=event.tool_name,
            mcp_server=event.mcp_server_name,
            harness_tool_id=event.harness_tool_id,
        )
    )
    # The hook ran, told the call is kiro-cli's execute_bash, and its exit 2 blocked.
    assert ran == [(HOOK_EVENT_PRE_TOOL_USE, "deny.sh", "execute_bash", "/w")]
    assert reason is not None


@pytest.mark.parametrize("backend", _MIRRORS)
def test_a_pre_tool_use_hook_for_another_tool_leaves_the_shell_call_alone(
    backend, agents_dir, tmp_path, monkeypatch
):
    _write_spec(agents_dir, "a1", {"preToolUse": [{"matcher": "fs_write", "command": "deny.sh"}]})
    ran = _fake_runs(monkeypatch, exit_code=2)
    turn = asyncio.run(spec_hooks.turn_spec_hooks(_provider(backend), "a1"))
    event = _permission_event(backend)
    reason = asyncio.run(
        hooks_mod.permission_pre_tool_block(
            ScriptHookStore(config_dir=tmp_path),
            turn.hooks,
            turn.cwd,
            event.title,
            event.tool_input,
            tool_identity=event.tool_name,
            mcp_server=event.mcp_server_name,
            harness_tool_id=event.harness_tool_id,
        )
    )
    assert ran == [] and reason is None


# ── the other four events reach the chat turn loop's fire ──


@pytest.mark.parametrize("backend", _MIRRORS)
def test_the_chat_turn_loop_fires_the_other_four_events(backend, agents_dir, tmp_path, monkeypatch):
    _write_spec(
        agents_dir,
        "a1",
        {
            "agentSpawn": [{"command": "spawn.sh"}],
            "userPromptSubmit": [{"command": "prompt.sh"}],
            "postToolUse": [{"command": "post.sh"}],
            "stop": [{"command": "stop.sh"}],
        },
    )
    ran = _fake_runs(monkeypatch, exit_code=0)
    hooks, unreadable, cwd = asyncio.run(
        chat_runner._prepare_spec_hooks(None, None, _provider(backend), "a1", is_new=False)
    )
    assert not unreadable and cwd == "/w"
    store = ScriptHookStore(config_dir=tmp_path)
    for event in (
        HOOK_EVENT_AGENT_SPAWN,
        HOOK_EVENT_USER_PROMPT_SUBMIT,
        HOOK_EVENT_POST_TOOL_USE,
        HOOK_EVENT_STOP,
    ):
        asyncio.run(store.fire(event, "ctx", extra_hooks=hooks, extra_hooks_cwd=cwd))
    assert [(e, c) for e, c, _, _ in ran] == [
        (HOOK_EVENT_AGENT_SPAWN, "spawn.sh"),
        (HOOK_EVENT_USER_PROMPT_SUBMIT, "prompt.sh"),
        (HOOK_EVENT_POST_TOOL_USE, "post.sh"),
        (HOOK_EVENT_STOP, "stop.sh"),
    ]


def _bare_client(backend: str):
    """An ``AcpClient`` with only the tool-call caches, as the goose tests build one."""
    from kiro_crew.acp.client import AcpClient
    from kiro_crew.acp.types import AcpPromptStats

    client = AcpClient.__new__(AcpClient)
    client._acp_backend = backend
    client._available_mode_ids = []
    client._modes_advertised = False
    client._session_key = ""
    client._agent = ""
    client._tool_call_inputs = {}
    client._tool_call_input_redacted = {}
    client._tool_call_is_shell = {}
    client._tool_call_unclassified = {}
    client._tool_call_mcp_server = {}
    client._tool_call_tool_name = {}
    client._tool_call_harness_tool_name = {}
    client._tool_call_params = {}
    client._tool_call_diff_path = {}
    client._permission_options = {}
    client.last_prompt_stats = AcpPromptStats()
    return client


@pytest.mark.parametrize("backend", _MIRRORS)
def test_the_direct_client_names_the_harness_tool_too(backend):
    # AcpClient parses tool_call frames on its own path, not parse_session_update.
    capture, native = _SHELL_CAPTURE[backend]
    client = _bare_client(backend)
    event = None
    for line in (_FRAMES / capture).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame = json.loads(line)
        method = frame.get("method")
        if method == "session/update":
            client._extract_tool_event(JsonRpcMessage(method=method, params=frame["params"]))
        elif method == "session/request_permission":
            msg = JsonRpcMessage(id=frame["id"], method=method, params=frame["params"])
            event = client._build_permission_event(msg)
            break
    assert event is not None
    assert event.harness_tool_id == f"{backend}#{native}"
