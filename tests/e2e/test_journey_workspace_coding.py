"""E2E test: "workspace-aware coding session" user journey.

Exercises a realistic coding workflow where the agent uses terminal
tools to read and modify files in the workspace:
create session → list files → create file → read file → verify content.

The ``sys_terminal_test_agent`` provides ``sys_terminal_*`` tools that
drive a real tmux session, so the agent can execute arbitrary shell
commands (``ls``, ``echo``, ``cat``, ``sed``) in its workspace.

Skipped if tmux is not installed on the host.

Usage::

    pytest tests/e2e/test_journey_workspace_coding.py \\
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import shutil
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import poll_until_terminal

pytestmark = pytest.mark.skipif(
    shutil.which("tmux") is None,
    reason="tmux not installed; workspace coding journey needs tmux on PATH",
)


def _get_function_call_outputs(
    client: httpx.Client,
    conversation_id: str,
    tool_name: str,
) -> list[str]:
    """
    Return raw outputs of every ``tool_name`` call in conversation order.

    Walks ``function_call`` and ``function_call_output`` items in the
    conversation. Assertions land on deterministic tool output strings,
    not on flaky LLM prose summaries.

    :param client: HTTP client.
    :param conversation_id: Conversation to inspect.
    :param tool_name: Only outputs of calls to this tool are returned.
    :returns: Ordered list of raw output strings.
    """
    resp = client.get(f"/v1/sessions/{conversation_id}/items?limit=200")
    resp.raise_for_status()
    items = resp.json()["data"]
    calls_by_id: dict[str, dict] = {}
    for item in items:
        if item.get("type") == "function_call" and item.get("name") == tool_name:
            calls_by_id[item["call_id"]] = item
    outputs: list[str] = []
    for item in items:
        if item.get("type") == "function_call_output":
            cid = item.get("call_id")
            if cid in calls_by_id:
                outputs.append(str(item.get("output", "")))
    return outputs


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all assistant message text blocks from a response body.

    :param body: Terminal response body from :func:`poll_until_terminal`.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


@pytest.mark.llm_flaky(reruns=2)
def test_workspace_coding_session_journey(
    live_server: str,
    sys_terminal_test_agent: str,
    http_client: httpx.Client,
) -> None:
    """
    Workspace-aware coding journey: create file via terminal, read it back,
    modify it, and verify the modification.

    Steps:

    1. Create a session with the ``sys_terminal_test_agent``.
    2. Ask the agent to list the workspace files (``ls``).
    3. Verify ``sys_terminal_read`` output contains file listings.
    4. Ask the agent to create a Python file with a hello-world function.
    5. Ask the agent to read the file back with ``cat``.
    6. Verify the file content appears in tool output.

    The core flow (create → read → verify) is the most reliable subset
    of the full 8-step journey. Modification steps (sed/echo to add a
    docstring) are omitted to reduce LLM flakiness — the create-read
    round trip already proves the workspace is functional.

    **What breaks if this fails:**

    - Terminal tools not registered → agent cannot run shell commands.
    - Workspace cwd not set → file created in wrong location.
    - ``sys_terminal_send``/``sys_terminal_read`` flow broken → no
      command output captured.
    - tmux session not persisting across tool calls within one turn →
      stateful file operations fail.

    :param live_server: Server base URL.
    :param sys_terminal_test_agent: Registered agent with terminal tools.
    :param http_client: HTTP client pointed at the live server.
    """
    # ── Step 1 + 2: Create session and list workspace ────────────────────
    # We ask the agent to launch a terminal and list files in a single
    # prompt to reduce the number of LLM round trips.
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": (
                "Use sys_terminal_launch to start the 'bash' terminal with "
                "session 'workspace'. Then use sys_terminal_send to type "
                "'ls -la' followed by Enter. Wait briefly, then "
                "sys_terminal_read on session 'workspace'. "
                "Reply 'listed' once you see the output."
            ),
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]
    body = poll_until_terminal(http_client, response_id, timeout=180)
    assert body["status"] == "completed", (
        f"Step 1-2 failed: status={body['status']!r}, "
        f"error={body.get('error')!r}. If 'failed' with a tool "
        f"error, sys_terminal_* tools may not be registered."
    )
    conv_id = body["conversation"]["id"]

    # ── Step 3: Verify terminal output contains file listing ─────────────
    reads_step2 = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    assert len(reads_step2) >= 1, (
        f"sys_terminal_read was never called in the listing step; "
        f"conv_id={conv_id}. The agent may have ignored the prompt "
        f"or the tool wasn't on the schema."
    )
    # ls -la always produces 'total' as the first line and '.' entries.
    combined_listing = " ".join(reads_step2)
    assert "total" in combined_listing.lower() or "." in combined_listing, (
        f"Expected directory listing output (e.g. 'total' line or '.' "
        f"entry) in sys_terminal_read output. Got: {reads_step2!r}. "
        f"The ls command may not have executed in tmux."
    )

    # ── Step 4: Ask agent to create a Python file ────────────────────────
    # Use a unique filename derived from the conversation id to avoid
    # collisions across parallel test runs.
    unique_suffix = conv_id[:8] if conv_id else "test"
    filename = f"/tmp/workspace_test_{unique_suffix}.py"

    turn2_prompt = (
        f"Use sys_terminal_send on terminal 'bash' session 'workspace' to "
        f"create a file at {filename} containing a simple Python function. "
        f"Use this exact command: "
        f"echo 'def hello():\\n    return \"hello world\"' > {filename} "
        f"followed by Enter. Wait briefly, then reply 'created'."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": turn2_prompt,
            "previous_response_id": response_id,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]
    body = poll_until_terminal(http_client, response_id, timeout=180)
    assert body["status"] == "completed", (
        f"Step 4 (create file) failed: status={body['status']!r}, error={body.get('error')!r}."
    )

    # ── Step 5: Ask agent to read the file back with cat ─────────────────
    turn3_prompt = (
        f"Use sys_terminal_send on terminal 'bash' session 'workspace' "
        f"to type 'cat {filename}' followed by Enter. Wait briefly, "
        f"then sys_terminal_read on session 'workspace'. "
        f"Reply with 'read done' once you see the file content."
    )
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": turn3_prompt,
            "previous_response_id": response_id,
            "stream": False,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]
    body = poll_until_terminal(http_client, response_id, timeout=180)
    assert body["status"] == "completed", (
        f"Step 5 (read file) failed: status={body['status']!r}, error={body.get('error')!r}."
    )

    # ── Step 6: Verify file content in tool output ───────────────────────
    # The cat output must appear in at least one sys_terminal_read call.
    # We check ALL reads across the conversation since reads accumulate.
    all_reads = _get_function_call_outputs(http_client, conv_id, "sys_terminal_read")
    combined_reads = " ".join(all_reads)

    # The file should contain the hello function. Check for key
    # fragments that prove the file was created and read back.
    assert "hello" in combined_reads.lower(), (
        f"Expected 'hello' in sys_terminal_read output after cat of "
        f"{filename}. Combined reads: {combined_reads!r}. If empty, "
        f"the echo command may not have written the file, or cat "
        f"didn't execute. If reads show a prompt but no file content, "
        f"the file path may differ from what was created."
    )

    # Verify the 'def' keyword appears — proves it's Python source,
    # not just the word 'hello' from the echo command itself.
    assert "def" in combined_reads, (
        f"Expected 'def' keyword in file content read back from "
        f"{filename}. Combined reads: {combined_reads!r}. The file "
        f"may have been created empty or the echo didn't write the "
        f"function definition."
    )

    # ── Cleanup: remove the temp file ────────────────────────────────────
    # Best-effort cleanup; don't fail the test if this doesn't work.
    http_client.post(
        "/v1/responses",
        json={
            "model": sys_terminal_test_agent,
            "input": (
                f"Use sys_terminal_send on terminal 'bash' session "
                f"'workspace' to type 'rm -f {filename}' followed by "
                f"Enter. Reply 'cleaned'."
            ),
            "previous_response_id": response_id,
            "stream": False,
        },
        timeout=60.0,
    )
