from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.background_agent import AGENT_SYSTEM_PROMPT, TASK_CHECKPOINT_TOOL
from harness.call import LIVE_CALL_SYSTEM_PROMPT
from portal.app import TOOL_RESULT_POLICY, create_app
from portal.tools import SAFE_TOOLS, tool_use_instructions
from qwen_omni_adapters.context import (
    CONTEXT_SCHEMA,
    ContextConfigError,
    context_catalog,
    load_context,
)
from runtime.adapter_server import (
    DEFAULT_LANGUAGE_SYSTEM_PROMPT,
    MEDIA_CHAT_SYSTEM_PROMPT,
)


def test_context_catalog_is_the_runtime_source_of_prompts_and_tools() -> None:
    catalog = context_catalog()

    assert catalog["schema"] == CONTEXT_SCHEMA
    assert catalog["prompts"]["live_call_system"] == LIVE_CALL_SYSTEM_PROMPT
    assert catalog["prompts"]["background_agent_system"] == AGENT_SYSTEM_PROMPT
    assert catalog["prompts"]["media_encoder_system"] == MEDIA_CHAT_SYSTEM_PROMPT
    assert catalog["prompts"][
        "default_language_system"
    ] == DEFAULT_LANGUAGE_SYSTEM_PROMPT
    assert catalog["directives"]["tool_result_policy"] == TOOL_RESULT_POLICY
    assert tool_use_instructions() == catalog["directives"]["tool_use"]
    assert catalog["control_tools"]["task_checkpoint"] == TASK_CHECKPOINT_TOOL
    assert [item["schema"] for item in catalog["tools"]] == SAFE_TOOLS


def test_foreground_gateway_is_described_as_execution_capability() -> None:
    catalog = context_catalog()
    live = catalog["prompts"]["live_call_system"]
    execution = catalog["directives"]["foreground_execution"]
    background = next(
        item["schema"]["function"]
        for item in catalog["tools"]
        if item["schema"]["function"]["name"] == "background_task"
    )

    assert "undiscovered, not unavailable" in live
    assert "Report a capability blocker only after a relevant tool attempt" in live
    assert "full allowed tool catalog" in execution
    assert "execution gateway" in background["description"]


def test_live_context_forbids_support_boilerplate_and_unsolicited_transport_meta() -> None:
    catalog = context_catalog()
    default = catalog["prompts"]["default_language_system"]
    live = catalog["prompts"]["live_call_system"]
    portal = catalog["prompts"]["portal_behavior_system"]

    assert "natural participant" in default
    assert "never as a support agent" in live
    assert "Never announce your role, availability, or readiness" in live
    assert "I'm here to help" in live
    assert "How can I assist?" in live
    assert "unless the speaker explicitly asks you to discuss" in live
    assert "example, or recommendation, give one" in live
    assert "never use headings, bullets, or a menu" in live
    assert "ordinary decimal digits" in live
    assert "Do not mention message delivery, the microphone" in live
    assert "unless the speaker explicitly asks about it" in live
    assert "not as a support agent" in portal


def test_mutable_or_explicitly_verified_facts_require_fresh_tool_evidence() -> None:
    tool_policy = context_catalog()["directives"]["tool_use"]

    assert "fact that can change after training" in tool_policy
    assert "explicitly asks you to check or verify" in tool_policy
    assert "until a fresh relevant tool result confirms it" in tool_policy


def test_browser_receives_prompts_from_the_same_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from portal.app import PortalConfig

    monkeypatch.setenv("OMNI_PORTAL_TOKEN", "context-test-token-with-32-characters")
    config = PortalConfig.from_environment()
    app = create_app(config)
    response = app.test_client().get("/")
    html = response.get_data(as_text=True)
    start = html.index('<script id="omni-context" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    browser_context = json.loads(html[start:end])

    assert browser_context == {
        "live_call_system": context_catalog()["prompts"]["live_call_system"],
        "media_conversation_system": context_catalog()["prompts"][
            "media_conversation_system"
        ],
    }


def test_context_override_is_validated(tmp_path: Path) -> None:
    invalid = tmp_path / "context.json"
    invalid.write_text('{"schema":"wrong"}', encoding="utf-8")

    with pytest.raises(ContextConfigError, match=CONTEXT_SCHEMA):
        load_context(invalid)
