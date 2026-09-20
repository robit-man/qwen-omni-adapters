from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.background_agent import AGENT_SYSTEM_PROMPT, TASK_CHECKPOINT_TOOL
from harness.call import LIVE_CALL_SYSTEM_PROMPT
from portal.app import LIVE_RESPONSE_TOOL, TOOL_RESULT_POLICY, create_app
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
    assert catalog["control_tools"]["respond_to_user"] == LIVE_RESPONSE_TOOL
    assert catalog["control_tools"]["task_checkpoint"] == TASK_CHECKPOINT_TOOL
    assert [item["schema"] for item in catalog["tools"]] == SAFE_TOOLS


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
