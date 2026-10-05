"""Emulated tool calling: prompt contract in, OpenAI tool_calls out.

Runs without a DeepSeek session — the upstream client is faked.

    pytest tests/test_tool_calling.py
"""

import json

from fastapi.testclient import TestClient

import server.api as api
from server.openai_format import (
    extract_tool_calls,
    messages_to_prompt,
    trim_at_role_boundary,
)
from server.schemas import ChatMessage

TOOLS = [{"type": "function", "function": {
    "name": "browser_navigate",
    "description": "Navigate to a URL",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                   "required": ["url"]}}}]

MCP_TOOLS = [{"type": "function", "function": {
    "name": "mcp__openbrowser__browser_navigate",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}]

TERMINAL_TOOLS = [{"type": "function", "function": {
    "name": "terminal",
    "description": "Run terminal command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def _names(calls):
    return [c["function"]["name"] for c in calls or []]


def _args(calls, i=0):
    return json.loads(calls[i]["function"]["arguments"])


def test_fenced_json_is_a_tool_call():
    calls, left = extract_tool_calls(
        '```json\n{"tool_calls":[{"name":"browser_navigate",'
        '"arguments":{"url":"https://youtube.com"}}]}\n```', TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert _args(calls)["url"] == "https://youtube.com"
    assert left == ""


def test_bare_json_and_prose_wrapper():
    for text in (
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com"}}]}',
        'Sure.\n```json\n{"tool_calls":[{"name":"browser_navigate",'
        '"arguments":{"url":"https://a.com"}}]}\n```',
    ):
        calls, _ = extract_tool_calls(text, TOOLS)
        assert _args(calls)["url"] == "https://a.com"


def test_singular_and_openai_shaped_wrappers():
    calls, _ = extract_tool_calls(
        '{"tool_call":{"name":"browser_navigate","arguments":{"url":"https://c.com"}}}', TOOLS)
    assert _names(calls) == ["browser_navigate"]
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"type":"function","function":{"name":"browser_navigate",'
        '"arguments":"{\\"url\\":\\"https://d.com\\"}"}}]}', TOOLS)
    assert _args(calls)["url"] == "https://d.com"


def test_plain_answer_is_not_a_tool_call():
    calls, left = extract_tool_calls("YouTube is a video site.", TOOLS)
    assert calls is None and left == "YouTube is a video site."


def test_explicit_envelope_passes_unknown_names_through():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"mcp__openbrowser__browser_tabs",'
        '"arguments":{"action":"list"}}]}', TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_tabs"]


def test_react_prose_is_salvaged():
    calls, left = extract_tool_calls(
        "I'll navigate to YouTube.com using OpenBrowser.\n\n"
        'Action: mcp__openbrowser__browser_navigate '
        'Action Input: {"url": "https://www.youtube.com"}', MCP_TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_navigate"]
    assert _args(calls)["url"] == "https://www.youtube.com"
    assert "Action:" not in left


def test_react_prose_resolves_a_dropped_namespace_prefix():
    calls, _ = extract_tool_calls(
        'Action: browser_navigate\nAction Input: {"url": "https://x.com"}', MCP_TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_navigate"]


def test_heuristic_paths_stay_strict_about_names():
    for text in ('Action: made_up_tool Action Input: {"x": 1}',
                 'made_up_tool({"x": 1})',
                 '<made_up_tool x="1">'):
        assert extract_tool_calls(text, TOOLS)[0] is None


def test_react_prose_for_an_unoffered_tool_stays_text():
    calls, _ = extract_tool_calls("Action: browser_get_all_pages Action Input: {}", MCP_TOOLS)
    assert calls is None


def test_prompt_carries_schemas_and_forbids_prose_calls():
    prompt = messages_to_prompt(
        [ChatMessage(role="user", content="go to youtube")], TOOLS)
    assert "browser_navigate" in prompt
    assert "tool_calls" in prompt
    assert "Action:" in prompt


def test_prompt_replays_past_calls_and_results():
    prompt = messages_to_prompt([
        ChatMessage(role="user", content="go to youtube"),
        ChatMessage(role="assistant", tool_calls=[{"id": "call_1", "type": "function",
            "function": {"name": "browser_navigate",
                         "arguments": '{"url":"https://youtube.com"}'}}]),
        ChatMessage(role="tool", tool_call_id="call_1", name="browser_navigate",
                    content="navigated ok"),
    ], TOOLS)
    assert "Tool result (browser_navigate): navigated ok" in prompt
    assert "https://youtube.com" in prompt


# --- native DSML tags (what DeepSeek web emits) ----------------------------

def test_dsml_tool_calls_extracted():
    raw = (
        "I'll check for available tools and grab that trailer.\n\n"
        "<｜｜DSML｜｜ calls>\n"
        '<｜｜DSML｜｜ invoke name="terminal">\n'
        '<｜｜DSML｜｜ parameter name="command" string="true">which yt-dlp</｜｜DSML｜｜ parameter>\n'
        "</｜｜DSML｜｜ invoke>\n"
        "</｜｜DSML｜｜ calls>"
    )
    calls, left = extract_tool_calls(raw, TERMINAL_TOOLS)
    assert _names(calls) == ["terminal"]
    assert _args(calls)["command"] == "which yt-dlp"
    assert left == "I'll check for available tools and grab that trailer."


def test_dsml_parallel_tool_calls():
    raw = (
        "<｜｜DSML｜｜ calls>\n"
        '<｜｜DSML｜｜ invoke name="terminal">\n'
        '<｜｜DSML｜｜ parameter name="command" string="true">ls</｜｜DSML｜｜ parameter>\n'
        "</｜｜DSML｜｜ invoke>\n"
        '<｜｜DSML｜｜ invoke name="terminal">\n'
        '<｜｜DSML｜｜ parameter name="command" string="true">pwd</｜｜DSML｜｜ parameter>\n'
        "</｜｜DSML｜｜ invoke>\n"
        "</｜｜DSML｜｜ calls>"
    )
    calls, left = extract_tool_calls(raw, TERMINAL_TOOLS)
    assert len(calls) == 2
    assert _args(calls, 0)["command"] == "ls"
    assert _args(calls, 1)["command"] == "pwd"
    assert left == ""


# --- endpoint level, with a faked upstream ---------------------------------

class _FakeReply:
    def __init__(self, text):
        self.text, self.conversation_id = text, "conv-1"


class _FakeStream:
    def __init__(self, text):
        self.parts = [text[i:i + 7] for i in range(0, len(text), 7)]
        self.conversation_id = "conv-1"

    def __iter__(self):
        return iter(self.parts)


class _FakeClient:
    canned = ""

    def chat(self, *a, **k):
        return _FakeReply(self.canned)

    def stream(self, *a, **k):
        return _FakeStream(self.canned)


def _client(canned):
    fake = _FakeClient()
    fake.canned = canned
    api.get_client = lambda: fake
    return TestClient(api.app)


def test_endpoint_returns_tool_calls_with_finish_reason():
    c = _client('```json\n{"tool_calls":[{"name":"browser_navigate",'
                '"arguments":{"url":"https://www.youtube.com"}}]}\n```')
    choice = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS,
        "messages": [{"role": "user", "content": "go to youtube"}],
    }).json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert _names(choice["message"]["tool_calls"]) == ["browser_navigate"]


def test_endpoint_final_answer_still_stops_normally():
    c = _client("YouTube loaded fine.")
    choice = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS,
        "messages": [{"role": "user", "content": "what happened"}],
    }).json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "YouTube loaded fine."
    assert "tool_calls" not in choice["message"]


def test_streaming_emits_a_tool_calls_delta():
    c = _client('{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://x.com"}}]}')
    body = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS, "stream": True,
        "messages": [{"role": "user", "content": "go"}],
    }).text
    assert '"tool_calls"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_tool_choice_none_suppresses_tools():
    c = _client("Just answering in text.")
    choice = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS, "tool_choice": "none",
        "messages": [{"role": "user", "content": "hi"}],
    }).json()["choices"][0]
    assert choice["finish_reason"] == "stop"


# --- parser robustness: the shapes that leaked in real use -----------------

def test_braces_inside_a_string_value_do_not_break_balance():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate",'
        '"arguments":{"url":"https://x.com/?q={a}","note":"hi {name}, bye}"}}]}', TOOLS)
    assert _args(calls)["url"] == "https://x.com/?q={a}"
    assert _args(calls)["note"] == "hi {name}, bye}"


def test_escapes_inside_string_values():
    tricky = 'say "hi" \\ then {x}'
    payload = json.dumps({"tool_calls": [
        {"name": "browser_navigate", "arguments": {"url": tricky}}]})
    calls, _ = extract_tool_calls(payload, TOOLS)
    assert _args(calls)["url"] == tricky


def test_unclosed_fence_still_parses():
    calls, _ = extract_tool_calls(
        '```json\n{"tool_calls":[{"name":"browser_navigate",'
        '"arguments":{"url":"https://a.com"}}]}', TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_truncated_reply_is_repaired_structurally():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com","depth":', TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_truncated_value_is_dropped_not_guessed():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com","tabId":15', TOOLS)
    assert "tabId" not in _args(calls)
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com","note":"half wri', TOOLS)
    assert "note" not in _args(calls)


# --- call-syntax dialect ---------------------------------------------------

def test_function_call_syntax_is_parsed():
    calls, left = extract_tool_calls('browser_navigate({"url": "https://a.com"})', TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert _args(calls)["url"] == "https://a.com"
    assert left == ""


def test_call_syntax_with_surrounding_prose():
    calls, _ = extract_tool_calls(
        'Let me check.\nbrowser_navigate({"url": "https://a.com"})\nOne moment.', TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_call_syntax_no_arguments():
    calls, _ = extract_tool_calls("browser_navigate()", TOOLS)
    assert _args(calls) == {}


def test_call_syntax_resolves_dropped_namespace():
    calls, _ = extract_tool_calls('browser_navigate({"url": "https://a.com"})', MCP_TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_navigate"]


def test_call_syntax_ignores_unoffered_tools_and_plain_prose():
    assert extract_tool_calls('delete_everything({"path": "/"})', TOOLS)[0] is None
    assert extract_tool_calls("I read the file (the one you named) and it says hi.", TOOLS)[0] is None


def test_prompt_forbids_announcing_and_puts_contract_last():
    prompt = messages_to_prompt(
        [ChatMessage(role="system", content="You are helpful."),
         ChatMessage(role="user", content="read a file")], TOOLS)
    assert "NEVER announce" in prompt
    assert prompt.index("browser_navigate") > prompt.index("You are helpful.")
    assert prompt.rstrip().endswith("Assistant:")


# --- batches: one bad entry must not sink the valid ones -------------------

def test_batch_keeps_every_well_formed_call():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com"}},'
        '{"name":"browser_get_all_pages","arguments":{}}]}', TOOLS)
    assert _names(calls) == ["browser_navigate", "browser_get_all_pages"]


def test_batch_drops_only_structurally_broken_entries():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com"}},'
        '{"arguments":{"no":"name"}},"not-an-object"]}', TOOLS)
    assert _names(calls) == ["browser_navigate"]


def test_batch_keeps_valid_calls_when_the_tail_is_truncated():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com"}},'
        '{"name":"browser_nav', TOOLS)
    assert _names(calls) == ["browser_navigate"]


def test_batch_with_no_usable_entries_stays_text():
    calls, _ = extract_tool_calls('{"tool_calls":[{"arguments":{}},{"no":"name"}]}', TOOLS)
    assert calls is None


def test_multiple_valid_calls_all_survive():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com"}},'
        '{"name":"browser_navigate","arguments":{"url":"https://b.com"}}]}', TOOLS)
    assert len(calls) == 2


# --- XML dialects ----------------------------------------------------------

def test_xml_attribute_form():
    calls, left = extract_tool_calls('<browser_navigate url="https://a.com">', TOOLS)
    assert _args(calls)["url"] == "https://a.com"
    assert left == ""


def test_xml_self_closing_coerces_scalars():
    calls, _ = extract_tool_calls('<browser_navigate url="https://a.com" depth="3"/>', TOOLS)
    assert _args(calls) == {"url": "https://a.com", "depth": 3}


def test_xml_tool_call_wrapper_around_json():
    calls, _ = extract_tool_calls(
        '<tool_call>{"name":"browser_navigate","arguments":{"url":"https://a.com"}}</tool_call>',
        TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_xml_tool_call_wrapper_with_bare_name():
    calls, _ = extract_tool_calls("<tool_call>browser_navigate</tool_call>", TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert _args(calls) == {}


def test_xml_wrapper_naming_something_that_isnt_a_tool_stays_text():
    calls, _ = extract_tool_calls("<tool_call> some-skill-name", TOOLS)
    assert calls is None


def test_angle_brackets_in_prose_are_not_calls():
    for text in ("Use a < b and c > d in your filter.",
                 'The page had a <div class="x"> element.',
                 "Compare <html> and <body> tags."):
        assert extract_tool_calls(text, TOOLS)[0] is None


# --- nested wrappers and truncated fenced blocks ---------------------------

def test_nested_tool_call_wrapper_is_unwrapped():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"tool_call","arguments":'
        '{"name":"browser_navigate","arguments":{"url":"https://a.com"}}}]}', TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert _args(calls)["url"] == "https://a.com"


def test_deeply_nested_wrappers_are_unwrapped():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"function","arguments":{"name":"tool_call","arguments":'
        '{"name":"browser_navigate","arguments":{"url":"https://a.com"}}}}]}', TOOLS)
    assert _names(calls) == ["browser_navigate"]


def test_wrapper_around_an_unknown_tool_still_unwraps_to_the_inner_name():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"tool_call","arguments":'
        '{"name":"mcp__openbrowser__browser_snapshot","arguments":{}}}]}', TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_snapshot"]


def test_wrapper_with_no_inner_name_is_not_a_call():
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"tool_call","arguments":{"foo":"bar"}}]}', TOOLS)
    assert calls is None


def test_truncated_fenced_block_is_repaired():
    calls, _ = extract_tool_calls(
        '```json\n{"tool_calls": [{"name": "browser_navigate", '
        '"arguments": {"url": "https://a.com"}}\n```', TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_truncated_fenced_nested_wrapper_the_real_world_case():
    calls, _ = extract_tool_calls(
        '```json\n{"tool_calls": [{"name": "tool_call", "arguments": '
        '{"name": "browser_navigate", "arguments": {"url": "https://mail.google.com"}}}\n```',
        TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert _args(calls)["url"] == "https://mail.google.com"


def test_fenced_non_call_json_stays_text():
    assert extract_tool_calls('```json\n{"note": "nothing here"}\n```', TOOLS)[0] is None


# --- the model running past its own turn -----------------------------------

def test_fabricated_continuation_is_trimmed():
    calls, left = extract_tool_calls(
        '{"tool_calls": [{"name": "browser_navigate", "arguments": {"url": "https://a.com"}}]}\n'
        "\nTool result (browser_navigate): opened tab 123\n"
        "\nAssistant: I opened it and the page loaded.", TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert "Tool result" not in (left or "")
    assert "opened tab 123" not in (left or "")


def test_trim_leaves_ordinary_prose_alone():
    for text in ("The inbox shows 32 unread messages.",
                 "Here is the result: 5 tabs open.",
                 "Assistant: is a word that can start a sentence."):
        assert trim_at_role_boundary(text) == text


# --- malformed envelopes ---------------------------------------------------

def test_calls_survive_an_envelope_whose_array_never_closes():
    calls, _ = extract_tool_calls(
        '{"tool_calls": [{"name": "browser_navigate", "arguments": {"url": "https://a.com"}}}', TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_multiple_calls_survive_a_broken_envelope():
    calls, _ = extract_tool_calls(
        '{"tool_calls": [{"name": "browser_navigate", "arguments": {"url": "https://a.com"}},'
        '{"name": "browser_navigate", "arguments": {"url": "https://b.com"}}}', TOOLS)
    assert len(calls) == 2


def test_json_quoted_in_prose_is_not_a_call():
    calls, _ = extract_tool_calls(
        'The config was {"name": "widget", "size": 3} which I updated.', TOOLS)
    assert calls is None


def test_bare_object_with_arguments_is_a_call():
    calls, _ = extract_tool_calls(
        '{"name": "browser_navigate", "arguments": {"url": "https://a.com"}}', TOOLS)
    assert _names(calls) == ["browser_navigate"]


# --- upstream errors must not look like an empty reply ---------------------

def test_stream_error_frame_raises_instead_of_yielding_nothing():
    from deepseek.client import _parse_sse
    from deepseek import UpstreamError
    frames = ['data: {"type":"error","content":"Messages too frequent. Try again '
              'later.","finish_reason":"rate_limit_reached"}']
    try:
        list(_parse_sse(iter(frames)))
    except UpstreamError as exc:
        assert exc.is_rate_limit
        assert "too frequent" in str(exc)
    else:
        raise AssertionError("rate-limit frame was swallowed")


def test_normal_stream_frames_still_parse():
    from deepseek.client import _parse_sse
    frames = ['data: {"p":"response/fragments/-1/content","o":"APPEND","v":"hi"}',
              'data: {"v":" there"}']
    assert list(_parse_sse(iter(frames))) == ["hi", " there"]


def _erroring_client(exc):
    class _C:
        def chat(self, *a, **k):
            raise exc

        def stream(self, *a, **k):
            def g():
                raise exc
                yield
            return g()
    api.get_client = lambda: _C()
    return TestClient(api.app, raise_server_exceptions=False)


def test_rate_limit_becomes_429_with_retry_after():
    from deepseek import UpstreamError
    c = _erroring_client(UpstreamError("Messages too frequent.", "rate_limit_reached"))
    for stream in (False, True):
        r = c.post("/v1/chat/completions", json={
            "model": "deepseek-chat", "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 429, (stream, r.status_code)
        assert r.headers.get("Retry-After")
        assert r.json()["error"]["type"] == "rate_limit_exceeded"


def test_other_upstream_errors_become_502():
    from deepseek import UpstreamError
    c = _erroring_client(UpstreamError("Something broke upstream.", "server_error"))
    r = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502


def test_prefetched_stream_does_not_duplicate_the_first_delta():
    class _S:
        conversation_id = "conv-1"

        def __init__(self):
            self.parts = ["I", "'ll", " go"]

        def __iter__(self):
            return iter(self.parts)

    raw = _S()
    it = iter(raw)
    first = next(it, None)
    assert "".join(api._Prefetched(raw, it, first)) == "I'll go"


# --- Python dict repr ------------------------------------------------------

def test_python_dict_repr_is_parsed():
    calls, _ = extract_tool_calls(
        "{'tool_calls': [{'name': 'browser_navigate', "
        "'arguments': {'url': 'https://a.com'}}]}", TOOLS)
    assert _args(calls)["url"] == "https://a.com"


def test_python_literals_are_coerced():
    calls, _ = extract_tool_calls(
        "{'tool_calls': [{'name': 'browser_navigate', "
        "'arguments': {'url': 'https://a.com', 'full': True, 'sel': None}}]}", TOOLS)
    assert _args(calls) == {"url": "https://a.com", "full": True, "sel": None}


def test_python_repr_in_prose_is_not_a_call():
    assert extract_tool_calls("It printed {'a': 1} to the console.", TOOLS)[0] is None
