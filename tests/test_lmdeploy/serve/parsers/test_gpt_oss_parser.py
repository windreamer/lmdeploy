from dataclasses import dataclass

import pytest

pytest.importorskip('openai_harmony')

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import _openai_harmony as openai_harmony_mod
from lmdeploy.serve.parsers import gpt_oss_response_parser as gpt_oss_mod

from .helpers import first_stream_delta


@dataclass
class _FakeMsg:
    channel: str
    recipient: str | None


class _FakeStreamableParser:
    """Scripted stand-in for openai_harmony.StreamableParser."""

    def __init__(self, script: dict[int, dict]):
        self._script = script
        self.current_channel = 'final'
        self.current_recipient = None
        self.last_content_delta = ''
        self.messages: list[_FakeMsg] = []

    def process(self, token: int):
        event = self._script[token]
        next_channel = event['channel']
        next_recipient = event.get('recipient')

        if (self.current_channel == 'commentary' and self.current_recipient
                and self.current_recipient.startswith('functions.') and next_recipient != self.current_recipient):
            self.messages.append(_FakeMsg(channel='commentary', recipient=self.current_recipient))

        self.current_channel = next_channel
        self.current_recipient = next_recipient
        self.last_content_delta = event.get('delta', '')


def _scripted_events() -> dict[int, dict]:
    return {
        1: {
            'channel': 'analysis',
            'recipient': None,
            'delta': 'Need tool. ',
        },
        2: {
            'channel': 'commentary',
            'recipient': 'functions.get_weather',
            'delta': '',
        },
        3: {
            'channel': 'commentary',
            'recipient': 'functions.get_weather',
            'delta': '{"location":"',
        },
        4: {
            'channel': 'commentary',
            'recipient': 'functions.get_weather',
            'delta': 'Beijing"}',
        },
        5: {
            'channel': 'commentary',
            'recipient': 'functions.get_time',
            'delta': '',
        },
        6: {
            'channel': 'commentary',
            'recipient': 'functions.get_time<|channel|>commentary',
            'delta': '{"tz":"UTC"}',
        },
        7: {
            'channel': 'final',
            'recipient': None,
            'delta': 'Result: ',
        },
        8: {
            'channel': 'final',
            'recipient': None,
            'delta': 'sunny',
        },
    }

class TestGptOssResponseParser:
    """Unit tests for :class:`GptOssResponseParser` (Harmony token
    streaming)."""

    @pytest.fixture(autouse=True)
    def _mock_get_encoding(self, monkeypatch):
        """Prevent ``get_encoding`` from loading the real Harmony vocab, which
        is unavailable in some test environments."""
        monkeypatch.setattr(openai_harmony_mod, 'get_encoding', lambda: None)
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )

    def test_stream_chunk_full_sequence(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(_scripted_events()),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk(delta_text='ignored',
                                                                     delta_token_ids=[1, 2, 3, 4, 5, 6, 7, 8]))
        assert delta is not None
        assert delta.content == 'Result: sunny'
        assert delta.reasoning_content == 'Need tool. '
        assert parser.reasoning_tokens == 1
        assert tool_emitted is True
        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 5

        # name delta + args delta for get_weather
        assert delta.tool_calls[0].function is not None
        assert delta.tool_calls[0].function.name == 'get_weather'
        assert delta.tool_calls[1].function is not None
        assert delta.tool_calls[1].function.arguments == '{"location":"'
        assert delta.tool_calls[2].function is not None
        assert delta.tool_calls[2].function.arguments == 'Beijing"}'

        # second tool: name delta + sanitized malformed recipient arguments delta.
        assert delta.tool_calls[3].function is not None
        assert delta.tool_calls[3].function.name == 'get_time'
        assert delta.tool_calls[4].function is not None
        assert delta.tool_calls[4].function.arguments == '{"tz":"UTC"}'

    def test_adjust_request_converts_tools_to_wrapper_dicts(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[],
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'city': {
                                    'type': 'string'
                                }
                            }
                        },
                    },
                },
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_time',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'tz': {
                                    'type': 'string'
                                }
                            }
                        },
                    },
                },
            ],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'get_time'
                },
            },
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        assert parser.request.tools == [{
            'type': 'function',
            'function': {
                'name': 'get_time',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'tz': {
                            'type': 'string'
                        }
                    },
                },
                'description': None,
            },
            }]

    def test_parse_complete_full_sequence(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(_scripted_events()),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        content, tool_calls, reasoning = parser.parse_complete(text='', token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
        assert content == 'Result: sunny'
        assert reasoning == 'Need tool. '
        assert parser.reasoning_tokens == 1
        assert tool_calls is not None
        assert [call.function.name for call in tool_calls] == ['get_weather', 'get_time']
        assert [call.function.arguments for call in tool_calls] == ['{"location":"Beijing"}', '{"tz":"UTC"}']

    def test_stream_chunk_bootstrap_empty_before_any_content(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', []))
        assert delta is not None
        assert delta.role == 'assistant'
        assert delta.content == ''
        assert tool_emitted is False

    def test_stream_chunk_empty_after_content_started_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        parser.stream_chunk('warmup', [])
        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', []))
        assert delta is None
        assert tool_emitted is False

    def test_stream_chunk_text_only_without_token_ids(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk('plain text', []))
        assert delta is not None
        assert delta.content == 'plain text'
        assert delta.reasoning_content is None
        assert delta.tool_calls is None
        assert tool_emitted is False

    def test_stream_chunk_token_ids_all_empty_delta_returns_none(self, monkeypatch):
        script = {
            10: {'channel': 'final', 'recipient': None, 'delta': ''},
        }
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(script),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', [10]))
        assert delta is None
        assert tool_emitted is False

    def test_stream_chunk_analysis_without_tool_accumulates_reasoning(self, monkeypatch):
        script = {
            1: {'channel': 'analysis', 'recipient': None, 'delta': 'think '},
            2: {'channel': 'analysis', 'recipient': None, 'delta': 'more'},
        }
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(script),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', [1, 2]))
        assert delta is not None
        assert delta.content is None
        assert delta.reasoning_content == 'think more'
        assert parser.reasoning_tokens == 2
        assert delta.tool_calls is None
        assert tool_emitted is False

    def test_parse_complete_without_token_ids_returns_raw_text(self):
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        content, tool_calls, reasoning = parser.parse_complete('hello', token_ids=[])
        assert content == 'hello'
        assert tool_calls is None
        assert reasoning is None

    def test_parse_complete_without_token_ids_empty_text(self):
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        content, tool_calls, reasoning = parser.parse_complete('', token_ids=None)
        assert content is None
        assert tool_calls is None
        assert reasoning is None

    def test_parse_complete_appends_tool_call_still_open_at_eof(self, monkeypatch):
        """Final `active` tool dict is appended when the stream ends in a tool
        channel."""
        script = {
            1: {
                'channel': 'commentary',
                'recipient': 'functions.echo',
                'delta': '{"x":1}',
            },
        }
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(script),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        content, tool_calls, reasoning = parser.parse_complete(text='', token_ids=[1])
        assert content is None
        assert reasoning is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == 'echo'
        assert tool_calls[0].function.arguments == '{"x":1}'

    @pytest.mark.parametrize(
        ('recipient', 'expected'),
        [
            (None, None),
            ('', None),
            ('not-a-tool', None),
            ('functions.', None),
            ('functions.foo', 'foo'),
            ('prefix functions.bar suffix', 'bar'),
            ('functions.bash<|channel|>commentary', 'bash'),
            ('functions.tool_name<|extra|', 'tool_name'),
        ],
    )
    def test_extract_tool_name(self, recipient, expected):
        assert gpt_oss_mod.GptOssResponseParser._extract_tool_name(recipient) == expected


class TestGptOssResponseFormatGrammarConversion:
    """Tests for GptOssResponseParser response_format → structural_tag
    conversion (replaces the old Harmony-native prompt injection)."""

    @pytest.fixture(autouse=True)
    def _patch_streamable_parser(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        monkeypatch.setattr(openai_harmony_mod, 'get_encoding', lambda: None)

    def test_json_schema_converted_to_structural_tag(self):
        """json_schema response_format is converted to a structural_tag, not
        cleared."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'x': {'type': 'integer'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None
        # The structural_tag JSON must contain the original schema
        st_json = _json.dumps(rf['structural_tag'])
        assert _json.dumps(schema_dict) in st_json
        # Messages must NOT be modified (no prompt injection)
        assert len(parser.request.messages) == 1
        assert parser.request.messages[0]['role'] == 'user'

    def test_regex_schema_converted_to_structural_tag(self):
        from lmdeploy.serve.openai.protocol import ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(type='regex_schema', regex_schema='[0-9]+'),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None

    def test_json_object_converted_to_structural_tag(self):
        from lmdeploy.serve.openai.protocol import ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(type='json_object'),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None

    def test_text_response_format_is_cleared(self):
        """Text response_format is cleared (no grammar needed)."""
        from lmdeploy.serve.openai.protocol import ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(type='text'),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)
        assert parser.request.response_format is None

    def test_no_response_format_leaves_request_unchanged(self):
        """When response_format is None the request is not modified."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)
        assert parser.request.response_format is None
        assert len(parser.request.messages) == 1

    def test_structural_tag_preserves_final_channel_begin(self):
        """The structural_tag must contain the Harmony final channel begin
        string so the grammar wraps the schema correctly."""
        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(
                    name='test',
                    schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
                ),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        import json as _json
        st_json = _json.dumps(parser.request.response_format['structural_tag'])
        assert '<|channel|>final<|message|>' in st_json
        assert '<|end|>' in st_json
        assert '<|channel|>analysis<|message|>' in st_json

    def test_non_pydantic_request_gets_structural_tag(self):
        """Non-Pydantic sentinel requests also get response_format
        converted."""
        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'y': {'type': 'number'}}}
        fmt = ResponseFormat(
            type='json_schema',
            json_schema=JsonSchema(name='test', schema=schema_dict),
        )

        class _Sentinel:
            messages = [{'role': 'user', 'content': 'hi'}]
            response_format = fmt

        sentinel = _Sentinel()
        parser = gpt_oss_mod.GptOssResponseParser(request=sentinel)

        assert parser.request.response_format is not None
        assert parser.request.response_format.type == 'structural_tag'

    def test_grammar_failure_falls_back_to_prompt_injection(self, monkeypatch):
        """When xgrammar is unavailable, response_format is injected into the
        system prompt and cleared (legacy Harmony-native fallback)."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        # Simulate grammar construction failure
        monkeypatch.setattr(
            gpt_oss_mod.GptOssResponseParser,
            '_build_response_format_grammar',
            staticmethod(lambda fmt: None),
        )

        schema_dict = {'type': 'object', 'properties': {'x': {'type': 'integer'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        # response_format must be cleared
        assert parser.request.response_format is None
        # A system message with the schema must have been inserted
        msgs = parser.request.messages
        assert msgs[0]['role'] == 'system'
        assert '# Response Formats' in msgs[0]['content']
        assert _json.dumps(schema_dict) in msgs[0]['content']


class TestGptOssToolGrammarInjection:
    """Tests for GptOssResponseParser tool-calling structural_tag injection."""

    @pytest.fixture(autouse=True)
    def _patch_streamable_parser(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        monkeypatch.setattr(openai_harmony_mod, 'get_encoding', lambda: None)

    def test_required_tool_choice_injects_structural_tag(self):
        """tool_choice=required injects a structural_tag with tool call
        grammar."""
        import json as _json

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'What is the weather?'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {
                        'type': 'object',
                        'properties': {'location': {'type': 'string'}},
                    },
                },
            }],
            tool_choice='required',
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None
        st_json = _json.dumps(rf['structural_tag'])
        # Must contain Harmony tool call begin strings
        assert 'functions.get_weather' in st_json
        assert '<|call|>' in st_json
        assert '<|constrain|>json' in st_json

    def test_auto_tool_choice_injects_structural_tag(self):
        """tool_choice=auto also injects a structural_tag (with final channel
        fallback)."""
        import json as _json

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'What is the weather?'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {
                        'type': 'object',
                        'properties': {'location': {'type': 'string'}},
                    },
                },
            }],
            tool_choice='auto',
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        st_json = _json.dumps(rf['structural_tag'])
        assert 'functions.get_weather' in st_json
        # auto mode should also allow final channel content
        assert '<|channel|>final' in st_json

    def test_specific_tool_choice_injects_structural_tag(self):
        """tool_choice={"type":"function","function":{"name":"X"}} injects
        grammar for only that function."""
        import json as _json

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'What is the weather?'}],
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'parameters': {'type': 'object', 'properties': {}},
                    },
                },
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_time',
                        'parameters': {'type': 'object', 'properties': {}},
                    },
                },
            ],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_weather'},
            },
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        st_json = _json.dumps(rf['structural_tag'])
        assert 'functions.get_weather' in st_json
        # The non-selected tool should NOT appear
        assert 'functions.get_time' not in st_json

    def test_none_tool_choice_does_not_inject_grammar(self):
        """tool_choice=none does not inject tool grammar."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            }],
            tool_choice='none',
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        # No grammar should be injected for tool_choice=none
        rf = parser.request.response_format
        assert rf is None or rf['type'] != 'structural_tag'

    def test_tools_priority_over_response_format(self):
        """When both tools and response_format are present, tool grammar takes
        priority."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            }],
            tool_choice='required',
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(
                    name='test',
                    schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
                ),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        st_json = _json.dumps(rf['structural_tag'])
        # Tool grammar wins — must contain tool call, not plain json_schema
        assert 'functions.get_weather' in st_json
