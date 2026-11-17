<<<<<<< Updated upstream
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Literal

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Parse Qwen function and parameter tags with the XML state machine.

    ``<function=...>`` resolves ``function``; ``<parameter=...>`` and
    ``</function>`` are handled by ``arg_start``; the remainder of the
    parameter opener resolves ``arg_name``.
    """

    structural_tag_model = 'qwen_3_coder'
    reasoning_structural_tag_model = 'qwen_3_5'
    strip_value_newlines = True

    func_prefix = '<function='
    func_suffix = '</function>'
    param_prefix = '<parameter='
    param_suffix = '</parameter>'
    arg_value_close_tag = param_suffix
    _param_close_start = '</'
    _param_close_without_end = '</parameter'
    arg_value_close_prefixes = (_param_close_without_end, _param_close_start)

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '</tool_call>'

    def _consume_function(
        self,
        payload: str,
        pos: int,
        final: bool,
    ) -> tuple[int, str, Literal['arg_start', 'done']] | None:
        """Consume a complete ``<function=name>`` opener from ``function``."""
        start = payload.find(self.func_prefix, pos)
        if start < 0:
            return None
        name_start = start + len(self.func_prefix)
        name_end = payload.find('>', name_start)
        if name_end < 0:
            return None

        return name_end + 1, payload[name_start:name_end].strip(), 'arg_start'

    def _consume_arg_start(self, payload: str, pos: int) -> tuple[int, Literal['arg_name', 'done']] | None:
        """Enter ``arg_name`` or finish at the inner function close tag."""
        param_start = payload.find(self.param_prefix, pos)
        func_end = payload.find(self.func_suffix, pos)
        if func_end >= 0 and (param_start < 0 or func_end < param_start):
            return func_end + len(self.func_suffix), 'done'
        if param_start < 0:
            return None

        return param_start + len(self.param_prefix), 'arg_name'

    def _consume_arg_name(self, payload: str, pos: int) -> tuple[int, str] | None:
        """Read the parameter name and return the raw-value start."""
        name_end = payload.find('>', pos)
        if name_end < 0:
            return None

        return name_end + 1, payload[pos:name_end].strip()

    def _stable_arg_value_end(self, payload: str, start: int) -> int:
        """Dispatch Qwen's two partial markers by their final character."""
        end = len(payload)
        if end <= start:
            return end

        last = payload[-1]
        if last == '/' and payload.endswith(self._param_close_start, start):
            # '/' means the last character of "</"
            return end - len(self._param_close_start)
        if last == 'r' and payload.endswith(self._param_close_without_end, start):
            # 'r' means the last character of "</parameter"
            return end - len(self._param_close_without_end)
        return end
||||||| Stash base
=======
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest


def _parse_tool_call_arguments_dict(arguments: Any) -> dict[str, Any] | None:
    """Return dict-like tool arguments for Qwen3Coder request normalization."""
    if not isinstance(arguments, str):
        return None

    try:
        parsed_arguments = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed_arguments, dict):
        return parsed_arguments
    return None


@ToolParserManager.register_module(['qwen3coder'])
class Qwen3CoderToolParser(XmlToolParser):
    """Tool parser for Qwen3Coder XML tool-call payloads."""

    tool_start_token = '<tool_call>'
    tool_end_token = '</tool_call>'
    func_prefix = '<function='
    func_end_token = '</function>'
    param_prefix = '<parameter='
    param_end_token = '</parameter>'

    def _normalize_request_messages(self, messages: list[dict]) -> list[dict] | None:
        """Return a render-safe copy of request messages when needed."""
        normalized_messages = None

        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            tool_calls = message.get('tool_calls')
            if not isinstance(tool_calls, list):
                continue

            normalized_tool_calls = None
            for tool_idx, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get('function')
                if not isinstance(function, dict) or isinstance(function.get('arguments'), dict):
                    continue

                parsed_arguments = _parse_tool_call_arguments_dict(function.get('arguments'))
                if parsed_arguments is None:
                    continue

                if normalized_messages is None:
                    normalized_messages = list(messages)
                if normalized_tool_calls is None:
                    normalized_tool_calls = list(tool_calls)
                    normalized_message = dict(message)
                    normalized_message['tool_calls'] = normalized_tool_calls
                    normalized_messages[msg_idx] = normalized_message

                normalized_function = dict(function)
                normalized_function['arguments'] = parsed_arguments

                normalized_tool_call = dict(tool_call)
                normalized_tool_call['function'] = normalized_function
                normalized_tool_calls[tool_idx] = normalized_tool_call

        return normalized_messages

    # Qwen3Coder closes tool argument JSON only when the model emits the
    # explicit function end marker (</function>). We intentionally avoid
    # auto-closing on stream final to prevent producing a syntactically
    # complete but semantically incomplete arguments object.
    def _close_json_on_final(self) -> bool:
        return False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        messages = request.messages
        if not isinstance(messages, list):
            return super().adjust_request(request)

        normalized_messages = self._normalize_request_messages(messages)
        if normalized_messages is None:
            return super().adjust_request(request)
        return super().adjust_request(request.model_copy(update={'messages': normalized_messages}))

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'xml'

    def _extract_incremental_state(self, payload: str, final: bool = False) -> tuple[str | None, dict[str, Any], bool]:
        return self._extract_params(payload)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict, _ = self._extract_params(payload)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        args_json = json.dumps(args_dict, ensure_ascii=False) if args_dict else '{}'
        return ToolCall(function=FunctionCall(name=func_name, arguments=args_json))

    def _extract_params(self, content: str) -> tuple[str | None, dict[str, Any], bool]:
        """Extract function name, parameter map, and close status from XML."""
        content = content.replace(self.tool_start_token, '').replace(self.tool_end_token, '').strip()

        func_name = None
        func_start = content.find(self.func_prefix)
        if func_start != -1:
            name_start = func_start + len(self.func_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if terminators:
                func_name = content[name_start:min(terminators)].strip()

        args_dict = {}
        search_idx = 0
        while True:
            param_start = content.find(self.param_prefix, search_idx)
            if param_start == -1:
                break

            name_start = param_start + len(self.param_prefix)
            terminators = [idx for idx in (content.find('>', name_start), content.find('\n', name_start)) if idx != -1]
            if not terminators:
                break

            name_end = min(terminators)
            param_name = content[name_start:name_end].strip()

            val_start = name_end + 1
            val_end = content.find(self.param_end_token, val_start)
            if val_end == -1:
                break

            param_val_str = content[val_start:val_end].strip()

            # Qwen3Coder XML payloads do not carry explicit type metadata.
            # Keep parameter values as strings to avoid implicit type coercion
            # (e.g., zip codes like 77004 being parsed into integers).
            try:
                parsed_val = json.loads(param_val_str)
                val = parsed_val if isinstance(parsed_val, str) else param_val_str
            except json.JSONDecodeError:
                val = param_val_str
            args_dict[param_name] = val
            search_idx = val_end + len(self.param_end_token)

        is_func_closed = self.func_end_token in content
        return func_name, args_dict, is_func_closed
>>>>>>> Stashed changes
