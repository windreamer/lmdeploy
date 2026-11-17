<<<<<<< Updated upstream
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import Tool


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(XmlToolParser):
    """Parse GLM-4.7 mixed text and tags with the XML state machine.

    Leading plain text before the first ``<arg_key>`` or outer close resolves
    ``function``. After arguments begin, ``arg_start`` distinguishes the next
    key from ``</tool_call>``; the key close and value opener are consumed while
    resolving ``arg_name``.
    """

    token_id_terminals = (
        '<tool_call>', '</tool_call>', '<arg_key>', '</arg_key>', '<arg_value>', '</arg_value>',
    )
    structural_tag_model = 'glm_4_7'
    arg_key_open_tag = '<arg_key>'
    arg_key_close_tag = '</arg_key>'
    arg_value_open_tag = '<arg_value>'
    arg_value_close_tag = '</arg_value>'
    # The complete closing marker is one token in supported GLM tokenizers.
    arg_value_close_prefixes = ()

    @classmethod
    def build_auto_response_format(
        cls, tools: list[Tool], *, tokenizer: PreTrainedTokenizerBase | None,
    ) -> dict[str, Any] | None:
        """Constrain declared argument names while keeping tool calls
        optional."""
        from .glm47_tool_constraints import build_auto_tool_format

        return build_auto_tool_format(tools, tokenizer)

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
        """Resolve the leading plain-text function name once."""
        arg_key_start = self._find_structure(payload, self.arg_key_open_tag, pos)
        block_end = self._find_structure(payload, self.get_tool_close_tag(), pos)
        if block_end >= 0 and (arg_key_start < 0 or block_end < arg_key_start):
            return block_end, payload[pos:block_end].strip(), 'done'
        if arg_key_start >= 0:
            return arg_key_start, payload[pos:arg_key_start].strip(), 'arg_start'

        if final and pos < len(payload):
            return len(payload), payload[pos:].strip(), 'done'
        return None

    def _consume_arg_start(self, payload: str, pos: int) -> tuple[int, Literal['arg_name', 'done']] | None:
        """Enter ``arg_name`` or stop before the outer closing tag."""
        arg_key_start = self._find_structure(payload, self.arg_key_open_tag, pos)
        block_end = self._find_structure(payload, self.get_tool_close_tag(), pos)
        if block_end >= 0 and (arg_key_start < 0 or block_end < arg_key_start):
            return block_end, 'done'
        if arg_key_start < 0:
            return None
        return arg_key_start + len(self.arg_key_open_tag), 'arg_name'

    def _consume_arg_name(self, payload: str, pos: int) -> tuple[int, str] | None:
        """Read the argument key and return the raw-value start."""
        key_end = self._find_structure(payload, self.arg_key_close_tag, pos)
        if key_end < 0:
            return None
        value_start = self._find_structure(payload, self.arg_value_open_tag, key_end + len(self.arg_key_close_tag))
        if value_start < 0:
            return None

        return value_start + len(self.arg_value_open_tag), payload[pos:key_end].strip()
||||||| Stash base
=======
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json

from lmdeploy.serve.openai.protocol import (
    FunctionCall,
    ToolCall,
)

from .tool_parser import ToolParserManager
from .xml_tool_parser import XmlToolParser  # type: ignore[reportMissingImports]


@ToolParserManager.register_module(['glm47'])
class Glm47ToolParser(XmlToolParser):
    """Tool parser for GLM-4.7 XML-like tool-call payloads.

    Expected format inside ``<tool_call>...</tool_call>``:
    ``function_name<arg_key>k</arg_key><arg_value>v</arg_value>...``
    """

    tool_start_token = '<tool_call>'
    tool_end_token = '</tool_call>'
    arg_key_start_token = '<arg_key>'
    arg_key_end_token = '</arg_key>'
    arg_value_start_token = '<arg_value>'
    arg_value_end_token = '</arg_value>'

    def get_tool_open_tag(self) -> str | None:
        return self.tool_start_token

    def get_tool_close_tag(self) -> str | None:
        return self.tool_end_token

    def get_tool_payload_format(self) -> str:
        return 'xml'

    def _extract_incremental_state(self, payload: str, final: bool = False) -> tuple[str | None, dict[str, str], bool]:
        func_name, args_dict = self._parse_payload(payload, final=final)
        return func_name, args_dict, False

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        func_name, raw_args_dict = self._parse_payload(payload, final=True)
        if not func_name:
            return None
        args_dict = self._coerce_args_by_schema(func_name, raw_args_dict)
        return ToolCall(function=FunctionCall(name=func_name, arguments=json.dumps(args_dict, ensure_ascii=False)))

    def _parse_payload(self, payload: str, *, final: bool = False) -> tuple[str | None, dict[str, str]]:
        payload = payload.strip()
        if not payload:
            return None, {}

        args_start_idx = payload.find(self.arg_key_start_token)
        if args_start_idx >= 0:
            func_name = payload[:args_start_idx].strip()
            args_text = payload[args_start_idx:]
        else:
            # Do not treat a growing prefix as the callee until ``<arg_key>`` or
            # end-of-payload (``final``); the latter covers zero-argument tools.
            if not final:
                return None, {}
            func_name = payload.strip()
            args_text = ''
        if not func_name:
            return None, {}

        args_dict: dict[str, str] = {}
        search_idx = 0
        while True:
            key_start = args_text.find(self.arg_key_start_token, search_idx)
            if key_start < 0:
                break
            key_content_start = key_start + len(self.arg_key_start_token)
            key_end = args_text.find(self.arg_key_end_token, key_content_start)
            if key_end < 0:
                break
            key = args_text[key_content_start:key_end].strip()
            value_start = args_text.find(self.arg_value_start_token, key_end + len(self.arg_key_end_token))
            if value_start < 0:
                break
            value_content_start = value_start + len(self.arg_value_start_token)
            value_end = args_text.find(self.arg_value_end_token, value_content_start)
            if value_end < 0:
                break
            if key:
                args_dict[key] = args_text[value_content_start:value_end]
            search_idx = value_end + len(self.arg_value_end_token)
        return func_name, args_dict
>>>>>>> Stashed changes
