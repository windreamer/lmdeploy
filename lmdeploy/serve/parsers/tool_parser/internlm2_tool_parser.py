<<<<<<< Updated upstream
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager

if TYPE_CHECKING:
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Tool


@ToolParserManager.register_module(['internlm', 'intern-s1'])
class Internlm2ToolParser(JsonToolParser):
    """Tool parser for InternLM JSON tool-call payloads."""

    argument_field = 'parameters'

    @classmethod
    def build_required_response_format(cls, tools: list[Tool], *, reasoning: bool) -> dict[str, Any]:
        """Require InternLM action blocks with JSON parameter constraints."""
        from xgrammar.structural_tag import (
            AnyTextFormat,
            ConstStringFormat,
            JSONSchemaFormat,
            RegexFormat,
            SequenceFormat,
            StructuralTag,
            TagFormat,
            TagsWithSeparatorFormat,
        )

        calls = TagsWithSeparatorFormat(
            tags=[TagFormat(
                begin=(f'{cls.get_tool_open_tag()}{{{json.dumps(cls.name_field)}: '
                       f'{json.dumps(tool.function.name)}, {json.dumps(cls.argument_field)}: '),
                content=JSONSchemaFormat(json_schema=tool.function.parameters or {'type': 'object'}),
                end='}' + cls.get_tool_close_tag(),
            ) for tool in tools],
            separator='\n',
            at_least_one=True,
        )
        # Whitespace after the last call must not require another call before EOS.
        content = SequenceFormat(elements=[calls, RegexFormat(pattern=r'\s*')])
        if reasoning:
            return StructuralTag(format=SequenceFormat(elements=[
                TagFormat(begin='', content=AnyTextFormat(), end='</think>'),
                ConstStringFormat(value='\n\n'),
                content,
            ])).model_dump(mode='json')
        return StructuralTag(format=content).model_dump(mode='json')

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)

    @classmethod
    def get_tool_open_tag(cls) -> str | None:
        return '<|action_start|><|plugin|>'

    @classmethod
    def get_tool_close_tag(cls) -> str | None:
        return '<|action_end|>'
||||||| Stash base
=======
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from .tool_parser import ToolParser, ToolParserManager

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import (
        ChatCompletionRequest,
        DeltaToolCall,
        ToolCall,
    )

@ToolParserManager.register_module(['internlm', 'intern-s1'])
class Internlm2ToolParser(ToolParser):
    """Tool parser for InternLM JSON tool-call payloads."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        request.spaces_between_special_tokens = False
        return super().adjust_request(request)

    def get_tool_open_tag(self) -> str | None:
        return '<|action_start|><|plugin|>'

    def get_tool_close_tag(self) -> str | None:
        return '<|action_end|>'

    def get_tool_payload_format(self) -> str:
        return 'json'

    def decode_tool_incremental(self, added_text: str, *, final: bool) -> list[DeltaToolCall]:
        """Decode incremental JSON tool payload."""
        return self._decode_tool_incremental_json(added_text=added_text, final=final)

    def parse_tool_call_complete(self, payload: str) -> ToolCall | None:
        return self._parse_tool_call_complete_json(payload)
>>>>>>> Stashed changes
