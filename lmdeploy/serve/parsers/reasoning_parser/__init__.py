<<<<<<< Updated upstream
# Copyright (c) OpenMMLab. All rights reserved.
from .deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from .deepseek_v4_reasoning_parser import DeepSeekV4ReasoningParser
from .deepseek_v32_reasoning_parser import DeepSeekV32ReasoningParser
from .reasoning_parser import LEGACY_REASONING_PARSER_NAMES, ReasoningParser, ReasoningParserManager

__all__ = [
    'LEGACY_REASONING_PARSER_NAMES',
    'ReasoningParser',
    'ReasoningParserManager',
    'DeepSeekV3ReasoningParser',
    'DeepSeekV32ReasoningParser',
    'DeepSeekV4ReasoningParser',
]
||||||| Stash base
=======
# Copyright (c) OpenMMLab. All rights reserved.
from .deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from .reasoning_parser import ReasoningParser, ReasoningParserManager

__all__ = [
    'ReasoningParser',
    'ReasoningParserManager',
    'DeepSeekV3ReasoningParser',
]
>>>>>>> Stashed changes
