# Copyright (c) OpenMMLab. All rights reserved.
"""Dump the AWQ Triton golden reference for the W4A16 gate.

Runs the pytorch engine with the AWQ Triton path (the reference backend) on
the fixed 32-prompt set and writes ``awq_ref.json``. Usage:

    .venv/bin/python tests/pytorch/engine/dump_awq_ref.py \
        --model models/llama-3.1-8b-awq --out awq_ref.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from w4a16_golden import dump_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='model path (AWQ W4A16 checkpoint)')
    parser.add_argument('--out', default='awq_ref.json', help='output json path')
    parser.add_argument('--backend',
                        default='triton',
                        help='reference backend selection; must stay off the TurboMind values')
    args = parser.parse_args()
    if args.backend in ('auto', 'turbomind'):
        raise SystemExit('the reference backend must not select the TurboMind path')
    payload = dump_reference(args.model, args.out, backend=args.backend)
    print(f"wrote {args.out}: {len(payload['prompts'])} prompts, "
          f"{payload['max_new_tokens']} greedy tokens each")


if __name__ == '__main__':
    main()
