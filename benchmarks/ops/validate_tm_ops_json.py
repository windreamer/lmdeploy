#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Validate op-level benchmark evidence JSON records under benchmarks/ops/.

Required top-level keys (tm-kernel-integration spec): device, capability,
shapes, impl, timings, git_sha. Records carry one entry per (path, impl)
case; ``timings`` maps case name -> list of seconds.

Usage: python benchmarks/ops/validate_tm_ops_json.py [files ...]

Note: the W4A16 ticket (02) may land its own validate harness; until that
merges, this standalone checker enforces the same schema so records stay
compatible.
"""
import json
import sys

REQUIRED_KEYS = ('device', 'capability', 'shapes', 'impl', 'timings', 'git_sha')


def validate(path: str) -> list[str]:
    errors = []
    with open(path) as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]
    for i, record in enumerate(records):
        for key in REQUIRED_KEYS:
            if key not in record:
                errors.append(f'{path}[{i}]: missing required key {key!r}')
        timings = record.get('timings', {})
        if not isinstance(timings, dict) or not timings:
            errors.append(f'{path}[{i}]: timings must be a non-empty object')
        elif any(not isinstance(v, list) or not v for v in timings.values()):
            errors.append(f'{path}[{i}]: each timings entry must be a non-empty list')
        cap = record.get('capability')
        if not (isinstance(cap, list) and len(cap) == 2 and all(isinstance(x, int) for x in cap)):
            errors.append(f'{path}[{i}]: capability must be [major, minor]')
    return errors


def main(argv):
    files = argv[1:]
    if not files:
        import glob
        files = sorted(glob.glob('benchmarks/ops/*.json'))
    errors = []
    for path in files:
        errors.extend(validate(path))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f'OK: {len(files)} record file(s) valid')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
