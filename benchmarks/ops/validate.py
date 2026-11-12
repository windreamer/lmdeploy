#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Validate benchmark record JSONs under benchmarks/ops/.

Checks the schema shared by all op benchmark records:
    family, device{name, capability}, git_sha, dtype, shapes[].impls timings.

Usage:
    python benchmarks/ops/validate.py                # validate all records
    python benchmarks/ops/validate.py <file.json>    # validate one record
"""
import json
import sys
from pathlib import Path


def _fail(path, message):
    print(f'FAIL {path}: {message}')
    return False


def validate(path: Path) -> bool:
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return _fail(path, f'invalid JSON: {exc}')

    for key in ('family', 'device', 'git_sha', 'dtype', 'shapes'):
        if key not in record:
            return _fail(path, f'missing top-level key {key!r}')

    device = record['device']
    for key in ('name', 'capability'):
        if key not in device:
            return _fail(path, f'missing device key {key!r}')
    capability = device['capability']
    if not (isinstance(capability, list) and len(capability) == 2 and all(isinstance(v, int) for v in capability)):
        return _fail(path, f'capability must be [major, minor], got {capability!r}')

    if not record['git_sha'] or len(record['git_sha']) < 7:
        return _fail(path, f'git_sha malformed: {record["git_sha"]!r}')

    if not record['shapes']:
        return _fail(path, 'no shape entries')
    for entry in record['shapes']:
        impls = entry.get('impls', {})
        timings = [k for k in impls if k.endswith('_ms')]
        if not timings:
            return _fail(path, f'shape {entry} has no timing under impls')
        for key, value in impls.items():
            if key.endswith('_ms') and (not isinstance(value, (int, float)) or value <= 0):
                return _fail(path, f'bad timing {key}={value!r}')

    print(f'OK   {path} (family={record["family"]}, capability={capability}, shapes={len(record["shapes"])})')
    return True


def main() -> int:
    args = sys.argv[1:]
    base = Path(__file__).resolve().parent
    files = [Path(a) for a in args] if args else sorted(base.glob('*.json'))
    if not files:
        print('no benchmark records found')
        return 1
    return 0 if all(validate(path) for path in files) else 1


if __name__ == '__main__':
    sys.exit(main())
