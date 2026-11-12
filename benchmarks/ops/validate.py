# Copyright (c) OpenMMLab. All rights reserved.
"""Validate op-level benchmark evidence records under benchmarks/ops/.

Every record must carry: family, impl, device, capability, shapes,
timings_us (positive finite), git_sha, timestamp. Usage:

    python benchmarks/ops/validate.py [files...]
"""

import json
import re
import sys

REQUIRED = ('family', 'impl', 'device', 'capability', 'shapes', 'timings_us', 'git_sha', 'timestamp')
SHA_RE = re.compile(r'^[0-9a-f]{40}$')


def validate(path):
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f'{path}: expected a non-empty list of records')
    for i, rec in enumerate(records):
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            raise ValueError(f'{path}[{i}]: missing fields {missing}')
        if not SHA_RE.match(rec['git_sha']):
            raise ValueError(f'{path}[{i}]: bad git_sha {rec["git_sha"]!r}')
        if not rec['shapes']:
            raise ValueError(f'{path}[{i}]: empty shapes')
        for key, us in rec['timings_us'].items():
            if isinstance(us, str):
                if not us.startswith('unavailable:'):
                    raise ValueError(f'{path}[{i}]: malformed timing {key}={us!r}')
                continue
            if not isinstance(us, (int, float)) or not us > 0:
                raise ValueError(f'{path}[{i}]: non-positive timing {key}={us}')
        keys = [f'M{s["M"]}_K{s["K"]}_N{s["N"]}' for s in rec['shapes']]
        for key in rec['timings_us']:
            if not any(key.startswith(k) for k in keys):
                raise ValueError(f'{path}[{i}]: timing {key} does not match any shape')
    print(f'{path}: OK ({len(records)} records)')


if __name__ == '__main__':
    files = sys.argv[1:] or ['benchmarks/ops/blocked_fp8_sm80.json']
    for f in files:
        validate(f)
