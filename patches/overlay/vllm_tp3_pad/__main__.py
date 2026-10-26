"""python3 -m vllm_tp3_pad --selftest -- torch-free layout selftest.

Asserts the exact intended segment layout of ALL 18 pad-table rules
(original-coordinate, tail-first insert application) so the ops-ordering
defect class -- and the append-only [[0,n]] op edits -- can never return
silently (see docs/FIXLOG.md). Exits 0 on pass, 1 on failure, 2 on usage
error.
"""

import sys

from . import selftest

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    if args:
        print(f"usage: python3 -m vllm_tp3_pad [--selftest]", file=sys.stderr)
        sys.exit(2)
    sys.exit(selftest())
