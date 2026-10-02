"""Build the cold-start split (TIGER paper Sec. 4.3) from the main dataset; see recsys.data.coldstart.

Usage: python scripts/make_coldstart.py [--source data/processed_tieshuffle] [--out data/processed_coldstart]
                                        [--fraction 0.05] [--seed 0]
"""

import argparse
import json
from pathlib import Path

from recsys.data.coldstart import make_coldstart_split, save_coldstart, select_held_out
from recsys.data.dataset import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("data/processed_tieshuffle"))
    parser.add_argument("--out", type=Path, default=Path("data/processed_coldstart"))
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split = load_split(args.source)
    held_out = select_held_out(split.test, args.fraction, args.seed)
    cold, info = make_coldstart_split(split, held_out)
    save_coldstart(args.out, args.source, cold, held_out, info, args.fraction, args.seed)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
