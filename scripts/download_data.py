"""Download Amazon 2014 Beauty (McAuley, SNAP-hosted) reviews (5-core) and metadata into data/raw/."""

import argparse
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
FILES = ["reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"]


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if total and tmp.stat().st_size != total:
        raise IOError(f"size mismatch for {dest.name}: got {tmp.stat().st_size}, expected {total}")
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in FILES:
        dest = out_dir / name
        url = f"{BASE_URL}/{name}"
        if dest.exists() and not args.force:
            print(f"skip {dest} (exists)")
            continue
        try:
            download(url, dest)
        except Exception as e:  # noqa: BLE001 - report every failure with its URL
            print(f"FAILED {url}: {e}", file=sys.stderr)
            failed.append(url)

    if failed:
        print("\nDownload these manually into", out_dir, file=sys.stderr)
        for url in failed:
            print("  " + url, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
