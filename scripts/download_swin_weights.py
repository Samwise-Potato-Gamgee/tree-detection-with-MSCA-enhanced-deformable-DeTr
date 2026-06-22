"""
Downloads Swin-T ImageNet-22K pretrained weights.

Usage:
  python scripts/download_swin_weights.py --output_dir pretrained_weights
"""

import argparse
import urllib.request
from pathlib import Path

SWIN_T_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/"
    "v1.0.0/swin_tiny_patch4_window7_224.pth"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="pretrained_weights",
                        help="Directory to save the downloaded weights")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "swin_tiny_patch4_window7_224.pth"

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print(f"Downloading Swin-T weights to {output_path} ...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(SWIN_T_URL, output_path, reporthook=_progress)
    print(f"\nDone. Saved to {output_path}")


if __name__ == "__main__":
    main()
