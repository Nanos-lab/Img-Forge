"""从 new.tif 中心裁剪一小块，用于拼接/配准测试。

用法:
    python scripts/crop_center.py [--size 200] [--output test_clip.tif]
"""

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window


def crop_center(src_path: str, out_path: str, crop_size: int) -> None:
    """从影像中心裁剪一个正方形小块。

    Args:
        src_path:  输入 TIFF 路径。
        out_path:  输出 TIFF 路径。
        crop_size: 裁剪边长（像素）。
    """
    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        if crop_size > h or crop_size > w:
            print(f"输入影像 {w}x{h}，裁剪尺寸 {crop_size} 过大")
            sys.exit(1)

        col_off = (w - crop_size) // 2
        row_off = (h - crop_size) // 2

        window = Window(col_off=col_off, row_off=row_off, width=crop_size, height=crop_size)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(height=crop_size, width=crop_size, transform=transform)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

    print(f"裁剪完成: {out_path}  ({crop_size}x{crop_size})")


def main():
    parser = argparse.ArgumentParser(description="从 new.tif 中心裁剪一块用于测试")
    parser.add_argument("--size", type=int, default=200, help="裁剪边长（像素，默认 200）")
    parser.add_argument("--output", type=str, default=None, help="输出文件名（默认 temp/test_clip_{size}.tif）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    src = project_root / "temp" / "new.tif"

    if not src.exists():
        print(f"❌ 未找到 {src}")
        sys.exit(1)

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = project_root / out
    else:
        out = project_root / "temp" / f"test_clip_{args.size}.tif"

    crop_center(str(src), str(out), args.size)


if __name__ == "__main__":
    main()
