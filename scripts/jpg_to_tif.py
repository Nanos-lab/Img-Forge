"""批量将 JPG 图片转换为普通 TIFF 格式。

用途：为 obb_detect 接口准备演示素材。转换后的 TIFF 不含地理参考信息
（无 CRS/transform），仅用作接口输入格式测试，输出的 GeoJSON 坐标不代表
真实经纬度。

用法:
    python scripts/jpg_to_tif.py --input test/DOTAv1/images/test --output test/DOTAv1/images/test_tif
    python scripts/jpg_to_tif.py --input test/DOTAv1/images/test --output test/DOTAv1/images/test_tif --limit 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine


def convert_one(src_path: Path, dst_path: Path) -> None:
    """读取单张 JPG，写出为无地理参考的 TIFF。

    Args:
        src_path: 输入 JPG 文件路径。
        dst_path: 输出 TIFF 文件路径。
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)  # (H, W, 3) BGR, uint8
    if img is None:
        raise ValueError(f"无法读取图片: {src_path}")

    # BGR -> RGB，(H, W, C) -> (C, H, W)，匹配 rasterio 的波段顺序
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = np.transpose(img_rgb, (2, 0, 1))

    height, width = img_rgb.shape[0], img_rgb.shape[1]

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": data.dtype,
        "crs": None,
        "transform": Affine.identity(),
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)


def convert_dir(input_dir: Path, output_dir: Path, limit: int = 0) -> None:
    """批量转换目录下的所有 JPG 文件。

    Args:
        input_dir: 输入目录，包含 .jpg / .jpeg 文件。
        output_dir: 输出目录，转换结果同名 .tif 文件。
        limit: 最多转换的文件数量，0 表示不限制。
    """
    jpg_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg"))
    if not jpg_files:
        print(f"未在 {input_dir} 找到 jpg/jpeg 文件")
        sys.exit(1)

    if limit > 0:
        jpg_files = jpg_files[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    for src_path in jpg_files:
        dst_path = output_dir / f"{src_path.stem}.tif"
        try:
            convert_one(src_path, dst_path)
            ok += 1
        except Exception as exc:
            print(f"转换失败: {src_path.name} -> {exc}")
            failed += 1

    print(f"转换完成: 成功 {ok} 个，失败 {failed} 个，输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="批量将 JPG 转换为普通 TIFF（无地理参考）")
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入目录，包含待转换的 jpg 文件",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出目录，转换结果按同名写入 .tif 文件",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最多转换的文件数量，0 表示不限制（默认 0）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    convert_dir(input_dir, output_dir, args.limit)


if __name__ == "__main__":
    main()
