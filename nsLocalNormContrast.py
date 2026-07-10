#!/usr/bin/env python3
"""
em_particle_pipeline.py

负染电镜（negative-stain EM）图像后处理流程（简化版）。

当前流程只保留一步：
    1. 读取 TIFF，裁掉底部 scale bar 条带
    2. Local normalization（局部均值/标准差归一化，消除染色/光照不均）

两种运行模式（在下方 CONFIG["mode"] 里选择）：
    - "single"：处理单张图片，结果存到指定的 output_dir
    - "batch" ：递归处理指定文件夹及其所有子文件夹里的图片，
                处理结果直接保存在每张原图所在的目录下，
                文件名前面加上 "local_" 前缀（如 local_xxx.tif）

每次处理图片时，会在结果保存的同一目录下写一个 log.txt，
记录本次处理使用的参数（追加写入，不会覆盖之前的记录）。

用法：
    直接在下方 CONFIG 里改参数，然后运行：
        python em_particle_pipeline.py
    不需要任何命令行参数。
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


# ===========================================================================
# CONFIG —— 在这里直接修改参数，不需要用命令行传参
# ===========================================================================
CONFIG = {
    # 运行模式："single"（单张图片）或 "batch"（批量处理文件夹）
    "mode": "batch",

    # ---- single 模式用 ----
    "input_path": "input.tif",             # 单张图片路径
    "output_dir": "./em_pipeline_output",  # 单张模式下的输出目录

    # ---- batch 模式用 ----
    "input_folder": "./100WT",   # 要批量处理的文件夹，会递归处理所有子文件夹
    "image_extensions": [".tif", ".tiff"],  # 会被处理的图片扩展名（不区分大小写）

    # ---- 通用处理参数 ----
    # 1. scale bar
    "scalebar_height": 128,   # 底部 scale bar 高度（像素），4096x4224 -> 128

    # 2. local normalization
    "norm_sigma": 20,         # 局部均值/标准差估计用的高斯 sigma，颗粒越密可适当调小
}


# ---------------------------------------------------------------------------
# 1. 读图 + 去除 scale bar
# ---------------------------------------------------------------------------

def load_image_strip_scalebar(path, scalebar_height=128):
    """读取 TIFF 并裁掉底部的 scale bar 条带。

    Parameters
    ----------
    path : str or Path
        输入 TIFF 路径
    scalebar_height : int
        底部 scale bar 的高度（像素），默认 128，对应 4096x4224 -> 4096x4096

    Returns
    -------
    np.ndarray (float64)
        裁剪后的灰度图
    """
    img = tifffile.imread(path)
    if img.ndim == 3:
        # 万一是 RGB / 多通道，合并为灰度
        img = img.mean(axis=-1)

    h, w = img.shape
    if h <= scalebar_height:
        raise ValueError(f"图像高度 {h} <= scalebar_height {scalebar_height}，请检查输入或该参数。")

    cropped = img[: h - scalebar_height, :]
    return cropped.astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Local normalization
# ---------------------------------------------------------------------------

def local_normalize(img, sigma=50, eps=1e-6):
    """局部 z-score 归一化：(img - local_mean) / local_std

    用大尺度高斯核估计局部均值和局部标准差，可以压平染色不均/光照不均
    造成的背景起伏，这也是你之前用过、效果不错的预处理方式。
    """
    local_mean = ndi.gaussian_filter(img, sigma=sigma)
    local_sq_mean = ndi.gaussian_filter(img ** 2, sigma=sigma)
    local_var = np.clip(local_sq_mean - local_mean ** 2, a_min=0, a_max=None)
    local_std = np.sqrt(local_var)

    normed = (img - local_mean) / (local_std + eps)

    # 用稳健百分位数拉伸到 0-1，避免个别极端像素把动态范围拉得过宽
    p_low, p_high = np.percentile(normed, [0.5, 99.5])
    normed = np.clip((normed - p_low) / (p_high - p_low + eps), 0, 1)
    return normed


# ---------------------------------------------------------------------------
# 核心处理：输入一张图，输出 local normalization 结果
# ---------------------------------------------------------------------------

def process_image(input_path, scalebar_height=128, norm_sigma=50):
    """对单张图片跑 local normalization，返回结果（0-1 浮点图）。"""
    img = load_image_strip_scalebar(input_path, scalebar_height=scalebar_height)
    normed = local_normalize(img, sigma=norm_sigma)
    return normed


# ---------------------------------------------------------------------------
# 日志：把本次处理使用的参数记录到目标目录下的 log.txt（追加写入）
# ---------------------------------------------------------------------------

def append_log(log_dir, image_name, params):
    """在 log_dir/log.txt 里追加一条记录，写明处理的文件名和使用的参数。"""
    log_path = Path(log_dir) / "log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {image_name}\n")
        for key, value in params.items():
            f.write(f"    {key} = {value}\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# single 模式：处理单张图片，存到指定 output_dir
# ---------------------------------------------------------------------------

def run_pipeline(input_path, output_dir, scalebar_height=128, norm_sigma=50):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)

    print(f"[1/2] 读取 {input_path} 并去除 scale bar ...")
    print("[2/2] Local normalization ...")
    normed = process_image(input_path, scalebar_height=scalebar_height, norm_sigma=norm_sigma)

    out_path = output_dir / f"local_{input_path.name}"
    tifffile.imwrite(out_path, (normed * 255).astype(np.uint8))

    append_log(
        output_dir,
        input_path.name,
        {"scalebar_height": scalebar_height, "norm_sigma": norm_sigma},
    )

    print("完成。")
    print(f"输出文件：{out_path}")
    return normed


# ---------------------------------------------------------------------------
# batch 模式：递归处理文件夹，结果存在原图所在目录，文件名加前缀
# ---------------------------------------------------------------------------

def find_images(folder, extensions, exclude_prefixes=("local_",)):
    """递归查找 folder 及其所有子文件夹中扩展名匹配的图片文件。

    会自动跳过文件名以 exclude_prefixes 开头的文件——这些通常是本脚本之前
    批量运行时生成的结果图，避免重复运行时把结果图再当成原图处理一遍。
    """
    folder = Path(folder)
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.startswith(exclude_prefixes)
    ]
    return sorted(files)


def process_and_save_inplace(image_path, scalebar_height=128, norm_sigma=50):
    """处理单张图片，把结果保存在该图片所在的同一目录下，文件名前面加 "local_" 前缀。"""
    image_path = Path(image_path)
    normed = process_image(image_path, scalebar_height=scalebar_height, norm_sigma=norm_sigma)

    out_dir = image_path.parent
    out_path = out_dir / f"local_{image_path.name}"
    tifffile.imwrite(out_path, (normed * 255).astype(np.uint8))

    append_log(
        out_dir,
        image_path.name,
        {"scalebar_height": scalebar_height, "norm_sigma": norm_sigma},
    )

    return [out_path]


def run_batch(input_folder, image_extensions, scalebar_height=128, norm_sigma=50):

    images = find_images(input_folder, image_extensions)
    if not images:
        print(f"在 {input_folder} 及其子文件夹中没有找到匹配 {image_extensions} 的图片。")
        return

    print(f"共找到 {len(images)} 张图片，开始批量处理 ...")
    n_ok, n_fail = 0, 0
    for i, image_path in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] 处理 {image_path} ...")
        try:
            saved_paths = process_and_save_inplace(
                image_path, scalebar_height=scalebar_height, norm_sigma=norm_sigma,
            )
            for p in saved_paths:
                print(f"    -> 已保存 {p}")
            n_ok += 1
        except Exception as e:
            # 单张图片出错不影响其他图片继续处理
            print(f"    !! 处理失败：{e}")
            n_fail += 1

    print(f"批量处理完成：成功 {n_ok} 张，失败 {n_fail} 张。")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    if CONFIG["mode"] == "batch":
        run_batch(
            CONFIG["input_folder"],
            CONFIG["image_extensions"],
            scalebar_height=CONFIG["scalebar_height"],
            norm_sigma=CONFIG["norm_sigma"],
        )
    elif CONFIG["mode"] == "single":
        run_pipeline(
            CONFIG["input_path"],
            CONFIG["output_dir"],
            scalebar_height=CONFIG["scalebar_height"],
            norm_sigma=CONFIG["norm_sigma"],
        )
    else:
        raise ValueError(f'CONFIG["mode"] 必须是 "single" 或 "batch"，当前是 {CONFIG["mode"]!r}')


if __name__ == "__main__":
    main()