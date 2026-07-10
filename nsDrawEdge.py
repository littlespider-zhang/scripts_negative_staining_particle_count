"""
draw_segmentation_contour.py

根据 ilastik（或类似工具）导出的概率图 HDF5 文件（*_Probabilities.h5），
计算分割 mask，并把分割轮廓画在对应的原始图像上，方便直观检查分割效果。

原图匹配规则：
    ilastik 通常把 "xxx.tif" 的概率图存成 "xxx_Probabilities.h5"，
    所以本脚本会把 h5 文件名去掉 "_Probabilities" 后缀，
    再去同一目录下按 raw_image_extensions 里的扩展名找同名原图。
    如果找不到匹配的原图，该文件会被跳过（不影响其他文件继续处理）。

输出：
    结果保存在原图同级目录下，文件名为 "contour_{原图文件名}"。

用法：
    直接在下方 CONFIG 里改参数，然后运行：
        python draw_segmentation_contour.py
    不需要任何命令行参数。
"""

from pathlib import Path

import numpy as np
import h5py
import tifffile as tiff
from skimage import measure, transform
from PIL import Image, ImageDraw

# =========================
# 1. 参数
# =========================

CONFIG = {
    # 要批量处理的文件夹，会递归处理所有子文件夹
    "input_folder": "./75WT",

    # 概率图文件的匹配规则
    "h5_pattern": "*_Probabilities.h5",

    # 原图可能的扩展名，会依次尝试匹配（不区分大小写）
    "raw_image_extensions": [".tif", ".tiff", ".png"],

    # HDF5 内部数据集路径，留 None 则自动查找第一个 dataset
    "dataset_path": None,

    "target_class": 0,
    "prob_threshold": 0.8,

    # 轮廓画法
    "contour_color": (255, 0 ,0),  # RGB，默认红色
    "contour_width": 2,            # 线宽（像素）
}


# =========================
# 2. 查找待处理文件 / 匹配原图
# =========================

def find_h5_files(folder, pattern):
    """递归查找 folder 及其所有子文件夹中匹配 pattern 的概率图 h5 文件。"""
    folder = Path(folder)
    return sorted(p for p in folder.rglob(pattern) if p.is_file())


def find_raw_image(h5_path, extensions):
    """根据 h5 文件名反推对应的原图文件，在同一目录下查找。

    ilastik 的命名习惯是 "原文件名(含扩展名去掉).h5" 之前加 "_Probabilities"，
    比如 "local_xxx.tif" -> "local_xxx_Probabilities.h5"。
    """
    stem = h5_path.stem
    suffix = "_Probabilities"
    raw_stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem

    folder = h5_path.parent
    for ext in extensions:
        candidate = folder / f"{raw_stem}{ext}"
        if candidate.exists():
            return candidate
        # 再尝试大小写不敏感匹配
        for p in folder.glob(f"{raw_stem}.*"):
            if p.suffix.lower() == ext.lower():
                return p
    return None


# =========================
# 3. 从概率图计算 mask
# =========================

def find_dataset(group):
    """在 HDF5 group 里递归找到第一个 dataset 的路径。"""
    for k in group.keys():
        item = group[k]
        if isinstance(item, h5py.Dataset):
            return item.name
        elif isinstance(item, h5py.Group):
            try:
                return find_dataset(item)
            except Exception:
                pass
    return None


def load_mask_from_h5(h5_path, dataset_path=None, target_class=0, prob_threshold=0.8):
    with h5py.File(h5_path, "r") as f:
        used_dataset_path = dataset_path
        if used_dataset_path is None:
            used_dataset_path = find_dataset(f)
        prob = f[used_dataset_path][:]

    if prob.ndim == 4:
        prob = prob[0]

    particle_prob = prob[..., target_class]
    mask = particle_prob >= prob_threshold
    return mask


# =========================
# 4. 在原图上画轮廓
# =========================

def load_raw_as_rgb(raw_path):
    """读取原图并转成 uint8 RGB，用于画彩色轮廓。"""
    img = tiff.imread(raw_path)

    if img.ndim == 3:
        # 多通道图，合并为灰度
        img = img.mean(axis=-1)

    if img.dtype != np.uint8:
        # 稳健归一化到 0-255
        img = img.astype(np.float64)
        lo, hi = np.percentile(img, [0.5, 99.5])
        img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
        img = (img * 255).astype(np.uint8)

    rgb = np.stack([img, img, img], axis=-1)
    return rgb


def draw_contours_on_image(rgb_array, mask, color=(255, 0, 0), width=2):
    """用 skimage 找轮廓，再用 PIL 在图像上画出来，保持原图分辨率不变。"""
    if mask.shape != rgb_array.shape[:2]:
        # mask 和原图尺寸不一致时，按最近邻缩放对齐（理论上不应发生，做个保险）
        mask = transform.resize(
            mask.astype(float), rgb_array.shape[:2], order=0, preserve_range=True
        ) >= 0.5

    contours = measure.find_contours(mask.astype(float), level=0.5)

    pil_img = Image.fromarray(rgb_array)
    draw = ImageDraw.Draw(pil_img)

    for contour in contours:
        # find_contours 返回的是 (row, col)，PIL 需要 (x, y) = (col, row)
        points = [(float(c), float(r)) for r, c in contour]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")

    return np.array(pil_img)


# =========================
# 5. 单个文件的处理逻辑
# =========================

def process_one(h5_path, raw_image_extensions, dataset_path=None, target_class=0,
                 prob_threshold=0.8, contour_color=(255, 0, 0), contour_width=2):

    raw_path = find_raw_image(h5_path, raw_image_extensions)
    if raw_path is None:
        raise FileNotFoundError(f"找不到与 {h5_path.name} 对应的原图（尝试的扩展名：{raw_image_extensions}）")

    mask = load_mask_from_h5(
        h5_path, dataset_path=dataset_path,
        target_class=target_class, prob_threshold=prob_threshold,
    )

    rgb = load_raw_as_rgb(raw_path)
    result = draw_contours_on_image(rgb, mask, color=contour_color, width=contour_width)

    out_path = raw_path.parent / f"contour_{raw_path.name}"
    # 轮廓图统一存成 PNG，避免用 TIFF 压缩参数处理 RGB 数据时出问题
    out_path = out_path.with_suffix(".png")
    Image.fromarray(result).save(out_path)

    return out_path


# =========================
# 6. 批量处理入口
# =========================

def run_batch(input_folder, h5_pattern, raw_image_extensions, dataset_path=None,
              target_class=0, prob_threshold=0.8, contour_color=(255, 0, 0),
              contour_width=2):

    h5_files = find_h5_files(input_folder, h5_pattern)
    if not h5_files:
        print(f"在 {input_folder} 及其子文件夹中没有找到匹配 {h5_pattern!r} 的文件。")
        return

    print(f"共找到 {len(h5_files)} 个概率图文件，开始批量处理 ...")
    n_ok, n_fail = 0, 0
    for i, h5_path in enumerate(h5_files, start=1):
        print(f"[{i}/{len(h5_files)}] 处理 {h5_path} ...")
        try:
            out_path = process_one(
                h5_path, raw_image_extensions,
                dataset_path=dataset_path, target_class=target_class,
                prob_threshold=prob_threshold,
                contour_color=contour_color, contour_width=contour_width,
            )
            print(f"    -> 已保存 {out_path}")
            n_ok += 1
        except Exception as e:
            # 单个文件出错不影响其他文件继续处理
            print(f"    !! 处理失败：{e}")
            n_fail += 1

    print(f"批量处理完成：成功 {n_ok} 个，失败 {n_fail} 个。")


def main():
    run_batch(
        CONFIG["input_folder"],
        CONFIG["h5_pattern"],
        CONFIG["raw_image_extensions"],
        dataset_path=CONFIG["dataset_path"],
        target_class=CONFIG["target_class"],
        prob_threshold=CONFIG["prob_threshold"],
        contour_color=CONFIG["contour_color"],
        contour_width=CONFIG["contour_width"],
    )


if __name__ == "__main__":
    main()