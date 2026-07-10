"""
nsProbability.py

批量读取 ilastik（或类似工具）导出的概率图 HDF5 文件（*_Probabilities.h5），
按阈值做二值分类，输出 mask 和伪彩色分割结果图。

处理结果保存在每个 h5 文件所在的同级目录下：
    - segmentation_{原文件名}.png       彩色分割结果
    - segmentation_mask_{原文件名}.png  二值 mask

用法：
    直接在下方 CONFIG 里改参数，然后运行：
        python nsProbability.py
    不需要任何命令行参数。
"""

from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from imageio import imwrite

# =========================
# 1. 参数
# =========================

CONFIG = {
    # 要批量处理的文件夹，会递归处理所有子文件夹
    "input_folder": "./75WT",

    # 会被处理的文件名匹配规则（glob pattern，不区分大小写扩展名）
    # 默认只处理以 _Probabilities.h5 结尾的文件，避免把其他 h5 文件也处理了
    "file_pattern": "*_Probabilities.h5",

    # HDF5 内部数据集路径，留 None 则自动查找第一个 dataset
    "dataset_path": None,

    "target_class": 0,
    "prob_threshold": 0.8,

    # 处理完每张图后是否弹窗显示（批量处理建议设为 False，避免逐个阻塞）
    "show_plot": False,
}


# =========================
# 2. 查找待处理文件
# =========================

def find_h5_files(folder, pattern, exclude_prefixes=("segmentation_",)):
    """递归查找 folder 及其所有子文件夹中匹配 pattern 的 h5 文件。

    会自动跳过文件名以 exclude_prefixes 开头的文件，避免把本脚本之前
    生成的结果文件当成输入再处理一遍（正常情况下结果是 png，不会匹配到，
    这里保留是为了防御性处理）。
    """
    folder = Path(folder)
    files = [
        p for p in folder.rglob(pattern)
        if p.is_file() and not p.name.startswith(exclude_prefixes)
    ]
    return sorted(files)


# =========================
# 3. 单个文件的处理逻辑
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


def process_h5_file(h5_file, dataset_path=None, target_class=0, prob_threshold=0.8,
                     show_plot=False):
    """处理单个概率图 h5 文件，保存 mask 和伪彩色分割结果到同级目录。"""
    h5_path = Path(h5_file)
    output_dir = h5_path.parent
    stem = h5_path.stem

    output_overlay = output_dir / f"segmentation_{stem}.png"
    output_mask = output_dir / f"segmentation_mask_{stem}.png"

    # ---- 读取 HDF5 ----
    with h5py.File(h5_path, "r") as f:
        used_dataset_path = dataset_path
        if used_dataset_path is None:
            used_dataset_path = find_dataset(f)
        print("    Using dataset:", used_dataset_path)
        prob = f[used_dataset_path][:]

    print("    shape:", prob.shape)

    # ---- 处理维度 ----
    if prob.ndim == 4:
        prob = prob[0]

    particle_prob = prob[..., target_class]

    # ---- 二值分类 ----
    mask = particle_prob >= prob_threshold

    # ---- 伪彩色（二值颜色映射）----
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # EM推荐配色
    particle_color = np.array([0.95, 0.78, 0.25])    # 亮黄色
    background_color = np.array([0.25, 0.55, 0.85])  # 蓝色

    rgb[mask] = particle_color
    rgb[~mask] = background_color

    # ---- 保存 mask ----
    imwrite(output_mask, (mask.astype(np.uint8) * 255))
    print("    Saved mask:", output_mask)

    # ---- 保存彩色结果 ----
    imwrite(output_overlay, (rgb * 255).astype(np.uint8))
    print("    Saved colored segmentation:", output_overlay)

    # ---- 显示（可选）----
    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.title(f"Binary segmentation (thr={prob_threshold})")
        plt.axis("off")
        plt.show()

    return output_overlay, output_mask


# =========================
# 4. 批量处理入口
# =========================

def run_batch(input_folder, file_pattern, dataset_path=None, target_class=0,
              prob_threshold=0.8, show_plot=False):

    files = find_h5_files(input_folder, file_pattern)
    if not files:
        print(f"在 {input_folder} 及其子文件夹中没有找到匹配 {file_pattern!r} 的文件。")
        return

    print(f"共找到 {len(files)} 个文件，开始批量处理 ...")
    n_ok, n_fail = 0, 0
    for i, h5_file in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] 处理 {h5_file} ...")
        try:
            process_h5_file(
                h5_file,
                dataset_path=dataset_path,
                target_class=target_class,
                prob_threshold=prob_threshold,
                show_plot=show_plot,
            )
            n_ok += 1
        except Exception as e:
            # 单个文件出错不影响其他文件继续处理
            print(f"    !! 处理失败：{e}")
            n_fail += 1

    print(f"批量处理完成：成功 {n_ok} 个，失败 {n_fail} 个。")


def main():
    run_batch(
        CONFIG["input_folder"],
        CONFIG["file_pattern"],
        dataset_path=CONFIG["dataset_path"],
        target_class=CONFIG["target_class"],
        prob_threshold=CONFIG["prob_threshold"],
        show_plot=CONFIG["show_plot"],
    )


if __name__ == "__main__":
    main()