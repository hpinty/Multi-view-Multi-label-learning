# -*- coding: utf-8 -*-
"""把 result/*/ 下的指标 xlsx 画成 K 折线图（每个指标一张 PNG）。

xlsx 布局（main.py writeperform 的累积格式）：
  数据集名单独一行（仅 A 列有值），随后每行 = [算法名, k=1..N 的指标值]。
  k 就是列号（1 起），不同数据集的 N 可以不同。

横坐标 = 选中特征占比（k / 总特征数 ×100%，学长定，对齐老师论文画法）。
总特征数 xlsx 里没存，从 data/*.mat 读各视图维度求和；mat 缺失时用
k 上限≈20% 反推（main.py 的 select_num = int(总特征数*0.2)）。

跑法：python plot_k_curves.py            # 扫描 result/ 下所有子目录
输出：result/<子目录>/figs/<指标名>.png
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
import scipy.io as sio

ROOT = Path(__file__).resolve().parent

METRICS = {  # 文件名 -> (纵轴显示名, 是否越高越好)
    "average_precision_score": ("Average Precision", True),
    "hamming_loss": ("Hamming Loss", False),
    "label_ranking_loss": ("Ranking Loss", False),
    "coverage_error": ("Coverage", False),
    "zero_one_loss": ("0/1 Loss", False),
}

# 固定分类色序（dataviz 参考调色板，浅色模式；按槽位顺序取用，不循环生成）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load_total_features():
    """数据集名(小写) -> 总特征数（各视图维度之和）。"""
    totals = {}
    for f in sorted((ROOT / "data").glob("*.mat")):
        try:
            views = sio.loadmat(f)["view"]
            totals[f.stem.lower()] = int(
                sum(views[0][s].shape[1] for s in range(views.shape[1])))
        except Exception:
            pass
    if not totals:
        print("注意：data/ 下没读到任何 .mat，占比将用 k 上限反推。")
    return totals


def parse_xlsx(path):
    """返回 [(数据集, 算法, ks, values), ...]；k=列号。"""
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    series, dataset = [], None
    for row in ws.iter_rows(values_only=True):
        cells = list(row)
        if not cells or cells[0] is None:
            continue
        values = []
        for cell in cells[1:]:
            if cell is None:
                break
            values.append(float(cell))
        if values:
            series.append((dataset or "?", str(cells[0]).strip(),
                           list(range(1, len(values) + 1)), values))
        else:
            dataset = str(cells[0]).strip()
    return series


def style_axes(ax, yname, higher_better):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Percentage of selected features (%)", color=INK2)
    ax.set_ylabel(yname, color=INK2)
    ax.text(0.995, 0.02, "higher is better" if higher_better else "lower is better",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=INK2)


def plot_metric(path, yname, higher_better, out_path, totals):
    series = parse_xlsx(path)
    if not series:
        print(f"跳过（无数据行）：{path}")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    xs_list = []
    for i, (dataset, alg, ks, vals) in enumerate(series):
        d_total = totals.get(dataset.lower())
        if d_total is None:  # mat 缺失时按 main.py 口径（k 上限=int(总特征数*0.2)）反推
            d_total = max(1, round(ks[-1] / 0.2))
            print(f"注意：{dataset} 在 data/ 里没找到，总特征数反推为 {d_total}。")
        xs = [k / d_total * 100 for k in ks]
        xs_list.append(xs)
        color = SERIES[i % len(SERIES)]  # 超过 8 条时重复色序并提示
        ax.plot(xs, vals, color=color, linewidth=2,
                marker="o", markersize=4.5,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"{dataset} · {alg}", zorder=3)
        # 选择性直标：只标最佳点（AP 取最高，损失类取最低）
        best = max(range(len(vals)), key=lambda j: vals[j]) if higher_better \
            else min(range(len(vals)), key=lambda j: vals[j])
        ax.annotate(f"{xs[best]:.1f}%\n{vals[best]:.3f}",
                    xy=(xs[best], vals[best]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8.5, color=INK2)
    n = len(series)
    if n > 8:
        print(f"注意：{path.name} 有 {n} 条序列，颜色已复用，建议分图。")
    if n > 1:
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9,
                  loc="lower right" if higher_better else "upper right")
    title = series[0][0] if n == 1 else f"{n} dataset curves"
    ax.set_title(f"{yname} vs percentage of selected features — {title}",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    style_axes(ax, yname, higher_better)
    x_hi = max(xs[-1] for xs in xs_list)
    span_p = x_hi - min(xs[0] for xs in xs_list)
    step = 1 if span_p <= 8 else 2 if span_p <= 14 else 5
    ticks = list(range(0, math.ceil(x_hi) + 1, step))
    ax.set_xticks(ticks)
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) or 1.0
    ax.set_ylim(y0 - 0.08 * span, y1 + 0.06 * span)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    # 附数据表（图的可替代视图）
    print(f"{out_path.relative_to(ROOT)}")
    for (dataset, alg, ks, vals), xs in zip(series, xs_list):
        head = " ".join(f"{v:.3f}" for v in vals[:14])
        print(f"  {dataset}·{alg}: {xs[0]:.1f}%..{xs[-1]:.1f}% (k=1..{ks[-1]})  "
              f"{head}{' …' if len(vals) > 14 else ''}")


def main():
    totals = load_total_features()
    found = 0
    for subdir in sorted(p for p in (ROOT / "result").iterdir() if p.is_dir()):
        for stem, (yname, higher) in METRICS.items():
            path = subdir / f"{stem}.xlsx"
            if path.is_file():
                plot_metric(path, yname, higher,
                            subdir / "figs" / f"{stem}.png", totals)
                found += 1
    print(f"共生成 {found} 张图。")


if __name__ == "__main__":
    main()
