"""生成可直接插入 PPT 的 PCA-VAE 数学卡片（16:9 PNG）。

每张卡片只讲一个主题，公式用 matplotlib mathtext 渲染。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT = "WenQuanYi Micro Hei"
W, H = 13.333, 7.5
INK = "#1F2933"
MUTED = "#5F6B76"
TEAL = "#1F6F78"
TEAL_LIGHT = "#E5F2F1"
ORANGE = "#C46A1B"
ORANGE_LIGHT = "#FBEBDD"
GRAY_LIGHT = "#F3F5F6"
LINE = "#C9D2D9"

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.sans-serif": [FONT],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 220,
    }
)


def _canvas(title: str, subtitle: str):
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ax.plot([0.55, 0.55], [0.55, 6.75], color=TEAL, linewidth=3.0, solid_capstyle="round")
    ax.text(0.85, 6.95, title, ha="left", va="top", fontsize=25, fontweight="bold", color=INK)
    ax.text(0.85, 6.52, subtitle, ha="left", va="top", fontsize=13.5, color=MUTED)
    ax.text(0.85, 0.32, "PCA-VAE 数据增强 · 过程与数学原理", ha="left", va="center", fontsize=8.5, color="#8A959E")
    return fig, ax


def _math(ax, x: float, y: float, text: str, size: float = 20, color: str = INK):
    ax.text(x, y, text, ha="left", va="center", fontsize=size, color=color)


def _bullet(ax, x: float, y: float, text: str, size: float = 13, color: str = INK):
    ax.text(x, y, "•", ha="left", va="center", fontsize=size, color=TEAL)
    ax.text(x + 0.24, y, text, ha="left", va="center", fontsize=size, color=color)


def _box(ax, x: float, y: float, w: float, h: float, text: str, color: str, fill: str, size: float = 13):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=fill, edgecolor=color, linewidth=1.2, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, zorder=3)


def _arrow(ax, start, end, *, rad=0.0, color=INK):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            linewidth=1.2, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )


def card_overview(path: Path) -> None:
    fig, ax = _canvas(
        "PCA-VAE 数据增强｜一页总览",
        "先把曲线压缩到低维，再分别学习“浓度趋势”和“剩余残差”",
    )
    y = 4.55
    _box(ax, 0.90, y, 2.15, 1.15, "真实响应曲线\n$X\\in\\mathbb{R}^{3\\times51}$", TEAL, TEAL_LIGHT, 14)
    _box(ax, 3.55, y, 2.05, 1.15, "逐气体 PCA\n$C\\in\\mathbb{R}^{3\\times6}$", TEAL, TEAL_LIGHT, 14)
    _box(ax, 6.10, y, 2.65, 1.15, "条件均值 + VAE 残差\n$C=\\hat C_{\\mathrm{mean}}+R$", ORANGE, ORANGE_LIGHT, 13)
    _box(ax, 9.25, y, 1.95, 1.15, "反 PCA\n$\\hat X$", TEAL, TEAL_LIGHT, 14)
    _box(ax, 11.55, y, 1.25, 1.15, "TCN\n训练", "#5A6670", GRAY_LIGHT, 13)
    for x0, x1 in ((3.05, 3.55), (5.60, 6.10), (8.75, 9.25), (11.20, 11.55)):
        _arrow(ax, (x0, y + 0.58), (x1, y + 0.58))
    ax.text(4.58, 3.98, "51 点 → 6 个系数/气体", ha="center", fontsize=11, color=TEAL)
    ax.text(7.42, 3.98, "标签学主趋势，VAE 学残差", ha="center", fontsize=11, color=ORANGE)
    ax.text(10.24, 3.98, "回到 3×51 曲线", ha="center", fontsize=11, color=TEAL)
    _math(ax, 0.95, 2.45, r"$X_{n,g}\in\mathbb{R}^{51},\quad g\in\{\mathrm{H2},\mathrm{NH3},\mathrm{NO2}\},\quad K=6$", 18)
    _math(ax, 0.95, 1.85, r"$\hat X_{n,g}=\bar X_g+V_{g,K}\,\left(s_g\odot(B_g^{T}\tilde z_n+\hat r_{n,g})\right)$", 19)
    ax.text(0.95, 1.20, "所有 PCA/条件均值/VAE 只在训练折上拟合；验证和测试保持真实曲线。", fontsize=11, color=MUTED)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def card_pca(path: Path) -> None:
    fig, ax = _canvas(
        "Step 1｜PCA 压缩：51 点 → 6 个系数",
        "每个气体单独拟合 PCA，只保留响应曲线的主要形状",
    )
    ys = [5.55, 4.85, 4.15, 3.45, 2.75]
    _math(ax, 0.95, ys[0], r"$x_{n,g}\in\mathbb{R}^{51},\quad n=1,\ldots,N$", 20)
    _math(ax, 0.95, ys[1], r"$\bar x_g=\frac{1}{N}\sum_{n=1}^{N}x_{n,g}$", 20)
    _math(ax, 0.95, ys[2], r"$\tilde X_g=X_g-\mathbf{1}\bar x_g^{T}=U_g\Sigma_g V_g^{T}$", 20)
    _math(ax, 0.95, ys[3], r"$c_{n,g}=V_{g,K}^{T}\tilde x_{n,g}\in\mathbb{R}^{6}$", 20)
    _math(ax, 0.95, ys[4], r"$\hat x_{n,g}=\bar x_g+V_{g,K}c_{n,g}$", 20)

    _box(ax, 9.10, 5.15, 3.25, 1.10, "每气体 51 → 6", TEAL, TEAL_LIGHT, 18)
    _box(ax, 9.10, 3.75, 3.25, 1.10, "三气体共 3×6=18 系数", TEAL, TEAL_LIGHT, 15)
    ax.text(10.72, 3.28, "K=6 解释方差 99.998%", ha="center", fontsize=12, color=MUTED)
    ax.text(9.10, 2.65, "K=6 的验证重构误差\n约量程的 0.094%", ha="left", va="top", fontsize=12, color=MUTED)
    ax.text(0.95, 1.75, "PCA 的作用：把“生成曲线”变成“生成 18 个系数”，并保证曲线落在真实响应子空间。", fontsize=11.5, color=MUTED)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def card_condition(path: Path) -> None:
    fig, ax = _canvas(
        "Step 2｜条件均值：标签决定主要趋势",
        "用三个浓度标签线性预测每个 PCA 系数，剩余部分交给 VAE",
    )
    ys = [5.55, 4.85, 4.05, 3.25, 2.45]
    _math(ax, 0.95, ys[0], r"$z_n=(y_n-\mu_y)\oslash\sigma_y\in\mathbb{R}^{3}$", 20)
    _math(ax, 0.95, ys[1], r"$\tilde z_n=[1;z_n]\in\mathbb{R}^{4}$", 20)
    _math(ax, 0.95, ys[2], r"$c_{n,g,k}=\beta_{0,g,k}+\sum_{j=1}^{3}\beta_{j,g,k}z_{n,j}+r_{n,g,k}$", 18)
    _math(ax, 0.95, ys[3], r"$B_g=(Z^{T}Z)^{-1}Z^{T}C_g$", 20)
    _math(ax, 0.95, ys[4], r"$\hat c_{n,g}=B_g^{T}\tilde z_n,\qquad r_{n,g}=c_{n,g}-\hat c_{n,g}$", 18)

    _box(ax, 9.05, 5.05, 3.35, 0.85, "截距 1", ORANGE, ORANGE_LIGHT, 16)
    ax.text(9.20, 4.78, "平均浓度下的平均系数；\n不强制模型过原点。", fontsize=10.5, color=MUTED, va="top")
    _box(ax, 9.05, 3.35, 3.35, 0.85, "条件均值 $B_g^{T}\\tilde z_n$", ORANGE, ORANGE_LIGHT, 15)
    ax.text(9.20, 3.08, "标签决定的主趋势；\n剩余残差交给 VAE。", fontsize=10.5, color=MUTED, va="top")
    ax.text(9.05, 2.55, "$Z=[\\mathbf{1},z]$，$B_g$ 由最小二乘拟合。", fontsize=11, color=MUTED)
    ax.text(0.95, 1.55, "这一步把“生成系数”拆成：标签可解释的均值 + 标签解释不了的残差。", fontsize=11.5, color=MUTED)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def card_vae(path: Path) -> None:
    fig, ax = _canvas(
        "Step 3｜条件 VAE：只学残差",
        "编码器把残差压到 4 维隐变量；解码器重建残差；KL 约束隐变量分布",
    )
    ys = [5.45, 4.75, 4.05, 3.10, 2.28]
    _math(ax, 0.95, ys[0], r"$q_\phi(u\mid r,z)=\mathcal{N}(\mu_\phi(r,z),\mathrm{diag}(\sigma_\phi^2(r,z)))$", 18)
    _math(ax, 0.95, ys[1], r"$u=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)$", 20)
    _math(ax, 0.95, ys[2], r"$\hat r=D_\theta(u,z)$", 20)
    _math(ax, 0.95, ys[3], r"$\mathcal{L}=\mathbb{E}\left[\|r-\hat r\|_2^2\right]+\beta\,D_{\mathrm{KL}}\left(q_\phi(u\mid r,z)\,\|\,p(u)\right)$", 18)
    _math(ax, 0.95, ys[4], r"$D_{\mathrm{KL}}=-\frac{1}{2}\sum\left(1+\log\sigma^2-\mu^2-\sigma^2\right)$", 17)
    _box(ax, 9.10, 4.85, 3.25, 1.00, "latent_dim = 4", TEAL, TEAL_LIGHT, 17)
    _box(ax, 9.10, 3.55, 3.25, 1.00, "hidden_dim = 64", TEAL, TEAL_LIGHT, 17)
    _box(ax, 9.10, 2.25, 3.25, 1.00, "beta = 0.1", ORANGE, ORANGE_LIGHT, 17)
    ax.text(9.10, 1.70, "beta 小：更重视重建；beta 大：更强约束隐变量。", fontsize=10.5, color=MUTED, va="top")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def card_sampling(path: Path) -> None:
    fig, ax = _canvas(
        "Step 4｜采样与反变换：回到响应曲线",
        "采样隐变量 → 解码残差 → 加条件均值 → 反 PCA → 截断到真实范围",
    )
    ys = [5.35, 4.65, 3.95, 3.25, 2.55]
    _math(ax, 0.95, ys[0], r"$u\sim\mathcal{N}(0,I)$", 20)
    _math(ax, 0.95, ys[1], r"$\hat r_g=D_\theta(u,z)$", 20)
    _math(ax, 0.95, ys[2], r"$\hat c_{z,g}=B_g^{T}\tilde z+\hat r_g$", 20)
    _math(ax, 0.95, ys[3], r"$\hat c_g=s_g\odot\hat c_{z,g}$", 20)
    _math(ax, 0.95, ys[4], r"$\hat x_g=\bar x_g+V_{g,K}\hat c_g,\qquad \hat x\leftarrow\mathrm{clip}(\hat x,x_{\min},x_{\max})$", 17)
    _box(ax, 8.90, 4.60, 3.45, 1.05, "可选：真实测量噪声", ORANGE, ORANGE_LIGHT, 15)
    ax.text(9.05, 4.20, "训练时额外加 0.1× 噪声视图；\n验证/测试不加合成数据。", fontsize=10.5, color=MUTED, va="top")
    ax.text(0.95, 1.65, "关键：VAE 输出的是标准化残差，要乘回 coefficient_std，再做反 PCA。", fontsize=11.5, color=MUTED)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def card_training(path: Path) -> None:
    fig, ax = _canvas(
        "训练协议｜只在训练折内生成",
        "5 折 OOF；每折拟合一次 PCA/条件均值/VAE；验证与测试保持真实",
    )
    y = 4.55
    _box(ax, 0.95, y, 2.05, 1.05, "训练折\n40 条真实曲线", TEAL, TEAL_LIGHT, 14)
    _box(ax, 3.70, y, 2.35, 1.05, "每折拟合\nPCA + 均值 + VAE", TEAL, TEAL_LIGHT, 13)
    _box(ax, 6.80, y, 2.10, 1.05, "采样 120 条\n+ 噪声 40 条", ORANGE, ORANGE_LIGHT, 13)
    _box(ax, 9.65, y, 2.15, 1.05, "拼接 200 条\n训练 TCN", "#5A6670", GRAY_LIGHT, 13)
    for x0, x1 in ((3.00, 3.70), (6.05, 6.80), (8.90, 9.65)):
        _arrow(ax, (x0, y + 0.52), (x1, y + 0.52))
    ax.text(1.97, 3.95, "只加训练折", ha="center", fontsize=11, color=TEAL)
    ax.text(5.88, 3.95, "3× 合成 + 1× 噪声", ha="center", fontsize=11, color=ORANGE)
    _math(ax, 0.95, 2.80, r"$\mathrm{OOF}: 5\ \mathrm{folds}\times(40\ \mathrm{train}+10\ \mathrm{valid})$", 17)
    _math(ax, 0.95, 2.10, r"$\mathrm{Final}: 50\ \mathrm{dev}\rightarrow3\ \mathrm{seeds}\rightarrow\mathrm{test}\ 12$", 17)
    ax.text(0.95, 1.35, "测试集只在方案冻结后评估，不参与 K、epoch、增强比例选择。", fontsize=11.5, color=MUTED)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 PCA-VAE 数学卡片")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pca_vae_ppt_images"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, function in (
        ("01_overview", card_overview),
        ("02_pca", card_pca),
        ("03_conditional_mean", card_condition),
        ("04_vae", card_vae),
        ("05_sampling", card_sampling),
        ("06_training", card_training),
    ):
        path = args.output_dir / name
        function(path)
        print(f"Wrote {path}.png")


if __name__ == "__main__":
    main()
