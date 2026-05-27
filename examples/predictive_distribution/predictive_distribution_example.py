"""TabPFN bar-distribution: visualize the full predictive density.

TabPFN regressors do not predict a single value. Internally they predict
a discrete *bar distribution* over the target axis (one probability per
bucket). That distribution is the model's full belief about y given x and
the in-context examples; point estimates (mean / median / mode) and
quantiles are all derived from it.

This example shows how to:

  1. Get the raw distribution via ``predict(X, output_type="full")``.
  2. Visualize it as a per-sample heatmap with ``plot_bar_distribution``.
  3. Showcase the different views ``plot_bar_distribution`` can produce.
  4. Recover point estimates and confidence intervals from the distribution.

The toy dataset evolves over x in three regimes (tight unimodal, then
growing variance, then bimodal), so the predictive distribution has
interesting structure throughout.

NOTE: This example requires the full TabPFN package (``pip install tabpfn``).
The cloud client does not return logits, so ``output_type="full"`` is
unavailable there.

CLI usage::

    # Full demo (default):
    python predictive_distribution_example.py

    # Tiny dataset for quick testing (no window pop-up, figures saved to /tmp):
    python predictive_distribution_example.py \\
        --n-train 60 --n-test 30 --n-estimators 1 --no-show --out-dir /tmp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from bar_distribution_plot import plot_bar_distribution

from tabpfn_extensions import TabPFNRegressor

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def main(
    n_train: int = 600,
    n_test: int = 240,
    n_estimators: int = 4,
    no_show: bool = False,
    out_dir: Path | None = None,
) -> None:
    """Run the bar-distribution example end-to-end.

    Args:
        n_train: Number of training samples to generate.
        n_test: Number of test-grid points.
        n_estimators: Number of TabPFN estimators (lower → faster).
        no_show: If True, skip ``plt.show()`` (useful for CI / headless runs).
        out_dir: Directory for saved PNGs.  Defaults to the script's directory.
    """
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR

    if no_show:
        plt.switch_backend("Agg")

    # Bigger fonts everywhere so the saved PNGs stay readable when embedded
    # in the README.
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 19,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 21,
        }
    )

    # ─────────────────────────────── toy dataset ────────────────────────────────
    # Three regimes along x (think of x as time):
    #   x in [0, 4]:   tight unimodal noise
    #   x in [4, 7]:   variance grows linearly (heteroscedastic)
    #   x in [7, 10]:  two equally likely modes (bimodal posterior)
    # This lets one heatmap show all three failure modes of a single point
    # estimate.

    rng = np.random.default_rng(0)

    x_train = rng.uniform(0.0, 10.0, size=n_train).astype(np.float32)
    y_clean = np.sin(0.6 * x_train)

    bimodal_mask = x_train > 7.0
    # Grows from 0.10 at x=4 to ~0.85 at x=7, then drops back to 0.20 inside
    # each bimodal mode so the two bands stay visually separated.
    sigma_heteroscedastic = np.clip(
        0.10 + np.maximum(0.0, x_train - 4.0) * 0.25, 0.10, 0.85
    ).astype(np.float32)
    sigma = np.where(bimodal_mask, 0.20, sigma_heteroscedastic).astype(np.float32)

    # Vary the upper-mode probability smoothly along x so the median flips
    # between modes instead of locking onto one. Outside the bimodal region
    # this has no effect.
    upper_prob = 0.5 + 0.42 * np.sin(2 * np.pi * (x_train - 7.0) / 1.2)
    upper_pick = rng.random(n_train) < upper_prob
    mode_offset = np.where(upper_pick, 1.5, -1.5).astype(np.float32)
    mode_offset = np.where(bimodal_mask, mode_offset, 0.0).astype(np.float32)

    y_train = y_clean + mode_offset + rng.normal(0.0, sigma).astype(np.float32)

    X_train = x_train.reshape(-1, 1)

    # Test grid covers the full input range, sorted for nicer plotting.
    X_test = np.linspace(0.0, 10.0, n_test, dtype=np.float32).reshape(-1, 1)

    # ─────────────────────────── fit & full prediction ──────────────────────────

    reg = TabPFNRegressor(n_estimators=n_estimators, random_state=0)
    reg.fit(X_train, y_train)

    # output_type="full" returns a dict containing:
    #   "mean", "median", "mode": point estimates over the bar distribution
    #   "quantiles": list of arrays, one per requested quantile
    #   "criterion": the BarDistribution object (.borders, .bucket_widths, ...)
    #   "logits": raw (n_samples, n_bars) logits parameterizing the distribution
    preds = reg.predict(X_test, output_type="full")

    # Move the bar distribution and its logits to CPU together so we can do
    # our own derivations and plot with matplotlib.
    criterion = preds["criterion"].to("cpu")
    logits = preds["logits"].detach().cpu()
    borders = criterion.borders

    print(
        f"Predicted distribution: {logits.shape[1]} bars over {len(X_test)} test points"
    )
    print(f"Bar borders span [{borders[0]:.2f}, {borders[-1]:.2f}]")

    # ─────────────────────────── point estimates from the distribution ──────────
    # The convenience entries in ``preds`` are computed from ``logits`` plus
    # ``criterion``. You can reproduce them yourself, which is useful when you
    # want a custom summary statistic (e.g. a trimmed mean or a custom quantile):

    mean_manual = criterion.mean(logits).numpy()
    median_manual = criterion.median(logits).numpy()

    # These match the dict entries up to float32 precision. The dict values are
    # computed by exactly the same calls under the hood.
    print(
        "mean   match: max |manual - preds['mean']|   = "
        f"{np.max(np.abs(mean_manual - preds['mean'])):.2e}"
    )
    print(
        "median match: max |manual - preds['median']| = "
        f"{np.max(np.abs(median_manual - preds['median'])):.2e}"
    )

    # A 90% central credible interval, directly from the bar distribution's
    # inverse CDF.
    q05 = criterion.icdf(logits, 0.05).numpy()
    q95 = criterion.icdf(logits, 0.95).numpy()

    # ─────────────────────────────── main figure ────────────────────────────────
    # Full predictive density across x, with point estimates overlaid. The
    # heatmap exposes the three regimes (tight, heteroscedastic, bimodal);
    # the mean and median collapse all of that to a single curve.

    Y_MIN, Y_MAX = -4.0, 4.0
    x_test_1d = torch.tensor(X_test[:, 0])

    fig, ax = plt.subplots(figsize=(15, 7.5))

    plot_bar_distribution(
        ax,
        x_test_1d,
        borders,
        logits,
        restrict_to_range=(Y_MIN, Y_MAX),
    )

    ax.scatter(
        x_train, y_train, s=10, alpha=0.4, color="black", label="train data", zorder=2
    )
    ax.plot(X_test[:, 0], preds["mean"], color="C0", lw=2.2, label="mean", zorder=3)
    ax.plot(
        X_test[:, 0],
        preds["median"],
        color="C1",
        lw=2.2,
        ls="--",
        label="median",
        zorder=3,
    )
    ax.fill_between(
        X_test[:, 0], q05, q95, color="C0", alpha=0.10, label="90% CI", zorder=1
    )

    # Annotate the three regimes so the structure is obvious at a glance.
    for x_edge in (4.0, 7.0):
        ax.axvline(x_edge, color="gray", lw=1, ls=":", alpha=0.7, zorder=0)
    ax.text(2.0, 3.6, "tight unimodal", ha="center", fontsize=15, color="gray")
    ax.text(5.5, 3.6, "growing variance", ha="center", fontsize=15, color="gray")
    ax.text(8.5, 3.6, "bimodal", ha="center", fontsize=15, color="gray")

    ax.set_xlabel("x  (interpret as time)")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_title(
        "TabPFN predictive distribution\n"
        "heatmap = full bar distribution, lines = derived point estimates"
    )
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_dir / "tabpfn_bar_distribution.png", dpi=150)
    if not no_show:
        plt.show()
    plt.close(fig)

    # ───────────────── showcase: plots you can build with plot_bar_distribution ──
    # ``plot_bar_distribution`` is the helper in ``bar_distribution_plot.py``.
    # The two panels below are the views it produces out of the box: the
    # default per-bucket density, and a coarser merged version for faster /
    # cleaner rendering on dense grids.

    fig_showcase, (ax_default, ax_coarse) = plt.subplots(
        1, 2, figsize=(18, 7), sharex=True, sharey=True
    )

    # Default density: linear scale, one column per test point.
    plot_bar_distribution(
        ax_default, x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX)
    )
    ax_default.set_title(
        "Default density\nplot_bar_distribution(ax, x, borders, logits)"
    )

    # Coarse merge: merges adjacent buckets for a faster, lower-resolution render.
    plot_bar_distribution(
        ax_coarse,
        x_test_1d,
        borders,
        logits,
        restrict_to_range=(Y_MIN, Y_MAX),
        merge_bars=4,
    )
    ax_coarse.set_title(
        "Coarse merge (merge_bars=4)\n"
        "merges adjacent buckets for a faster, lower-resolution render"
    )

    for ax_s in (ax_default, ax_coarse):
        ax_s.set_xlim(0.0, 10.0)
        ax_s.set_ylim(Y_MIN, Y_MAX)
        ax_s.set_xlabel("x")
        ax_s.set_ylabel("y")

    fig_showcase.suptitle(
        "Plots you can build with plot_bar_distribution (bar_distribution_plot.py)"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "tabpfn_bar_distribution_variants.png", dpi=150)
    if not no_show:
        plt.show()
    plt.close(fig_showcase)

    # ───────────────────── 1D density slices at a few test points ───────────────
    # Each column of the heatmap above is a 1D density over y. Plotting those
    # slices makes the three regimes obvious side by side: tight at low x,
    # wide and unimodal in the middle, bimodal on the right.

    slice_targets = np.array([1.0, 3.0, 5.5, 7.0, 9.0], dtype=np.float32)
    # Clip slice targets to valid X_test range and deduplicate
    slice_indices = list(
        dict.fromkeys(int(np.argmin(np.abs(X_test[:, 0] - t))) for t in slice_targets)
    )

    # Rebin the bar masses into uniform y-bins. The raw bars are non-uniform
    # (narrow near the bulk, wide in the tails); converting to a uniform grid
    # avoids spikes from very narrow bars and gives a clean histogram look.
    n_coarse = 80
    y_edges = np.linspace(Y_MIN, Y_MAX, n_coarse + 1)
    y_centers_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    y_widths_c = np.full(n_coarse, (Y_MAX - Y_MIN) / n_coarse)

    probs_np = logits.softmax(-1).numpy()
    bar_centers = 0.5 * (borders[1:] + borders[:-1])
    bin_idx = np.digitize(bar_centers.numpy(), y_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_coarse)

    fig2, axes = plt.subplots(
        1, len(slice_indices), figsize=(19, 5.5), sharey=True, squeeze=False
    )
    axes = axes.flatten()
    for ax_i, idx in zip(axes, slice_indices, strict=False):
        mass = np.bincount(
            bin_idx[valid], weights=probs_np[idx, valid], minlength=n_coarse
        )
        density_slice = mass / y_widths_c
        x_val = X_test[idx, 0]
        ax_i.barh(
            y_centers_c,
            density_slice,
            height=y_widths_c,
            color="C0",
            alpha=0.7,
            align="center",
        )
        ax_i.axhline(preds["mean"][idx], color="black", lw=1.8, label="mean")
        ax_i.axhline(preds["median"][idx], color="C1", lw=1.8, ls="--", label="median")
        ax_i.set_title(f"p(y | x = {x_val:.2f})")
        ax_i.set_xlabel("density")
    axes[0].set_ylabel("y")
    axes[0].set_ylim(Y_MIN, Y_MAX)
    axes[-1].legend(loc="upper right")
    fig2.suptitle(
        "Predictive density slices: tight, then heteroscedastic, then bimodal"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "tabpfn_density_slices.png", dpi=150)
    if not no_show:
        plt.show()
    plt.close(fig2)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=600,
        metavar="N",
        help="Number of training samples (default: 600).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=240,
        metavar="N",
        help="Number of test-grid points (default: 240).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=4,
        metavar="N",
        help="Number of TabPFN estimators — lower values run faster (default: 4).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Skip plt.show() — useful for headless / CI runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for saved PNG files (default: same directory as this script).",
    )
    args = parser.parse_args()
    main(
        n_train=args.n_train,
        n_test=args.n_test,
        n_estimators=args.n_estimators,
        no_show=args.no_show,
        out_dir=args.out_dir,
    )
