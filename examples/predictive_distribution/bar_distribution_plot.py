"""Plot TabPFN's bar-distribution predictions as a per-sample heatmap.

`plot_bar_distribution` is the entry point: given a `matplotlib` axis, the
1D x-positions of your test samples, the criterion borders, and the raw
logits returned by ``TabPFNRegressor.predict(..., output_type="full")``,
it renders each predicted density as a vertical heatmap column.

This is a visualization helper for examples / notebooks; it has no runtime
dependency on `tabpfn_extensions` itself.
"""

from __future__ import annotations

import seaborn as sns
import torch
from matplotlib import patches
from matplotlib.collections import PatchCollection


def _get_rect(coord, height, width):
    return patches.Rectangle(coord, height, width)


def heatmap_with_box_sizes(
    ax,
    data: torch.Tensor,
    x_starts,
    x_ends,
    y_starts,
    y_ends,
    palette=None,
    set_lims=True,
    threshold_i=0.0,
    y_min=None,
    y_max=None,
    transpose=False,
    per_col_normalize=False,
):
    """Draw a heatmap where each cell has its own (x, y) extent.

    All x/y arrays must be sorted ascending; small indices map to lower
    axis values. `data` has shape (n_rows, n_cols) and is normalized to
    [0, 1] before being mapped through `palette`.
    """
    if palette is None:
        palette = sns.cubehelix_palette(
            start=2.9,
            rot=0.0,
            dark=0.6,
            light=1,
            gamma=4.0,
            hue=9.0,
            as_cmap=True,
        )

    if set_lims:
        ax.set_xlim(x_starts[0], x_ends[-1])
        if y_min is None or y_max is None:
            assert len(y_starts.shape) == 1, (
                "If y_min and y_max are not provided, y_starts should be 1D. "
                "Please set y_min and y_max manually."
            )
            ax.set_ylim(y_starts[0], y_ends[-1])
        else:
            ax.set_ylim(y_min, y_max)

    if per_col_normalize:
        data = (data - data.min(0, keepdim=True).values) / (
            data.max(0, keepdim=True).values - data.min(0, keepdim=True).values
        )
    else:
        data = (data - data.min()) / (data.max() - data.min())

    rects, colors = [], []

    assert y_ends.shape == y_starts.shape
    if len(y_starts.shape) == 1:
        y_starts = y_starts.unsqueeze(0).expand(len(x_starts), -1)
        y_ends = y_ends.unsqueeze(0).expand(len(x_starts), -1)

    for col_i, (col_start, col_end) in enumerate(zip(x_starts, x_ends, strict=False)):
        for row_i, (row_start, row_end) in enumerate(
            zip(y_starts[col_i], y_ends[col_i], strict=False)
        ):
            intensity = data[row_i, col_i].item()
            intensity = max(0.0, (intensity - threshold_i)) / (1 - threshold_i)

            if intensity <= 0:
                continue
            if y_max is not None and y_min is not None and (row_start > y_max or row_end < y_min):
                continue
            if row_start >= row_end or col_start >= col_end:
                continue
            if palette(intensity) == (1.0, 1.0, 1.0, 1.0):
                continue

            if transpose:
                rects.append(
                    _get_rect(
                        (row_start, col_start),
                        row_end - row_start,
                        col_end - col_start,
                    )
                )
            else:
                rects.append(
                    _get_rect(
                        (col_start, row_start),
                        col_end - col_start,
                        row_end - row_start,
                    )
                )
            colors.append(palette(intensity))

    rect_collection = PatchCollection(
        rects, facecolors=colors, edgecolor="none", linewidth=1
    )
    ax.add_collection(rect_collection)
    ax.set_rasterized(True)


def plot_bar_distribution(
    ax,
    x: torch.Tensor,
    bar_borders: torch.Tensor,
    logits: torch.Tensor,
    merge_bars=None,
    restrict_to_range=None,
    plot_log_probs=False,
    **kwargs,
):
    """Plot TabPFN's per-sample bar distribution as a vertical heatmap column.

    Args:
        ax: A matplotlib axis (``fig, ax = plt.subplots()``).
        x: 1D positions of shape ``(num_examples,)`` to place along the x-axis.
        bar_borders: Borders of the bar distribution, taken from
            ``preds["criterion"].borders`` when calling
            ``TabPFNRegressor.predict(..., output_type="full")``.
        logits: Raw logits of shape ``(num_examples, len(bar_borders) - 1)``
            from ``preds["logits"]``.
        merge_bars: If set, merge this many adjacent bars into one for a
            faster, coarser plot.
        restrict_to_range: ``(min_y, max_y)`` to crop the y-axis to a range
            of target values.
        plot_log_probs: If True, plot log-densities (useful when a few bars
            dominate).
        **kwargs: Forwarded to :func:`heatmap_with_box_sizes` (e.g. ``palette``,
            ``threshold_i``).
    """
    x = x.flatten()
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    predictions = logits.softmax(-1)
    assert len(x.shape) == 1
    assert len(predictions.shape) == 2
    assert len(predictions) == len(x)
    assert len(bar_borders.shape) == 1
    assert len(bar_borders) - 1 == predictions.shape[1]
    assert isinstance(x, torch.Tensor)

    if merge_bars and merge_bars > 1:
        new_borders_inds = torch.arange(0, len(bar_borders), merge_bars)
        if new_borders_inds[-1] != len(bar_borders) - 1:
            new_borders_inds = torch.cat(
                [new_borders_inds, torch.tensor([len(bar_borders) - 1])]
            )
        bar_borders = bar_borders[new_borders_inds]
        pred_cumsum = torch.cat(
            [torch.zeros(len(predictions), 1), predictions.cumsum(-1)], dim=-1
        )
        predictions = (
            pred_cumsum[:, new_borders_inds[1:]] - pred_cumsum[:, new_borders_inds[:-1]]
        )
        assert len(bar_borders) - 1 == predictions.shape[-1]

    if restrict_to_range is not None:
        min_y, max_y = restrict_to_range
        border_mask = (min_y <= bar_borders) & (bar_borders <= max_y)
        border_mask[:-1] = border_mask[1:] | border_mask[:-1]
        border_mask[1:] = border_mask[1:] | border_mask[:-1]
        logit_mask = border_mask[:-1] & border_mask[1:]
        bar_borders = bar_borders[border_mask]
        predictions = predictions[:, logit_mask]

    y_starts = bar_borders[:-1]
    y_ends = bar_borders[1:]

    x, order = x.sort(0)

    # Convert probability mass to probability density (per unit of y).
    predictions = predictions[order] / (bar_borders[1:] - bar_borders[:-1])
    predictions[torch.isinf(predictions)] = 0.0
    predictions[:, (bar_borders[1:] - bar_borders[:-1]) < 1e-10] = 0.0

    if plot_log_probs:
        predictions = predictions.log()
        predictions[predictions.isinf()] = torch.min(predictions[~predictions.isinf()])

    # x positions are widened to the midpoint between neighbours.
    x_starts = torch.cat([x[0].unsqueeze(0), (x[1:] + x[:-1]) / 2])
    x_ends = torch.cat([(x[1:] + x[:-1]) / 2, x[-1].unsqueeze(0)])

    heatmap_with_box_sizes(
        ax, predictions.T, x_starts, x_ends, y_starts, y_ends, **kwargs
    )
