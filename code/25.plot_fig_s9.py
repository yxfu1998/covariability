"""TCWV vs TMT / TCWV vs TLT, 1988–2024: RSS TCWV; two rows with separate axis labels, shared colorbar."""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from supply_plot import (
    find_files_with_variable,
    exclude_era5_merra2,
    exclude_cdr_files,
    filter_rss_filename,
    _prepare_pair_datasets,
    _draw_pair_panel,
    get_axis_limits,
    axis_label,
    _panel_label_char,
    save_figure_png_pdf,
)

COMMON_START = 198801
COMMON_END = 202412


def draw_tcwv_tmt_tlt_figure(cwv_files, tmt_files, tlt_files, output_path):
    """Row 1: TCWV vs TMT; row 2: TCWV vs TLT; separate x/y titles per row, shared colorbar."""
    rows_spec = [
        ("tmt", tmt_files, "cwv", "tmt"),
        ("tlt", tlt_files, "cwv", "tlt"),
    ]

    row_data = []
    for _tag, x_files, var_y, var_x in rows_spec:
        y_ds, x_ds = _prepare_pair_datasets(
            cwv_files, x_files, var_y, var_x, COMMON_START, COMMON_END,
        )
        if not y_ds or not x_ds:
            print(f"No valid data found: {var_y} vs {var_x}")
            return
        x_min, x_max, y_min, y_max = get_axis_limits(var_x, var_y)
        row_data.append({
            "var_y": var_y,
            "var_x": var_x,
            "y_datasets": y_ds,
            "x_datasets": x_ds,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "bottom_title": axis_label(var_x),
            "left_title": axis_label(var_y),
        })

    n_panel_rows = len(row_data)
    n_cols = max(len(r["x_datasets"]) for r in row_data)

    label_ratio = 0.07
    cbar_ratio = 0.09
    hspace_panels = 0.12

    fig_w = 4 * n_cols
    fig_h = 4 * n_panel_rows + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")

    # Panel row + x-axis title row per row, then colorbar row
    height_ratios = []
    for _ in range(n_panel_rows):
        height_ratios.append(1)
        height_ratios.append(label_ratio)
    height_ratios.append(cbar_ratio)
    n_gs_rows = len(height_ratios)

    gs = GridSpec(
        n_gs_rows, n_cols, figure=fig,
        height_ratios=height_ratios, hspace=hspace_panels,
    )

    last_c = None
    panel_idx = 0
    gs_row = 0

    for row_i, rd in enumerate(row_data):
        y_datasets = rd["y_datasets"]
        x_datasets = rd["x_datasets"]
        n_x = len(x_datasets)

        axes = []
        for j in range(n_cols):
            ax = fig.add_subplot(gs[gs_row, j])
            if j < n_x:
                y_name, y_time, y_data = y_datasets[0]
                x_name, x_time, x_data = x_datasets[j]
                panel_char = _panel_label_char(panel_idx)
                panel_idx += 1
                c = _draw_pair_panel(
                    ax, y_name, x_name, y_time, y_data, x_time, x_data,
                    panel_char,
                    rd["x_min"], rd["x_max"], rd["y_min"], rd["y_max"],
                )
                if c is not None:
                    last_c = c
                if j > 0:
                    ax.set_yticklabels([])
            else:
                ax.set_visible(False)
            axes.append(ax)

        gs_row += 1

        # This row x-axis title (not shared with other rows)
        xlab_ax = fig.add_subplot(gs[gs_row, :])
        xlab_ax.set_axis_off()
        xlab_ax.text(
            0.5, 0.5, rd["bottom_title"],
            transform=xlab_ax.transAxes, fontsize=22, fontweight="bold",
            ha="center", va="center",
        )
        gs_row += 1

        # Y-axis title on left of this row (first column only)
        if axes and axes[0].get_visible():
            axes[0].set_ylabel(
                rd["left_title"], fontsize=18, fontweight="bold", labelpad=8,
            )

    cbar_ax = fig.add_subplot(gs[gs_row, :])
    if last_c is not None:
        cbar_ax.set_axis_off()
        inset_cax = inset_axes(
            cbar_ax, width="62%", height="52%", loc="center", borderpad=0,
        )
        cb = fig.colorbar(
            last_c, cax=inset_cax, orientation="horizontal", extend="max",
        )
        cb.set_ticks([0, 5, 10, 15, 20])
        cb.set_label("Count", fontsize=16, fontweight="bold", labelpad=4)
        cb.ax.tick_params(labelsize=16)
        for t in cb.ax.get_xticklabels():
            t.set_fontweight("bold")
    else:
        cbar_ax.set_visible(False)

    save_figure_png_pdf(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close()


def main():
    data_dir = "../data"
    all_y = find_files_with_variable(data_dir, "cwv")
    cwv_files = filter_rss_filename(all_y, "cwv")
    tmt_files = exclude_cdr_files(
        exclude_era5_merra2(find_files_with_variable(data_dir, "tmt"))
    )
    tlt_files = exclude_cdr_files(
        exclude_era5_merra2(find_files_with_variable(data_dir, "tlt"))
    )

    print(f"RSS CWV files（{len(cwv_files)}）:", [os.path.basename(f) for f in cwv_files])
    print(f"TMT x-axis files（{len(tmt_files)}）:", [os.path.basename(f) for f in tmt_files])
    print(f"TLT x-axis files（{len(tlt_files)}）:", [os.path.basename(f) for f in tlt_files])
    print(f"Time period: {COMMON_START} - {COMMON_END}")

    out_path = os.path.join("../plot", "fig_s9.png")
    draw_tcwv_tmt_tlt_figure(cwv_files, tmt_files, tlt_files, out_path)


if __name__ == "__main__":
    main()
