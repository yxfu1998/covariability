import os
import importlib.util
import numpy as np
import matplotlib

# non-GUI backend for server/WSL
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supply_plot import save_figure_png_pdf


def _load_plot_fig1():
    """Load data helpers from 13.plot_fig1.py in the same directory."""
    fig1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "13.plot_fig1.py")
    if not os.path.isfile(fig1_path):
        raise FileNotFoundError(f"13.plot_fig1.py not found: {fig1_path}")
    spec = importlib.util.spec_from_file_location("plot_fig1", fig1_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_dataset_label(filename: str) -> str:
    """Label logic consistent with plot_anomaly_time_series in 13.plot_fig1.py."""
    fname = os.path.basename(filename).lower()
    if "ustc_" in fname:
        return "USTC"
    if "star_" in fname:
        return "STAR"
    if "rss_" in fname:
        return "RSS"
    if "uah_" in fname:
        return "UAH"
    if "cmsaf_" in fname:
        return "CMSAF"
    if "merra2_" in fname:
        return "MERRA2"
    if "era5_" in fname:
        return "ERA5"
    if "cobe_" in fname:
        return "COBE"
    if "ersst_" in fname:
        return "ERSST"
    if "hadisst_" in fname:
        return "HADISST"
    if "hadsst_" in fname:
        return "HADSST"
    if "oisst_" in fname:
        return "OISST"
    return filename.split("_")[0].upper()


def main():
    """
    Plot USTC TCWV anomaly minus other TCWV anomalies and trends.
    Data read and anomaly computation use 13.plot_fig1.py  functions.
    """

    try:
        fig1 = _load_plot_fig1()
    except Exception as e:
        print(f"Cannot load 13.plot_fig1.py: {e}")
        return

    for name in ["find_data_files", "load_data", "calculate_monthly_anomaly", "cal_trend"]:
        if not hasattr(fig1, name):
            print(f"13.plot_fig1.py missing function {name}; check file contents.")
            return

    find_data_files = fig1.find_data_files
    load_data = fig1.load_data
    calculate_monthly_anomaly = fig1.calculate_monthly_anomaly
    cal_trend = fig1.cal_trend

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
    if not os.path.isdir(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return

    data_files = find_data_files(data_dir)
    if not data_files:
        print(f"In directory {data_dir} no data files found")
        return

    # --- Read each dataset TCWV anomaly(percent), 1980-2024 ---
    tcwv_dict = {}

    for filepath in data_files:
        time, variables = load_data(filepath)
        if time is None or variables is None:
            continue
        if "cwv" not in variables:
            continue

        # Same full period as 13.plot_fig1
        mask = (time >= 198001) & (time <= 202412)
        if np.sum(mask) == 0:
            continue

        time_sel = time[mask]
        cwv_sel = variables["cwv"][mask]

        # use 13.plot_fig1 in  calculate_monthly_anomaly algorithm
        anomaly = calculate_monthly_anomaly(cwv_sel)

        # Same as 13.plot_fig1: CWV anomaly to percent, ~41 mm reference
        anomaly_pct = anomaly * 100.0 / 41.0

        label = get_dataset_label(filepath)
        tcwv_dict[label] = (time_sel.astype(int), anomaly_pct.astype(float))

    if "USTC" not in tcwv_dict:
        print("USTC TCWV not found; cannot compute differences.")
        return

    ref_time, ref_anom = tcwv_dict["USTC"]

    # Period 2002-06 to 2024-12 only (same as draw_trend_difference.py)
    start_yyyymm = 200206
    end_yyyymm = 202412

    ref_mask = (ref_time >= start_yyyymm) & (ref_time <= end_yyyymm)
    ref_time_sub = ref_time[ref_mask]
    ref_anom_sub = ref_anom[ref_mask]

    if len(ref_time_sub) == 0:
        print("USTC TCWV No data in 2002-06 to 2024-12.")
        return

    # Map time_yyyymm -> anomaly
    def build_anom_map(time_arr, anom_arr):
        return {int(t): float(v) for t, v in zip(time_arr, anom_arr) if not np.isnan(v)}

    # Helper for correlation(drop NaN)
    def calc_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        if np.sum(mask) < 2:
            return np.nan
        cc = np.corrcoef(a[mask], b[mask])
        return float(cc[0, 1])

    ref_map = build_anom_map(ref_time_sub, ref_anom_sub)

    # Other datasets (exclude USTC)
    other_labels = [lab for lab in sorted(tcwv_dict.keys()) if lab != "USTC"]
    if not other_labels:
        print("No other TCWV datasets besides USTC.")
        return

    diff_results = []  # each entry: dict(label, years, diff_series, trend, error, std, …)

    for lab in other_labels:
        time_o, anom_o = tcwv_dict[lab]
        mask_o = (time_o >= start_yyyymm) & (time_o <= end_yyyymm)
        time_o_sub = time_o[mask_o]
        anom_o_sub = anom_o[mask_o]
        if len(time_o_sub) == 0:
            continue

        other_map = build_anom_map(time_o_sub, anom_o_sub)

        # common time steps
        common_times = sorted(set(ref_map.keys()) & set(other_map.keys()))
        if len(common_times) < 24:  # too short for trend
            continue

        # aligned anomaly series
        ref_vals = np.array([ref_map[t] for t in common_times], dtype=float)
        oth_vals = np.array([other_map[t] for t in common_times], dtype=float)

        # CC between the two datasets(200206-202412)
        corr_val = calc_corr(ref_vals, oth_vals)

        # Align using 2003-2007 mean difference
        align_start = 200301
        align_end = 200712
        align_mask = np.array([(t >= align_start and t <= align_end) for t in common_times])
        if np.sum(align_mask) > 0:
            d_align = np.nanmean(ref_vals[align_mask] - oth_vals[align_mask])
        else:
            # if no2003-2007 data,use first 24 months
            first_n = min(24, len(common_times))
            d_align = np.nanmean(ref_vals[:first_n] - oth_vals[:first_n])
        diff_vals = ref_vals - oth_vals - d_align

        # Convert to decimal years (same as other figures: months as (month-0.5)/12)
        years = np.array(
            [t // 100 + (t % 100 - 0.5) / 12.0 for t in common_times], dtype=float
        )

        # Trend in %/decade via cal_trend
        yy = diff_vals.copy()
        missing = -9999.0
        yy_filled = np.where(np.isnan(yy), missing, yy)

        N = len(years)
        trd, err, sigma, amp = cal_trend(years, yy_filled, N, 0, N - 1)
        std_val = float(np.nanstd(diff_vals))

        diff_results.append(
            {
                "label": lab,
                "years": years,
                "diff": diff_vals,
                "trend": trd,
                "error": err,
                "std": std_val,
                "corr": corr_val,
                "panel_type": "ustc",
                "panel_title": f"USTC - {lab}",
                "period_stats": [
                    {
                        "label1": "USTC",
                        "label2": lab,
                        "start": start_yyyymm,
                        "end": end_yyyymm,
                        "corr": corr_val,
                        "trend": trd,
                        "error": err,
                        "std": std_val,
                    }
                ],
            }
        )


    def compute_pair_metrics(label1, label2, start_yyyymm_seg, end_yyyymm_seg):
        """For two datasets and period: CC, bias (difference) trend, and std."""
        if (label1 not in tcwv_dict) or (label2 not in tcwv_dict):
            return None
        time1, anom1 = tcwv_dict[label1]
        time2, anom2 = tcwv_dict[label2]

        # Common times within segment
        common_times_all = sorted(set(time1) & set(time2))
        common_times = [t for t in common_times_all if start_yyyymm_seg <= t <= end_yyyymm_seg]
        if len(common_times) < 24:
            return None

        a1 = np.array(
            [anom1[np.where(time1 == t)[0][0]] for t in common_times],
            dtype=float,
        )
        a2 = np.array(
            [anom2[np.where(time2 == t)[0][0]] for t in common_times],
            dtype=float,
        )

        corr_val = calc_corr(a1, a2)

        # Align using 2003-2007 mean difference
        align_start = 200301
        align_end = 200712
        align_mask = np.array([(t >= align_start and t <= align_end) for t in common_times])
        if np.sum(align_mask) > 0:
            d_align = np.nanmean(a1[align_mask] - a2[align_mask])
        else:
            # if no2003-2007 data,use first 24 months
            first_n = min(24, len(common_times))
            d_align = np.nanmean(a1[:first_n] - a2[:first_n])
        diff_vals = a1 - a2 - d_align

        years = np.array(
            [t // 100 + (t % 100 - 0.5) / 12.0 for t in common_times],
            dtype=float,
        )

        yy = diff_vals.copy()
        missing = -9999.0
        yy_filled = np.where(np.isnan(yy), missing, yy)

        N = len(years)
        trd, err, sigma, amp = cal_trend(years, yy_filled, N, 0, N - 1)
        std_val = float(np.nanstd(diff_vals))

        return {
            "years": years,
            "diff": diff_vals,
            "corr": corr_val,
            "trend": trd,
            "error": err,
            "std": std_val,
            "start": start_yyyymm_seg,
            "end": end_yyyymm_seg,
        }

    def add_pair_panel(label1, label2, segs):
        """Add subplot for two datasets; segs lists sub-periods for multi-segment stats."""
        pair_name = f"{label1} - {label2}"

        # Plot period: union of segment overlaps:union of segment common times
        time1, _a1 = tcwv_dict.get(label1, (None, None))
        time2, _a2 = tcwv_dict.get(label2, (None, None))
        if time1 is None or time2 is None:
            return
        common_times_all = sorted(set(time1) & set(time2))
        if not common_times_all:
            return

        # Plot from earliest to latest common time
        plot_start = common_times_all[0]
        plot_end = common_times_all[-1]
        base_metrics = compute_pair_metrics(label1, label2, plot_start, plot_end)
        if base_metrics is None:
            return

        # Stats for three periods (draw_supply2 style)
        period_stats = []
        seen = set()  # (start, end) avoid duplicates
        for (s, e) in segs:
            m = compute_pair_metrics(label1, label2, s, e)
            if m is None:
                continue
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            period_stats.append(
                {
                    "label1": label1,
                    "label2": label2,
                    "start": s,
                    "end": e,
                    "corr": m["corr"],
                    "trend": m["trend"],
                    "error": m["error"],
                    "std": m["std"],
                }
            )

        # Backfill early segment stats for right column if missing
        has_early_2001 = any(ps["end"] == 200112 for ps in period_stats)
        if not has_early_2001:
            for start_early, end_early in [(198001, 200112), (198801, 200112)]:
                if (start_early, end_early) in seen:
                    continue
                m = compute_pair_metrics(label1, label2, start_early, end_early)
                if m is not None:
                    period_stats.append({
                        "label1": label1,
                        "label2": label2,
                        "start": start_early,
                        "end": end_early,
                        "corr": m["corr"],
                        "trend": m["trend"],
                        "error": m["error"],
                        "std": m["std"],
                    })
                    break

        if not period_stats:
            return

        diff_results.append(
            {
                "label": pair_name,
                "years": base_metrics["years"],
                "diff": base_metrics["diff"],
                "trend": base_metrics["trend"],
                "error": base_metrics["error"],
                "std": base_metrics["std"],
                "corr": base_metrics["corr"],
                "panel_type": "pair",
                "panel_title": pair_name,
                "period_stats": period_stats,
            }
        )

    # RSS and ERA5、MERRA2:198801-202412、198801-200112、200206-202412
    add_pair_panel(
        "RSS",
        "ERA5",
        segs=[(198801, 202412), (198801, 200112), (200206, 202412)],
    )
    add_pair_panel(
        "RSS",
        "MERRA2",
        segs=[(198801, 202412), (198801, 200112), (200206, 202412)],
    )

    add_pair_panel(
        "ERA5",
        "MERRA2",
        segs=[(198001, 202412), (198001, 200112), (200206, 202412)],
    )

    if not diff_results:
        print("No valid difference/trend for any dataset pair.")
        return

    ustc_results = [d for d in diff_results if d.get("panel_type", "ustc") == "ustc"]
    pair_results = [d for d in diff_results if d.get("panel_type", "") == "pair"]
    
    # Left: USTC panels; right: last 3 pair panels
    n_left = len(ustc_results)
    n_right = len(pair_results)
    n_rows = max(n_left, n_right)

    # Unified y and x limits(spanallpanel,maximum overlap)
    all_diff = np.concatenate([d["diff"] for d in diff_results])
    all_years = np.concatenate([d["years"] for d in diff_results])
    ymin = float(np.nanmin(all_diff)) - 2
    ymax = float(np.nanmax(all_diff)) + 1

    xmin = float(np.nanmin(all_years))
    xmax = float(np.nanmax(all_years))

    base_colors = ["#04599B", "#FF3F3F", "#5506CA", "#008B45", "#FF8C00", "#8B008B"]

    # sharex='col' shares x within column; left/right columns may differ
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=2,
        sharex='col',
        sharey=True,
        figsize=(12, 1.8 * n_rows + 1.0),
    )
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    for i, item in enumerate(ustc_results):
        ax = axs[i, 0]
        years = item["years"]
        diff_vals = item["diff"]
        label_panel = item.get("label", "")
        trd = item["trend"]
        err = item["error"]
        std_val = item["std"]
        corr_val = item.get("corr", np.nan)
        panel_type = item.get("panel_type", "ustc")
        panel_title = item.get("panel_title", label_panel)
        period_stats = item.get("period_stats", [])

        color = base_colors[i % len(base_colors)]

        ax.plot([xmin, xmax], [0.0, 0.0], linestyle="--", color="black", linewidth=1.0)

        # timeseries(no label,no legend)
        ax.plot(
            years,
            diff_vals,
            color=color,
            linewidth=1.0,
        )

        ax.plot(
            [2002.0 + 6.0 / 12.0, 2002.0 + 6.0 / 12.0],
            [ymin - 5.0, ymax + 5.0],
            color="black",
            linestyle="--",
        )

        ax.set_ylim(ymin-3., ymax + 0.5)
        # Left column x range 2002-2025
        ax.set_xlim(2002.0, 2025.0)
        ax.set_yticks([-2, 0, 2])

        ax.set_xticks([2003, 2008, 2013, 2018, 2023])

        # Panel labels a, b, c... left-to-right, top-to-bottom,white background box
        panel_char = chr(97 + i * 2 + 0)  # leftcol:i*2+0
        title_text = f"({panel_char}) {panel_title}"
        ax.text(
            0.02,
            0.86,
            title_text,
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1)
        )

        if panel_type == "ustc":
            # Period 2002-2024; label CC not r
            info_text = f"2002-2024: CC={corr_val:.3f}, Trend={trd:.3f}±{err:.3f} % Dec$^-$$^1$, STD={std_val:.3f} %"
            ax.text(
                0.1,
                0.1,
                info_text,
                transform=ax.transAxes,
                fontsize=8,
                color="black",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=1)
            )

        ax.tick_params(labelsize=9, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        if i == n_left - 1:
            xlabel = ax.set_xlabel("Year", fontsize=10)
            xlabel.set_fontweight('bold')

    # Plot right column: pair panels(RSS-ERA5、RSS-MERRA2、ERA5-MERRA2)
    for i, item in enumerate(pair_results):
        ax = axs[i, 1]
        years = item["years"]
        diff_vals = item["diff"]
        panel_title = item.get("panel_title", "")
        trd = item["trend"]
        err = item["error"]
        std_val = item["std"]
        period_stats = item.get("period_stats", [])

        color = base_colors[(n_left + i) % len(base_colors)]

        ax.plot([xmin, xmax], [0.0, 0.0], linestyle="--", color="black", linewidth=1.0)

        # timeseries(no label,no legend)
        ax.plot(
            years,
            diff_vals,
            color=color,
            linewidth=1.0,
        )

        ax.plot(
            [2002.0 + 6.0 / 12.0, 2002.0 + 6.0 / 12.0],
            [ymin - 5.0, ymax + 5.0],
            color="black",
            linestyle="--",
        )

        ax.set_ylim(ymin-3, ymax + 0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_yticks([-4, -2, 0, 2])

        ax.set_xticks([1980, 1990, 2000, 2010, 2020])

        # Panel labels a, b, c... left-to-right, top-to-bottom,white background box
        panel_char = chr(97 + i * 2 + 1)  # rightcol:i*2+1
        title_text = f"({panel_char}) {panel_title}"
        ax.text(
            0.02,
            0.86,
            title_text,
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=1)
        )

        ax.tick_params(labelsize=9, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Fixed display order:① 1980/1988-2024  ② 1980/1988-2001  ③ 2002-2024,ensure 1980 or 1988-2001 always shown
        if period_stats:
            def _display_order(ps):
                s, e = ps["start"], ps["end"]
                if e == 202412 and s in (198001, 198801):
                    return 0  # full period
                if e == 200112:  # 1980-2001 or 1988-2001
                    return 1
                if s == 200206:
                    return 2  # 2002-2024
                return 3

            sorted_stats = sorted(period_stats, key=_display_order)
            lines = []
            for j, ps in enumerate(sorted_stats):
                if j >= 3:
                    break
                s = ps["start"]
                e = ps["end"]
                txt = (
                    f"{int(s/100)}-{int(e/100)}:"
                    f" CC={ps['corr']:.3f},"
                    f" Trend={ps['trend']:.3f}±{ps['error']:.3f} % Dec$^-$$^1$,"
                    f" STD={ps['std']:.3f} %"
                )
                lines.append(txt)

            if lines:
                info_text = "\n".join(lines)
                ax.text(
                    0.1,
                    0.1,
                    info_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="black",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=1)
                )

        if i == n_right - 1:
            xlabel = ax.set_xlabel("Year", fontsize=10)
            xlabel.set_fontweight('bold')

    # Hide unused right-column panels(if left column has more rows than right)
    for i in range(n_right, n_rows):
        axs[i, 1].set_visible(False)

    fig.text(0.04, 0.5, "TCWV Anomaly Difference (%)", va="center", rotation="vertical", fontsize=10, weight='bold')

    # Layout: more left margin,more left margin for left column,Reduce gap between left and right columns
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    # use subplots_adjust Reduce gap between left and right columns
    plt.subplots_adjust(wspace=0.1)
    out_name = "../plot/fig_s1.png"
    save_figure_png_pdf(out_name, dpi=600, fig=fig)


if __name__ == "__main__":
    main()

