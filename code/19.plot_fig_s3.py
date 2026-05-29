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
    Plot TMT anomaly differences and trends for all dataset pairs.
    Data read and anomaly computation use 13.plot_fig1.py  functions.
    Layout 4x3: col 1 STAR minus others; other cols remaining combos.
    """

    try:
        fig1 = _load_plot_fig1()
    except Exception as e:
        print(f"Cannotload 13.plot_fig1.py: {e}")
        return

    for name in ["find_data_files", "load_data", "calculate_monthly_anomaly", "cal_trend"]:
        if not hasattr(fig1, name):
            print(f"13.plot_fig1.py missing function {name}，check file contents。")
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

    # --- Read each dataset  TMT anomaly,time 1980–2024 ---
    # Store: label -> (time_yyyymm, anomaly)
    tmt_dict = {}

    for filepath in data_files:
        time, variables = load_data(filepath)
        if time is None or variables is None:
            continue
        if "tmt" not in variables:
            continue

        # Same full period as 13.plot_fig1
        mask = (time >= 198001) & (time <= 202412)
        if np.sum(mask) == 0:
            continue

        time_sel = time[mask]
        tmt_sel = variables["tmt"][mask]

        # use 13.plot_fig1 in  calculate_monthly_anomaly algorithm
        anomaly = calculate_monthly_anomaly(tmt_sel)

        # tmt anomaly no percent conversion; use K
        label = get_dataset_label(filepath)
        tmt_dict[label] = (time_sel.astype(int), anomaly.astype(float))

    # tmtno USTC; use STAR as reference
    if "STAR" not in tmt_dict:
        print("not found STAR   tmt data，Cannot compute differences.")
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

    def compute_pair_metrics(label1, label2, start_yyyymm_seg, end_yyyymm_seg):
        """For two datasets and period: CC, bias (difference) trend, and std."""
        if (label1 not in tmt_dict) or (label2 not in tmt_dict):
            return None
        time1, anom1 = tmt_dict[label1]
        time2, anom2 = tmt_dict[label2]

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

    # --- Build all dataset pairs ---
    all_labels = sorted(tmt_dict.keys())
    print(f"find {len(all_labels)}  TMT datasets: {all_labels}")

    star_results = []
    rss_results = []
    uah_results = []
    
    if "STAR" in all_labels:
        ref_time, ref_anom = tmt_dict["STAR"]
        # From 1980, max overlap period
        start_yyyymm = 198001
        end_yyyymm = 202412
        ref_mask = (ref_time >= start_yyyymm) & (ref_time <= end_yyyymm)
        ref_time_sub = ref_time[ref_mask]
        ref_anom_sub = ref_anom[ref_mask]
        ref_map = build_anom_map(ref_time_sub, ref_anom_sub)

        other_labels = [lab for lab in all_labels if lab != "STAR"]
        for lab in other_labels:
            time_o, anom_o = tmt_dict[lab]
            # use max overlap period
            common_times_all = sorted(set(ref_time_sub) & set(time_o))
            if not common_times_all:
                continue
            
            mask_o = np.isin(time_o, common_times_all)
            time_o_sub = time_o[mask_o]
            anom_o_sub = anom_o[mask_o]
            if len(time_o_sub) == 0:
                continue

            other_map = build_anom_map(time_o_sub, anom_o_sub)
            common_times = sorted(set(ref_map.keys()) & set(other_map.keys()))
            if len(common_times) < 24:
                continue

            ref_vals = np.array([ref_map[t] for t in common_times], dtype=float)
            oth_vals = np.array([other_map[t] for t in common_times], dtype=float)
            corr_val = calc_corr(ref_vals, oth_vals)

            align_start = 200301
            align_end = 200712
            align_mask = np.array([(t >= align_start and t <= align_end) for t in common_times])
            if np.sum(align_mask) > 0:
                d_align = np.nanmean(ref_vals[align_mask] - oth_vals[align_mask])
            else:
                first_n = min(24, len(common_times))
                d_align = np.nanmean(ref_vals[:first_n] - oth_vals[:first_n])
            diff_vals = ref_vals - oth_vals - d_align

            years = np.array(
                [t // 100 + (t % 100 - 0.5) / 12.0 for t in common_times], dtype=float
            )

            yy = diff_vals.copy()
            missing = -9999.0
            yy_filled = np.where(np.isnan(yy), missing, yy)
            N = len(years)
            trd, err, sigma, amp = cal_trend(years, yy_filled, N, 0, N - 1)
            std_val = float(np.nanstd(diff_vals))

            # Compute stats for three periods:1980-2024、1980-2001 and 200206-202412
            period_stats = []
            m1 = compute_pair_metrics("STAR", lab, 198001, 202412)
            if m1 is not None:
                period_stats.append({
                    "start": 198001,
                    "end": 202412,
                    "corr": m1["corr"],
                    "trend": m1["trend"],
                    "error": m1["error"],
                    "std": m1["std"],
                })
            m2 = compute_pair_metrics("STAR", lab, 198001, 200112)
            if m2 is not None:
                period_stats.append({
                    "start": 198001,
                    "end": 200112,
                    "corr": m2["corr"],
                    "trend": m2["trend"],
                    "error": m2["error"],
                    "std": m2["std"],
                })
            m3 = compute_pair_metrics("STAR", lab, 200206, 202412)
            if m3 is not None:
                period_stats.append({
                    "start": 200206,
                    "end": 202412,
                    "corr": m3["corr"],
                    "trend": m3["trend"],
                    "error": m3["error"],
                    "std": m3["std"],
                })

            star_results.append(
                {
                    "label": lab,
                    "years": years,
                    "diff": diff_vals,
                    "trend": trd,
                    "error": err,
                    "std": std_val,
                    "corr": corr_val,
                    "panel_type": "star",
                    "panel_title": f"STAR - {lab}",
                    "col": 0,
                    "period_stats": period_stats,
                }
            )

    # Col 2: RSS minus others (exclude STAR in col 1)
    if "RSS" in all_labels:
        ref_time, ref_anom = tmt_dict["RSS"]
        start_yyyymm = 198001
        end_yyyymm = 202412
        ref_mask = (ref_time >= start_yyyymm) & (ref_time <= end_yyyymm)
        ref_time_sub = ref_time[ref_mask]
        ref_anom_sub = ref_anom[ref_mask]
        ref_map = build_anom_map(ref_time_sub, ref_anom_sub)

        other_labels = [lab for lab in all_labels if lab not in ["STAR", "RSS"]]
        for lab in other_labels:
            time_o, anom_o = tmt_dict[lab]
            common_times_all = sorted(set(ref_time_sub) & set(time_o))
            if not common_times_all:
                continue
            
            mask_o = np.isin(time_o, common_times_all)
            time_o_sub = time_o[mask_o]
            anom_o_sub = anom_o[mask_o]
            if len(time_o_sub) == 0:
                continue

            other_map = build_anom_map(time_o_sub, anom_o_sub)
            common_times = sorted(set(ref_map.keys()) & set(other_map.keys()))
            if len(common_times) < 24:
                continue

            ref_vals = np.array([ref_map[t] for t in common_times], dtype=float)
            oth_vals = np.array([other_map[t] for t in common_times], dtype=float)
            corr_val = calc_corr(ref_vals, oth_vals)

            align_start = 200301
            align_end = 200712
            align_mask = np.array([(t >= align_start and t <= align_end) for t in common_times])
            if np.sum(align_mask) > 0:
                d_align = np.nanmean(ref_vals[align_mask] - oth_vals[align_mask])
            else:
                first_n = min(24, len(common_times))
                d_align = np.nanmean(ref_vals[:first_n] - oth_vals[:first_n])
            diff_vals = ref_vals - oth_vals - d_align

            years = np.array(
                [t // 100 + (t % 100 - 0.5) / 12.0 for t in common_times], dtype=float
            )

            yy = diff_vals.copy()
            missing = -9999.0
            yy_filled = np.where(np.isnan(yy), missing, yy)
            N = len(years)
            trd, err, sigma, amp = cal_trend(years, yy_filled, N, 0, N - 1)
            std_val = float(np.nanstd(diff_vals))

            period_stats = []
            m1 = compute_pair_metrics("RSS", lab, 198001, 202412)
            if m1 is not None:
                period_stats.append({
                    "start": 198001,
                    "end": 202412,
                    "corr": m1["corr"],
                    "trend": m1["trend"],
                    "error": m1["error"],
                    "std": m1["std"],
                })
            m2 = compute_pair_metrics("RSS", lab, 198001, 200112)
            if m2 is not None:
                period_stats.append({
                    "start": 198001,
                    "end": 200112,
                    "corr": m2["corr"],
                    "trend": m2["trend"],
                    "error": m2["error"],
                    "std": m2["std"],
                })
            m3 = compute_pair_metrics("RSS", lab, 200206, 202412)
            if m3 is not None:
                period_stats.append({
                    "start": 200206,
                    "end": 202412,
                    "corr": m3["corr"],
                    "trend": m3["trend"],
                    "error": m3["error"],
                    "std": m3["std"],
                })

            rss_results.append(
                {
                    "label": lab,
                    "years": years,
                    "diff": diff_vals,
                    "trend": trd,
                    "error": err,
                    "std": std_val,
                    "corr": corr_val,
                    "panel_type": "rss",
                    "panel_title": f"RSS - {lab}",
                    "col": 1,
                    "period_stats": period_stats,
                }
            )

    # Col 3: remaining combos (exclude STAR/RSS pairs in cols 1-2)
    other_results = []
    for i, label1 in enumerate(all_labels):
        for label2 in all_labels[i + 1:]:
            # Skip STAR/RSS pairs already in cols 1-2
            if label1 == "STAR" or label2 == "STAR" or label1 == "RSS" or label2 == "RSS":
                continue

            time1, anom1 = tmt_dict[label1]
            time2, anom2 = tmt_dict[label2]
            common_times_all = sorted(set(time1) & set(time2))
            if not common_times_all:
                continue

            plot_start = common_times_all[0]
            plot_end = common_times_all[-1]
            base_metrics = compute_pair_metrics(label1, label2, plot_start, plot_end)
            if base_metrics is None:
                continue

            # Compute stats for three periods:1980-2024、1980-2001 and 200206-202412
            period_stats = []
            m1 = compute_pair_metrics(label1, label2, 198001, 202412)
            if m1 is not None:
                period_stats.append({
                    "start": 198001,
                    "end": 202412,
                    "corr": m1["corr"],
                    "trend": m1["trend"],
                    "error": m1["error"],
                    "std": m1["std"],
                })
            m2 = compute_pair_metrics(label1, label2, 198001, 200112)
            if m2 is not None:
                period_stats.append({
                    "start": 198001,
                    "end": 200112,
                    "corr": m2["corr"],
                    "trend": m2["trend"],
                    "error": m2["error"],
                    "std": m2["std"],
                })
            m3 = compute_pair_metrics(label1, label2, 200206, 202412)
            if m3 is not None:
                period_stats.append({
                    "start": 200206,
                    "end": 202412,
                    "corr": m3["corr"],
                    "trend": m3["trend"],
                    "error": m3["error"],
                    "std": m3["std"],
                })

            other_results.append(
                {
                    "label": f"{label1} - {label2}",
                    "years": base_metrics["years"],
                    "diff": base_metrics["diff"],
                    "trend": base_metrics["trend"],
                    "error": base_metrics["error"],
                    "std": base_metrics["std"],
                    "corr": base_metrics["corr"],
                    "panel_type": "other",
                    "panel_title": f"{label1} - {label2}",
                    "period_stats": period_stats,
                }
            )

    print(f"col 1 (STAR-others): {len(star_results)} panels")
    print(f"col 2 (RSS-others): {len(rss_results)} panels")
    print(f"col 3 (other combos): {len(other_results)} panels")
    print(f"Total: {len(star_results) + len(rss_results) + len(other_results)}  combos")

    if len(star_results) == 0 and len(rss_results) == 0 and len(other_results) == 0:
        print("No valid difference/trend for any dataset pair.")
        return

    n_rows = 4
    n_cols = 3

    for i, result in enumerate(rss_results):
        result["row"] = i % n_rows

    for i, result in enumerate(other_results):
        result["row"] = i % n_rows
        result["col"] = 2  # Row3col

    # Computeunified  y axis limits
    all_diff = []
    all_years = []
    for r in star_results + rss_results + other_results:
        all_diff.append(r["diff"])
        all_years.append(r["years"])
    all_diff = np.concatenate(all_diff)
    all_years = np.concatenate(all_years)
    ymin = float(np.nanmin(all_diff)) - 0.5
    ymax = float(np.nanmax(all_diff))

    base_colors = ["#04599B", "#FF3F3F", "#5506CA", "#008B45", "#FF8C00", "#8B008B", "#FF1493", "#00CED1"]

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex='col',
        sharey=True,
        figsize=(18, 1.8 * n_rows + 1.0),
    )
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    if n_cols == 1:
        axs = axs.reshape(-1, 1)

    for i, item in enumerate(star_results):
        if i >= n_rows:
            break
        ax = axs[i, 0]
        years = item["years"]
        diff_vals = item["diff"]
        panel_title = item["panel_title"]
        trd = item["trend"]
        err = item["error"]
        std_val = item["std"]
        corr_val = item["corr"]

        color = base_colors[i % len(base_colors)]

        xmin_plot = float(np.nanmin(years))
        xmax_plot = float(np.nanmax(years))
        ax.plot([xmin_plot, xmax_plot], [0.0, 0.0], linestyle="--", color="black", linewidth=1.0)

        # timeseries
        ax.plot(years, diff_vals, color=color, linewidth=1.0)

        ax.plot(
            [2002.0 + 6.0 / 12.0, 2002.0 + 6.0 / 12.0],
            [ymin - 5.0, ymax + 5.0],
            color="black",
            linestyle="--",
        )

        ax.set_ylim(ymin-0.5, ymax+0.2)
        ax.set_xlim(1980.0, 2025.0)
        ax.set_yticks([-0.5, 0, 0.5])

        ax.set_xticks([1980, 1990, 2000, 2010, 2020])

        # panel title
        panel_char = chr(97 + i)  # a, b, c, ...
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

        # Statistics: show three-period statistics(1980-2024、1980-2001 and 2002-2024)
        period_stats = item.get("period_stats", [])
        if period_stats:
            lines = []
            for ps in period_stats:
                s = ps["start"]
                e = ps["end"]
                # year only, no month
                txt = (
                    f"{int(s/100)}-{int(e/100)}:"
                    f" CC={ps['corr']:.3f},"
                    f" Trend={ps['trend']:.3f}±{ps['error']:.3f} K Dec$^-$$^1$,"
                    f" STD={ps['std']:.3f} K"
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

        ax.tick_params(labelsize=9, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        if i == len(star_results) - 1:
            xlabel = ax.set_xlabel("Year", fontsize=10)
            xlabel.set_fontweight('bold')

    for i, item in enumerate(rss_results):
        if i >= n_rows:
            break
        ax = axs[i, 1]
        years = item["years"]
        diff_vals = item["diff"]
        panel_title = item["panel_title"]
        trd = item["trend"]
        err = item["error"]
        std_val = item["std"]
        corr_val = item["corr"]

        color = base_colors[(i + len(star_results)) % len(base_colors)]

        xmin_plot = float(np.nanmin(years))
        xmax_plot = float(np.nanmax(years))
        ax.plot([xmin_plot, xmax_plot], [0.0, 0.0], linestyle="--", color="black", linewidth=1.0)

        # timeseries
        ax.plot(years, diff_vals, color=color, linewidth=1.0)

        ax.plot(
            [2002.0 + 6.0 / 12.0, 2002.0 + 6.0 / 12.0],
            [ymin - 5.0, ymax + 5.0],
            color="black",
            linestyle="--",
        )

        ax.set_ylim(ymin-0.5, ymax+0.2)
        ax.set_xlim(1980.0, 2025.0)
        ax.set_yticks([-0.5, 0, 0.5])

        ax.set_xticks([1980, 1990, 2000, 2010, 2020])

        # panel title
        panel_idx = n_cols + i  # row 2 col panel index from 5
        panel_char = chr(97 + panel_idx)
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

        period_stats = item.get("period_stats", [])
        if period_stats:
            lines = []
            for ps in period_stats:
                s = ps["start"]
                e = ps["end"]
                txt = (
                    f"{int(s/100)}-{int(e/100)}:"
                    f" CC={ps['corr']:.3f},"
                    f" Trend={ps['trend']:.3f}±{ps['error']:.3f} K Dec$^-$$^1$,"
                    f" STD={ps['std']:.3f} K"
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

        ax.tick_params(labelsize=9, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        if i == len(rss_results) - 1:
            xlabel = ax.set_xlabel("Year", fontsize=10)
            xlabel.set_fontweight('bold')

    # Plot row 3 col: other combos
    for item in other_results:
        row = item.get("row", 0)
        col = item.get("col", 1)
        if row >= n_rows or col >= n_cols:
            continue

        ax = axs[row, col]
        years = item["years"]
        diff_vals = item["diff"]
        panel_title = item["panel_title"]
        trd = item["trend"]
        err = item["error"]
        std_val = item["std"]
        corr_val = item["corr"]

        color = base_colors[(row * n_cols + col) % len(base_colors)]

        xmin_other = float(np.nanmin(years))
        xmax_other = float(np.nanmax(years))
        ax.plot([xmin_other, xmax_other], [0.0, 0.0], linestyle="--", color="black", linewidth=1.0)

        # timeseries
        ax.plot(years, diff_vals, color=color, linewidth=1.0)

        ax.plot(
            [2002.0 + 6.0 / 12.0, 2002.0 + 6.0 / 12.0],
            [ymin - 5.0, ymax + 5.0],
            color="black",
            linestyle="--",
        )

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(1980.0, 2025.0)
        ax.set_yticks([-0.5, 0, 0.5])

        ax.set_xticks([1980, 1990, 2000, 2010, 2020])

        # panel title(index left-to-right, top-to-bottom)
        panel_idx = row * n_cols + col
        panel_char = chr(97 + panel_idx)
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

        # Statistics: show three-period statistics(1980-2024、1980-2001 and 2002-2024)
        period_stats = item.get("period_stats", [])
        if period_stats:
            lines = []
            for ps in period_stats:
                s = ps["start"]
                e = ps["end"]
                # year only, no month
                txt = (
                    f"{int(s/100)}-{int(e/100)}:"
                    f" CC={ps['corr']:.3f},"
                    f" Trend={ps['trend']:.3f}±{ps['error']:.3f} K Dec$^-$$^1$,"
                    f" STD={ps['std']:.3f} K"
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

        ax.tick_params(labelsize=9, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        if row == n_rows - 1:
            xlabel = ax.set_xlabel("Year", fontsize=10)
            xlabel.set_fontweight('bold')

    # hide unused panels
    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0 and row >= len(star_results):
                axs[row, col].set_visible(False)
            elif col == 1 and row >= len(rss_results):
                axs[row, col].set_visible(False)
            elif col == 2:
                used = any(item.get("row") == row and item.get("col") == col for item in other_results)
                if not used:
                    axs[row, col].set_visible(False)

    fig.text(0.04, 0.5, "TMT Anomaly Difference (K)", va="center", rotation="vertical", fontsize=10, weight='bold')

    # adjustlayout
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    out_name = "../plot/fig_s3.png"
    save_figure_png_pdf(out_name, dpi=600, fig=fig)


if __name__ == "__main__":
    main()
