"""Monte Carlo scatter grid for observational variable pairs (draw_supply10+; TCWV/TMT/TLT/SST combos)."""
import numpy as np
from scipy.stats import linregress
from scipy import odr
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob


def save_figure_png_pdf(output_path, dpi=300, fig=None, **savefig_kw):
    """Save PNG and PDF with the same basename."""
    target = fig if fig is not None else plt.gcf()
    base, _ = os.path.splitext(output_path)
    png_path = base + '.png'
    pdf_path = base + '.pdf'
    target.savefig(png_path, dpi=dpi, format='png', **savefig_kw)
    target.savefig(pdf_path, format='pdf', **savefig_kw)
    print(f"Figure saved to: {png_path}")
    print(f"Figure saved to: {pdf_path}")


def calculate_monthly_anomaly(data):
    n = len(data)
    full_years = n // 12
    remaining = n % 12
    reshaped_data = data[:full_years * 12].reshape(full_years, 12)
    climatological_mean = np.nanmean(reshaped_data, axis=0)
    anomaly = reshaped_data - climatological_mean
    anomaly_flattened = anomaly.flatten()
    if remaining > 0:
        remaining_data = data[full_years * 12:]
        remaining_anomaly = remaining_data - climatological_mean[:remaining]
        anomaly_flattened = np.concatenate((anomaly_flattened, remaining_anomaly))
    return anomaly_flattened


def load_data(filepath, start_yyyymm=None, end_yyyymm=None):
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        return None, None
    try:
        data = np.loadtxt(filepath, skiprows=1)
        time = data[:, 0].astype(int)
        with open(filepath, 'r') as f:
            header = f.readline().strip()
        column_names = header.split()
        variables = {}
        for i, col_name in enumerate(column_names):
            col_name_lower = col_name.lower()
            if 'time' in col_name_lower or 'yyyymm' in col_name_lower:
                continue
            elif 'cwv' in col_name_lower or 'vapor' in col_name_lower or 'water' in col_name_lower:
                variables['cwv'] = data[:, i]
            elif 'sst' in col_name_lower or 'sea' in col_name_lower or 'temp' in col_name_lower:
                variables['sst'] = data[:, i]
            elif 'tlt' in col_name_lower or col_name_lower == 'lt':
                variables['tlt'] = data[:, i]
            elif 'tmt' in col_name_lower or col_name_lower == 'mt':
                variables['tmt'] = data[:, i]
        if start_yyyymm is not None:
            mask = time >= start_yyyymm
            time = time[mask]
            for var_name in variables:
                variables[var_name] = variables[var_name][mask]
        if end_yyyymm is not None:
            mask = time <= end_yyyymm
            time = time[mask]
            for var_name in variables:
                variables[var_name] = variables[var_name][mask]
        return time, variables
    except Exception as e:
        print("Error reading file", filepath, ":", e)
        return None, None


def find_files_with_variable(directory, var_key):
    """TXT files in directory whose header contains var_key (exclude cmsaf). var_key: cwv|sst|tmt|tlt"""
    assert var_key in ('cwv', 'sst', 'tmt', 'tlt')
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    out = []
    for file in txt_files:
        try:
            with open(file, 'r') as f:
                header = f.readline().strip()
            column_names = header.split()
            ok = False
            for col_name in column_names:
                col_name_lower = col_name.lower()
                if 'time' in col_name_lower or 'yyyymm' in col_name_lower:
                    continue
                if var_key == 'cwv':
                    if 'cwv' in col_name_lower or 'vapor' in col_name_lower or 'water' in col_name_lower:
                        ok = True
                        break
                elif var_key == 'sst':
                    if 'sst' in col_name_lower or 'sea' in col_name_lower or 'temp' in col_name_lower:
                        ok = True
                        break
                elif var_key == 'tlt':
                    if 'tlt' in col_name_lower or col_name_lower == 'lt':
                        ok = True
                        break
                elif var_key == 'tmt':
                    if 'tmt' in col_name_lower or col_name_lower == 'mt':
                        ok = True
                        break
            if not ok:
                continue
            filename = os.path.basename(file)
            if 'cmsaf' in filename.lower():
                continue
            out.append(file)
        except Exception as e:
            print("Error reading header of", file, ":", e)
            continue
    return sorted(set(out))


def find_data_files(directory):
    return find_files_with_variable(directory, 'cwv'), find_files_with_variable(directory, 'sst')


def exclude_cdr_files(paths):
    """Exclude filenames with cdr_ (consistent with supply12 USTC/CDR exclusion)"""
    out = []
    for p in paths:
        if 'cdr_' in os.path.basename(p).lower():
            continue
        out.append(p)
    return sorted(set(out))


def exclude_era5_merra2(paths):
    """Exclude era5_ / merra2_ prefixes; keep other obs/satellite datasets."""
    out = []
    for p in paths:
        b = os.path.basename(p).lower()
        if b.startswith("era5_") or b.startswith("merra2_"):
            continue
        out.append(p)
    return sorted(set(out))


def generate_valid_pairs(range_max, min_diff):
    while True:
        a = random.randint(0, range_max)
        b = random.randint(0, range_max)
        if a < b and (b - a) >= min_diff:
            return a, b
        if b < a and (a - b) >= min_diff:
            return b, a


def odr_linear_fit(x, y):
    def linear_func(beta, x_val):
        return beta[0] * x_val + beta[1]

    model = odr.Model(linear_func)
    data = odr.RealData(x, y)
    initial = np.polyfit(x, y, 1)
    odr_output = odr.ODR(data, model, beta0=[initial[0], initial[1]]).run()
    return odr_output.beta[0], odr_output.beta[1]


def monte_carlo_trend_analysis(cwv, sst, n_samples=3000, min_period=120):
    data_length = len(cwv)
    cwv_trends = []
    sst_trends = []
    for _ in range(n_samples):
        start_idx, end_idx = generate_valid_pairs(data_length - 1, min_period)
        time_points = np.arange(end_idx - start_idx + 1) / 120.0
        cwv_segment = cwv[start_idx:end_idx + 1]
        sst_segment = sst[start_idx:end_idx + 1]
        cwv_trends.append(linregress(time_points, cwv_segment)[0])
        sst_trends.append(linregress(time_points, sst_segment)[0])
    return np.array(cwv_trends), np.array(sst_trends)


def get_dataset_name(filepath):
    filename = os.path.basename(filepath)
    low = filename.lower()
    if 'cdr_' in low:
        return 'CDR'
    if 'era5_' in low:
        return 'ERA5'
    if 'merra2_' in low:
        return 'MERRA2'
    if 'rss_' in low:
        return 'RSS'
    if 'hadsst_' in low:
        return 'HADSST'
    if 'ersst_' in low:
        return 'ERSST'
    if 'cobe_' in low:
        return 'COBE'
    if 'oisst_' in low:
        return 'OISST'
    return filename.split('_')[0].upper()


def axis_label(var_key):
    return {
        'cwv': 'TCWV Trend (%/decade)',
        'sst': 'SST Trend (K/decade)',
        'tmt': 'TMT Trend (K/decade)',
        'tlt': 'TLT Trend (K/decade)',
    }[var_key]


def get_axis_limits(var_x, var_y):
    """Axis limits consistent with draw_fig2: x=var_x, y=var_y"""
    if var_x == 'sst':
        x_min, x_max = -0.2, 0.5
    else:
        x_min, x_max = -0.3, 0.7
    if var_y == 'cwv':
        y_min, y_max = -1.0, 5.0
    else:
        y_min, y_max = -0.3, 0.7
    return x_min, x_max, y_min, y_max


def _series_for_variable(variables, var_key):
    if var_key not in variables:
        return None
    raw = variables[var_key]
    if var_key == 'cwv':
        return calculate_monthly_anomaly(raw) * 100. / 41.
    return calculate_monthly_anomaly(raw)


def _prepare_pair_datasets(y_files, x_files, var_y, var_x, common_start, common_end,
                           sort_y_key=None, sort_x_key=None):
    y_datasets = []
    x_datasets = []
    for fp in y_files:
        time, variables = load_data(fp, common_start, common_end)
        if time is None:
            continue
        data = _series_for_variable(variables, var_y)
        if data is None:
            continue
        y_datasets.append((get_dataset_name(fp), time, data))
    for fp in x_files:
        time, variables = load_data(fp, common_start, common_end)
        if time is None:
            continue
        data = _series_for_variable(variables, var_x)
        if data is None:
            continue
        x_datasets.append((get_dataset_name(fp), time, data))

    sy = sort_y_key if sort_y_key is not None else (lambda t: (0 if t[0] == 'CDR' else 1, t[0]))
    sx = sort_x_key if sort_x_key is not None else (lambda t: (0 if t[0] == 'CDR' else 1, t[0]))
    y_datasets.sort(key=sy)
    x_datasets.sort(key=sx)
    return y_datasets, x_datasets


def _prepare_datasets(cwv_files, sst_files, common_start, common_end, sort_cwv_key=None, sort_sst_key=None):
    return _prepare_pair_datasets(
        cwv_files, sst_files, 'cwv', 'sst', common_start, common_end,
        sort_cwv_key, sort_sst_key,
    )


def _draw_pair_panel(ax, y_name, x_name, y_time, y_data, x_time, x_data,
                     panel_char, x_min, x_max, y_min, y_max):
    """y=var_y trends, x=var_x trends; returns pcolormesh mappable"""
    common_times = np.intersect1d(y_time, x_time)
    if len(common_times) < 120:
        print(f"Skip {y_name} vs {x_name}: common period too short ({len(common_times)}  months)")
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
        return None

    y_matched = []
    x_matched = []
    for t in common_times:
        iy = np.where(y_time == t)[0]
        ix = np.where(x_time == t)[0]
        if len(iy) > 0 and len(ix) > 0:
            y_matched.append(y_data[iy[0]])
            x_matched.append(x_data[ix[0]])
    y_matched = np.array(y_matched)
    x_matched = np.array(x_matched)

    y_trends, x_trends = monte_carlo_trend_analysis(y_matched, x_matched, n_samples=3000, min_period=120)
    x = x_trends
    y = y_trends

    ax.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--', zorder=1)
    ax.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--', zorder=1)

    H, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[x_min, x_max], [y_min, y_max]])
    H_masked = np.ma.masked_where(H == 0, H)
    c = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='jet', shading='auto', vmin=0, vmax=20, zorder=2)

    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        slope, intercept = odr_linear_fit(x[valid_mask], y[valid_mask])
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept],
                color='red', linewidth=2, zorder=3)
        rmse = np.sqrt(np.nanmean((x[valid_mask] * slope + intercept - y[valid_mask]) ** 2))
        stats_text = f'CC = {correlation:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.82, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        slope_text = f'Slope = {slope:.3f}\nIntercept = {intercept:.3f}'
        ax.text(0.4, 0.15, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    display_y = 'USTC' if y_name == 'CDR' else y_name
    display_x = 'USTC' if x_name == 'CDR' else x_name
    title_text = f'({panel_char}) {display_y} vs {display_x}'
    ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.tick_params(axis='both', which='major', labelsize=16, pad=7)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return c


def _draw_one_panel(ax, cwv_name, sst_name, cwv_time, cwv_data, sst_time, sst_data,
                    panel_char, x_min, x_max, y_min, y_max):
    """Alias: TCWV vs SST"""
    return _draw_pair_panel(ax, cwv_name, sst_name, cwv_time, cwv_data, sst_time, sst_data,
                            panel_char, x_min, x_max, y_min, y_max)


def _panel_label_char(panel_idx):
    if panel_idx < 26:
        return chr(97 + panel_idx)
    first_char_code = 97 + (panel_idx // 26) - 1
    second_char_code = 97 + (panel_idx % 26)
    return chr(first_char_code) + chr(second_char_code)


def draw_obs_pair_figure(y_files, x_files, output_path, var_y, var_x,
                         common_start, common_end,
                         single_y_two_rows=False,
                         sort_y_key=None, sort_x_key=None):
    """
    Generic Monte Carlo trend scatter grid (var_y vs var_x). single_y_two_rows: one y dataset uses 2 rows × ceil(N_x/2) cols.
    """
    assert var_y in ('cwv', 'sst', 'tmt', 'tlt') and var_x in ('cwv', 'sst', 'tmt', 'tlt')
    print(f"Time period: {common_start} - {common_end}  |  {var_y} vs {var_x}")
    y_datasets, x_datasets = _prepare_pair_datasets(
        y_files, x_files, var_y, var_x, common_start, common_end,
        sort_y_key, sort_x_key,
    )

    if not y_datasets or not x_datasets:
        print("No valid data found")
        return

    n_y = len(y_datasets)
    n_x = len(x_datasets)
    x_min, x_max, y_min, y_max = get_axis_limits(var_x, var_y)
    bottom_axis_title = axis_label(var_x)
    left_axis_title = axis_label(var_y)

    use_wrapped = single_y_two_rows and n_y == 1 and n_x >= 1

    last_c = None

    label_ratio = 0.09
    cbar_ratio = 0.09
    hspace_panels = 0.14

    if use_wrapped:
        n_cols = max(1, int(np.ceil(n_x / 2)))
        n_panel_rows = 2
        fig_w = 4 * n_cols
        fig_h = 4 * n_panel_rows + 1.4
        fig = plt.figure(figsize=(fig_w, fig_h), layout='constrained')

        height_ratios = [1] * n_panel_rows + [label_ratio, cbar_ratio]
        gs = GridSpec(
            len(height_ratios), n_cols, figure=fig,
            height_ratios=height_ratios, hspace=hspace_panels,
        )

        axes = np.empty((n_panel_rows, n_cols), dtype=object)
        for i in range(n_panel_rows):
            for j in range(n_cols):
                axes[i, j] = fig.add_subplot(gs[i, j])

        panel_idx = 0
        for j in range(n_x):
            y_name, y_time, y_data = y_datasets[0]
            x_name, x_time, x_data = x_datasets[j]
            row, col = j // n_cols, j % n_cols
            ax = axes[row, col]
            panel_char = _panel_label_char(panel_idx)
            panel_idx += 1
            c = _draw_pair_panel(ax, y_name, x_name, y_time, y_data, x_time, x_data,
                                 panel_char, x_min, x_max, y_min, y_max)
            if c is not None:
                last_c = c

        total_slots = n_panel_rows * n_cols
        for j in range(n_x, total_slots):
            row, col = j // n_cols, j % n_cols
            axes[row, col].set_visible(False)

        for j in range(n_x):
            row, col = j // n_cols, j % n_cols
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            if row < n_panel_rows - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

        ir = n_panel_rows
        label_ax = fig.add_subplot(gs[ir, :])
        label_ax.set_axis_off()
        label_ax.text(
            0.5, 0.5, bottom_axis_title,
            transform=label_ax.transAxes, fontsize=25, fontweight='bold',
            ha='center', va='center',
        )
        cbar_ax = fig.add_subplot(gs[ir + 1, :])

    else:
        fig_w = 4 * n_x
        fig_h = 4 * n_y + 1.4
        fig = plt.figure(figsize=(fig_w, fig_h), layout='constrained')

        height_ratios = [1] * n_y + [label_ratio, cbar_ratio]
        gs = GridSpec(
            n_y + 2, n_x, figure=fig,
            height_ratios=height_ratios, hspace=hspace_panels,
        )

        axes = np.empty((n_y, n_x), dtype=object)
        for i in range(n_y):
            for j in range(n_x):
                axes[i, j] = fig.add_subplot(gs[i, j])

        for i in range(n_y):
            for j in range(n_x):
                axes[i, j].set_xlim(x_min, x_max)
                axes[i, j].set_ylim(y_min, y_max)

        panel_idx = 0
        for i, (y_name, y_time, y_data) in enumerate(y_datasets):
            for j, (x_name, x_time, x_data) in enumerate(x_datasets):
                ax = axes[i, j]
                panel_char = _panel_label_char(panel_idx)
                panel_idx += 1
                c = _draw_pair_panel(ax, y_name, x_name, y_time, y_data, x_time, x_data,
                                     panel_char, x_min, x_max, y_min, y_max)
                if c is not None:
                    last_c = c

                if i < n_y - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

        label_ax = fig.add_subplot(gs[n_y, :])
        label_ax.set_axis_off()
        label_ax.text(
            0.5, 0.5, bottom_axis_title,
            transform=label_ax.transAxes, fontsize=25, fontweight='bold',
            ha='center', va='center',
        )
        cbar_ax = fig.add_subplot(gs[n_y + 1, :])

    fig.supylabel(left_axis_title, fontsize=25, fontweight='bold')

    if last_c is not None:
        cbar_ax.set_axis_off()
        inset_cax = inset_axes(
            cbar_ax, width="62%", height="52%", loc="center", borderpad=0
        )
        cb = fig.colorbar(
            last_c, cax=inset_cax, orientation="horizontal", extend="max"
        )
        cb.set_ticks([0, 5, 10, 15, 20])
        cb.set_label("Count", fontsize=16, fontweight='bold', labelpad=4)
        cb.ax.tick_params(labelsize=16)
        for t in cb.ax.get_xticklabels():
            t.set_fontweight("bold")
    else:
        cbar_ax.set_visible(False)

    save_figure_png_pdf(
        output_path, dpi=300, bbox_inches='tight', pad_inches=0.12,
    )
    plt.close()


def draw_tcwv_sst_figure(cwv_files, sst_files, output_path,
                         common_start, common_end,
                         single_cwv_two_rows=False,
                         sort_cwv_key=None, sort_sst_key=None):
    """TCWV vs SST (draw_supply10–13 compatible)"""
    draw_obs_pair_figure(
        cwv_files, sst_files, output_path, 'cwv', 'sst',
        common_start, common_end,
        single_y_two_rows=single_cwv_two_rows,
        sort_y_key=sort_cwv_key,
        sort_x_key=sort_sst_key,
    )


def filter_rss_filename(paths, substr):
    """basename contains rss_ and substr (cwv/tmt/tlt)"""
    out = []
    for p in paths:
        b = os.path.basename(p).lower()
        if 'rss_' in b and substr in b:
            out.append(p)
    return sorted(set(out))
