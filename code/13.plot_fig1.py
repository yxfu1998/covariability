import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob

from supply_plot import save_figure_png_pdf


def correlation(a, b, ib, ie):
    rel = 0.0
    rms = 0.0
    missing = -9999.0

    amean = bmean = a2mean = b2mean = abmean = 0.0
    sumcos = 0

    for i in range(ib, ie + 1):
        if a[i] != missing and b[i] != missing:
            rms += (a[i] - b[i]) ** 2
            amean += a[i]
            bmean += b[i]
            a2mean += a[i] ** 2
            b2mean += b[i] ** 2
            abmean += a[i] * b[i]
            sumcos += 1

    if sumcos != 0:
        rms = np.sqrt(rms / sumcos)
        amean /= sumcos
        bmean /= sumcos
        a2mean /= sumcos
        b2mean /= sumcos
        abmean /= sumcos
    else:
        rms = missing

    rms1 = a2mean1 = b2mean1 = 0.0
    sumcos = 0

    for i in range(ib, ie + 1):
        if a[i] != missing and b[i] != missing:
            rms1 += (a[i] - amean) * (b[i] - bmean)
            a2mean1 += (a[i] - amean) ** 2
            b2mean1 += (b[i] - bmean) ** 2
            sumcos += 1

    if sumcos != 0:
        rel = rms1 / np.sqrt(a2mean1 * b2mean1)
    else:
        rel = missing

    return rel, rms


def cal_trend(tms, yy, N, is_, ie):
    missing = -9999.0
    yy1 = np.full(N, missing, dtype=np.float32)

    for k in range(is_, ie):
        yy1[k] = yy[k + 1]

    rel, rms = correlation(yy, yy1, is_, ie - 1)
    rlag1 = rel

    sum1 = tmean = bt_m = 0.0

    for k in range(is_, ie + 1):
        if yy[k] != missing:
            tmean += tms[k]
            bt_m += yy[k]
            sum1 += 1

    if sum1 > 0:
        bt_m /= sum1
        tmean /= sum1
    else:
        bt_m = tmean = missing

    top = bot = 0.0
    sum_trd = 0

    for k in range(is_, ie + 1):
        if yy[k] != missing:
            top += (tms[k] - tmean) * (yy[k] - bt_m)
            bot += (tms[k] - tmean) ** 2
            sum_trd += 1

    if bot != 0.0:
        trd = top / bot
        am = bt_m - trd * tmean
    else:
        trd = am = missing

    sigma = 0.0
    nn = 0
    for k in range(is_, ie + 1):
        if yy[k] != missing:
            sigma += (yy[k] - am - trd * tms[k]) ** 2
            nn += 1

    sigma = np.sqrt(sigma / (nn - 2)) if nn > 2 else missing

    trd *= 10.0  # per decade
    tmsq = np.sqrt(bot) if bot > 0 else 0.0
    amp = np.sqrt((1 + rlag1) / (1 - rlag1)) if (1 - rlag1) != 0 else missing
    err = 1.96 * sigma / tmsq * 10.0 * amp if tmsq != 0 and amp != missing else missing

    return trd, err, sigma, amp


def calculate_monthly_anomaly(data):
    """Monthly anomalies using trailing baseline climatology (see baseline_months)."""
    n = len(data)
    baseline_months = 204

    if n >= baseline_months:
        baseline_data = data[-baseline_months:]
        baseline_years = baseline_months // 12
        reshaped_baseline = baseline_data.reshape(baseline_years, 12)
        climatological_mean = np.nanmean(reshaped_baseline, axis=0)
    else:
        full_years = n // 12
        if full_years > 0:
            reshaped_data = data[: full_years * 12].reshape(full_years, 12)
            climatological_mean = np.nanmean(reshaped_data, axis=0)
        else:
            climatological_mean = np.full(12, np.nanmean(data))

    full_years = n // 12
    remaining = n % 12
    anomaly = np.full(n, np.nan)

    if full_years > 0:
        full_years_data = data[-full_years * 12:]
        reshaped_full_years = full_years_data.reshape(full_years, 12)
        anomaly_full_years = reshaped_full_years - climatological_mean
        anomaly[-full_years * 12:] = anomaly_full_years.flatten()

    if remaining > 0:
        remaining_data = data[:remaining]
        remaining_anomaly = remaining_data - climatological_mean[-remaining:]
        anomaly[:remaining] = remaining_anomaly

    return anomaly


def load_data(filepath):
    """Load whitespace-separated time series; infer cwv/sst/tlt/tmt from column names."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None

    try:
        data = np.loadtxt(filepath, skiprows=1)
        with open(filepath, 'r') as f:
            header = f.readline().strip()
        column_names = header.split()
        time = data[:, 0].astype(int)

        variables = {}
        for i, col_name in enumerate(column_names):
            col_name_lower = col_name.lower()
            if 'time' in col_name_lower or 'yyyymm' in col_name_lower:
                continue
            elif 'cwv' in col_name_lower or 'vapor' in col_name_lower or 'water' in col_name_lower:
                variables['cwv'] = data[:, i]
            elif 'sst' in col_name_lower or 'sea' in col_name_lower or 'temp' in col_name_lower:
                variables['sst'] = data[:, i]
            elif 'tlt' in col_name_lower or 'lt' in col_name_lower:
                variables['tlt'] = data[:, i]
            elif 'tmt' in col_name_lower or 'mt' in col_name_lower:
                variables['tmt'] = data[:, i]

        return time, variables

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None


def find_data_files(directory):
    """Return climate-related .txt files under directory."""
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    keywords = [
        'ustc', 'star', 'rss', 'uah', 'cmsaf', 'merra2', 'era5',
        'oisst', 'cobe', 'ersst', 'hadisst', 'hadsst',
    ]
    climate_files = []
    for file in txt_files:
        filename = os.path.basename(file)
        if any(keyword in filename.lower() for keyword in keywords):
            climate_files.append(file)
    return climate_files


def load_cmip6_data(filepath, start_yyyymm=None, end_yyyymm=None):
    """Load cal_cmip_trend.py output: time(yyyymm) ts_cwv ts_sst ts_tlt ts_tmt."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None

    try:

        data = np.loadtxt(filepath, skiprows=1)
        yyyymm = data[:, 0].astype(int)
        cwv = data[:, 1]
        sst = data[:, 2]
        tlt = data[:, 3]
        tmt = data[:, 4]

        if start_yyyymm is not None:
            mask = yyyymm >= start_yyyymm
            sst = sst[mask]
            cwv = cwv[mask]
            tlt = tlt[mask]
            tmt = tmt[mask]
            yyyymm = yyyymm[mask]

        if end_yyyymm is not None:
            mask = yyyymm <= end_yyyymm
            sst = sst[mask]
            cwv = cwv[mask]
            tlt = tlt[mask]
            tmt = tmt[mask]
            yyyymm = yyyymm[mask]

        if len(sst) == 0:
            return None, None

        sst_anomaly = calculate_monthly_anomaly(sst)
        cwv_anomaly = calculate_monthly_anomaly(cwv)
        tlt_anomaly = calculate_monthly_anomaly(tlt)
        tmt_anomaly = calculate_monthly_anomaly(tmt)
        cwv_anomaly = cwv_anomaly * 100.0 / 41.0

        variables = {
            'cwv': cwv_anomaly,
            'sst': sst_anomaly,
            'tlt': tlt_anomaly,
            'tmt': tmt_anomaly,
        }
        return yyyymm, variables

    except Exception as e:
        print(f"Error reading CMIP6 file {filepath}: {e}")
        return None, None


def load_all_cmip6_anomalies(cmip6_dir, var_name, start_yyyymm=198001, end_yyyymm=202412):
    """CMIP6 ensemble 5th/95th percentile anomalies at each month."""
    cmip6_files = glob.glob(os.path.join(cmip6_dir, "*.txt"))

    if not cmip6_files:
        print(f"No CMIP6 files found in {cmip6_dir}")
        return None, None, None, None

    all_anomalies_by_time = {}

    for filepath in cmip6_files:
        time_yyyymm, variables = load_cmip6_data(filepath, start_yyyymm, end_yyyymm)
        if time_yyyymm is None or variables is None:
            continue
        if var_name not in variables:
            continue
        anomaly = variables[var_name]
        for t, a in zip(time_yyyymm, anomaly):
            if not np.isnan(a):
                all_anomalies_by_time.setdefault(t, []).append(a)

    if not all_anomalies_by_time:
        print(f"No valid CMIP6 data for variable: {var_name}")
        return None, None, None, None

    sorted_times = sorted(all_anomalies_by_time.keys())
    p5_values = []
    p95_values = []
    valid_times = []

    for t in sorted_times:
        values = np.array(all_anomalies_by_time[t])
        if len(values) > 0:
            p5_values.append(np.percentile(values, 5))
            p95_values.append(np.percentile(values, 95))
            valid_times.append(t)

    if not valid_times:
        return None, None, None, None

    time_yyyymm = np.array(valid_times)
    time_years = time_yyyymm // 100 + (time_yyyymm % 100 - 0.5) / 12.0
    return time_yyyymm, time_years, np.array(p5_values), np.array(p95_values)


def plot_anomaly_time_series(data_files, output_path, cmip6_dir=None):
    """4-panel anomaly time series with trend bar charts on the right."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(
        4, 3, width_ratios=[3, 1, 0.2], figure=fig,
        left=0.1, right=0.95, top=0.95, bottom=0.1,
        hspace=0.3, wspace=0.3,
    )

    main_axes = []
    trend_axes = []
    for i in range(4):
        main_axes.append(fig.add_subplot(gs[i, 0]))
        trend_axes.append(fig.add_subplot(gs[i, 1]))

    var_names = ['cwv', 'sst', 'tlt', 'tmt']
    var_labels = ['CWV Anomaly', 'SST Anomaly', 'TLT Anomaly', 'TMT Anomaly']
    var_units = ['mm', 'K', 'K', 'K']

    cmip6_data = {}
    if cmip6_dir is not None and os.path.exists(cmip6_dir):
        print(f"Loading CMIP6 data from: {cmip6_dir}")
        for var_name in var_names:
            time_yyyymm, time_years, p5_values, p95_values = load_all_cmip6_anomalies(
                cmip6_dir, var_name, start_yyyymm=198001, end_yyyymm=202412
            )
            if time_years is not None:
                cmip6_data[var_name] = {
                    'time': time_years,
                    'p5': p5_values,
                    'p95': p95_values,
                }
                print(f"  {var_name}: {len(time_years)} CMIP6 time steps")
            else:
                print(f"  {var_name}: no valid CMIP6 data")

    color_map = {
        'USTC': 'red',
        'STAR': 'red',
        'RSS': 'blue',
        'ERA5': 'purple',
        'MERRA2': 'orange',
        'UAH': 'green',
        'CMSAF': 'brown',
        'COBE': 'pink',
        'ERSST': 'cyan',
        'HADISST': 'magenta',
        'HADSST': 'olive',
        'OISST': 'teal',
    }

    for i, (var_name, var_label, var_unit) in enumerate(zip(var_names, var_labels, var_units)):
        ax = main_axes[i]
        trend_ax = trend_axes[i]

        all_times = []
        all_anomalies = []
        file_labels = []
        trend_data = []

        for filepath in data_files:
            time, variables = load_data(filepath)
            if time is not None and var_name in variables:
                time_mask = (time >= 198001) & (time <= 202412)
                if np.sum(time_mask) == 0:
                    continue

                filtered_time = time[time_mask]
                filtered_data = variables[var_name][time_mask]
                anomaly = calculate_monthly_anomaly(filtered_data)

                if var_name == 'cwv':
                    anomaly = anomaly * 100.0 / 41.0

                years = filtered_time // 100 + (filtered_time % 100 - 0.5) / 12.0

                filename = os.path.basename(filepath)
                if 'ustc_' in filename.lower():
                    label = 'USTC'
                elif 'star_' in filename.lower():
                    label = 'STAR'
                elif 'rss_' in filename.lower():
                    label = 'RSS'
                elif 'uah_' in filename.lower():
                    label = 'UAH'
                elif 'cmsaf_' in filename.lower():
                    label = 'CMSAF'
                elif 'merra2_' in filename.lower():
                    label = 'MERRA2'
                elif 'era5_' in filename.lower():
                    label = 'ERA5'
                elif 'cobe_' in filename.lower():
                    label = 'COBE'
                elif 'ersst_' in filename.lower():
                    label = 'ERSST'
                elif 'hadisst_' in filename.lower():
                    label = 'HADISST'
                elif 'hadsst_' in filename.lower():
                    label = 'HADSST'
                elif 'oisst_' in filename.lower():
                    label = 'OISST'
                else:
                    label = filename.split('_')[0].upper()

                trend_start = 2002.4
                trend_end = 2025
                trend_mask = (years >= trend_start) & (years <= trend_end)
                trend_years = years[trend_mask]
                trend_anomaly = anomaly[trend_mask]

                if len(trend_years) > 12:
                    trend_years_clean = []
                    trend_anomaly_clean = []
                    for k in range(len(trend_years)):
                        if not np.isnan(trend_anomaly[k]):
                            trend_years_clean.append(trend_years[k])
                            trend_anomaly_clean.append(trend_anomaly[k])

                    if len(trend_years_clean) > 12:
                        trend_years_array = np.array(trend_years_clean)
                        trend_anomaly_array = np.array(trend_anomaly_clean)

                        trend_anomaly_filled = np.where(
                            np.isnan(trend_anomaly_array), -9999.0, trend_anomaly_array
                        )
                        trd, err, sigma, amp = cal_trend(
                            trend_years_array,
                            trend_anomaly_filled,
                            len(trend_years_array),
                            0,
                            len(trend_years_array) - 1,
                        )

                        if trd != -9999.0 and err != -9999.0:
                            trend_data.append((label, trd, err))

                all_times.append(years)
                all_anomalies.append(anomaly)
                file_labels.append(label)

        if var_name in cmip6_data:
            cmip6_info = cmip6_data[var_name]
            ax.fill_between(
                cmip6_info['time'], cmip6_info['p5'], cmip6_info['p95'],
                color='black', alpha=0.1, zorder=0,
            )

        for times, anomalies, label in zip(all_times, all_anomalies, file_labels):
            color = color_map.get(label, 'gray')
            ax.plot(times, anomalies, color=color, linewidth=1, alpha=0.8)

        ax.set_ylabel('', fontsize=0)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1980, 2025)
        ax.set_xticks([1980, 1990, 2000, 2010, 2020])

        if var_name in ['tlt', 'tmt']:
            ax.set_ylim(-1.5, 1.5)

        ax.tick_params(axis='both', which='major', labelsize=14, width=1.5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        subplot_labels = [
            '(a) TCWV (%, Trend: % decade$^{-1}$)',
            '(b) SST (K, Trend: K decade$^{-1}$)',
            '(c) TLT (K, Trend: K decade$^{-1}$)',
            '(d) TMT (K, Trend: K decade$^{-1}$ )',
        ]
        ax.text(
            0.02, 0.94, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', verticalalignment='top',
        )

        if trend_data:
            labels = [item[0] for item in trend_data]
            trends = [item[1] for item in trend_data]
            uncertainties = [item[2] for item in trend_data]

            if var_name == 'cwv':
                desired_order = ['USTC', 'RSS', 'ERA5', 'MERRA2']
            elif var_name == 'sst':
                desired_order = [
                    'USTC', 'OISST', 'COBE', 'ERSST', 'HADSST',
                    'HADISST', 'ERA5', 'MERRA2',
                ]
            elif var_name in ['tlt', 'tmt']:
                desired_order = ['STAR', 'RSS', 'UAH', 'ERA5', 'MERRA2']
            else:
                desired_order = []

            ordered_data = []
            for order_label in desired_order:
                if order_label in labels:
                    idx = labels.index(order_label)
                    ordered_data.append((labels[idx], trends[idx], uncertainties[idx]))

            for label, trend, uncert in zip(labels, trends, uncertainties):
                if label not in desired_order:
                    ordered_data.append((label, trend, uncert))

            if ordered_data:
                labels_ordered = [item[0] for item in ordered_data]
                trends_ordered = [item[1] for item in ordered_data]
                uncertainties_ordered = [item[2] for item in ordered_data]
                colors_ordered = [color_map.get(label, 'gray') for label in labels_ordered]

                y_pos = np.arange(len(labels_ordered))
                trend_ax.barh(y_pos, trends_ordered, color=colors_ordered, alpha=0.7)

                trend_ax.set_yticks(y_pos)
                yticklabels = trend_ax.set_yticklabels(
                    labels_ordered, fontsize=12, fontweight='bold'
                )
                for j, (label, color) in enumerate(zip(labels_ordered, colors_ordered)):
                    yticklabels[j].set_color(color)
                trend_ax.invert_yaxis()

                if var_name == 'cwv':
                    trend_ax.set_xlim(0, 3.7)
                elif var_name == 'sst':
                    trend_ax.set_xlim(0, 0.38)
                elif var_name == 'tlt':
                    trend_ax.set_xlim(0, 0.48)
                elif var_name == 'tmt':
                    trend_ax.set_xlim(0, 0.5)

                trend_ax.grid(True, alpha=0.3, axis='x')
                trend_ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

                trend_ax.tick_params(axis='x', which='major', labelsize=12, width=1.5)
                for label in trend_ax.get_xticklabels():
                    label.set_fontweight('bold')

                xlim_range = trend_ax.get_xlim()[1] - trend_ax.get_xlim()[0]
                for j, (trend, error, bar) in enumerate(
                    zip(trends_ordered, uncertainties_ordered, range(len(labels_ordered)))
                ):
                    text_x = trend + 0.02 * xlim_range
                    trend_ax.text(
                        text_x, j, f'{trend:.3f}±{error:.3f}',
                        va='center', fontsize=11, fontweight='bold',
                    )

    main_axes[-1].set_xlabel('Year', fontsize=16, fontweight='bold')
    trend_axes[-1].set_xlabel('Trend (2002-2024)', fontsize=14, fontweight='bold')
    fig.text(
        0.04, 0.5, 'Climate Anomaly', va='center', rotation='vertical',
        fontsize=18, fontweight='bold',
    )

    save_figure_png_pdf(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    data_dir = "../data"
    cmip6_dir = "../data/cmip6"

    data_files = find_data_files(data_dir)

    if not data_files:
        print(f"No data files found in {data_dir}")
        return

    print(f"Found {len(data_files)} data files:")
    for file in data_files:
        print(f"  {os.path.basename(file)}")

    output_path = os.path.join("../plot", "fig1.png")
    plot_anomaly_time_series(data_files, output_path, cmip6_dir=cmip6_dir)


if __name__ == "__main__":
    main()
