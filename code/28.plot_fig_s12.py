import numpy as np
from scipy import odr
from scipy.stats import linregress
import random
import matplotlib
matplotlib.use('Agg')  # non-GUI backend to avoid Qt issues in WSL
import matplotlib.pyplot as plt
import os
import glob

from supply_plot import save_figure_png_pdf


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


def odr_linear_regression(x, y):
    """Linear fit using orthogonal distance regression (ODR)"""
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 2:
        return None, None, None, None, None
    
    def linear_model(B, x):
        return B[0] * x + B[1]
    
    model = odr.Model(linear_model)
    data = odr.Data(x_valid, y_valid)
    
    initial_slope = np.sum((x_valid - np.mean(x_valid)) * (y_valid - np.mean(y_valid))) / np.sum((x_valid - np.mean(x_valid))**2)
    initial_intercept = np.mean(y_valid) - initial_slope * np.mean(x_valid)
    odr_obj = odr.ODR(data, model, beta0=[initial_slope, initial_intercept])
    
    output = odr_obj.run()
    
    slope = output.beta[0]
    intercept = output.beta[1]
    
    if len(x_valid) > 1:
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
    else:
        correlation = 0.0
    
    std_err_slope = output.sd_beta[0]
    std_err = std_err_slope
    p_value = None
    
    return slope, intercept, correlation, p_value, std_err


def generate_valid_pairs(range_max, min_diff):
    """Generate valid start/end index pairs"""
    while True:
        a = random.randint(0, range_max)
        b = random.randint(0, range_max)
        if a < b and (b - a) >= min_diff:
            return a, b
        elif b < a and (a - b) >= min_diff:
            return b, a


def monte_carlo_trend_analysis(cwv, sst, n_samples=3000, min_period=120):
    """Monte Carlo trend analysis: random sub-periods for CWV and SST trends (observations)"""
    data_length = len(cwv)
    cwv_trends = []
    sst_trends = []
    
    for _ in range(n_samples):
        start_idx, end_idx = generate_valid_pairs(data_length-1, min_period)
        
        # Time axis in decades
        time_points = np.arange(end_idx - start_idx + 1) / 120.0
        
        cwv_segment = cwv[start_idx:end_idx + 1]
        sst_segment = sst[start_idx:end_idx + 1]
        
        cwv_trend = linregress(time_points, cwv_segment)[0]
        sst_trend = linregress(time_points, sst_segment)[0]
        
        cwv_trends.append(cwv_trend)
        sst_trends.append(sst_trend)
    
    return np.array(cwv_trends), np.array(sst_trends)


def draw_single_subplot(ax, sst_trends, cwv_trends, title_text, x_min=None, x_max=None, y_min=None, y_max=None, 
                         xlabel=None, ylabel=None):
    """Draw density scatter on given axes"""
    if x_min is None:
        x_min = -0.2
    if x_max is None:
        x_max = 0.5
    if y_min is None:
        y_min = -1
    if y_max is None:
        y_max = 5
    
    ax.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--', zorder=1)
    ax.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--', zorder=1)
    
    x = sst_trends
    y = cwv_trends
    
    H, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[x_min, x_max], [y_min, y_max]])
    H_masked = np.ma.masked_where(H == 0, H)
    
    c = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='jet', shading='auto', vmin=0, vmax=20, zorder=2)
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x[valid_mask], y[valid_mask])
        
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
               color='red', linewidth=2, zorder=3)
        
        rmse = np.sqrt(np.nanmean((x[valid_mask] * slope + intercept - y[valid_mask])**2))
        stats_text = 'CC = {:.3f}\nRMSE = {:.3f}'.format(correlation, rmse)
        ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        slope_text = 'Slope = {:.3f}\nIntercept = {:.3f}'.format(slope, intercept)
        ax.text(0.35, 0.18, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Tick font size and bold
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Set axis labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if x_max <= 0.5:
        ax.set_xticks([0, 0.2, 0.4])
    else:
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    
    # Set y ticks from y-axis limits
    if y_max <= 0.7:
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
    else:
        ax.set_yticks([-1, 0, 1, 2, 3, 4])
    
    return c


def get_dataset_label(filename):
    """Dataset label from filename"""
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


def load_obs_data(data_dir, dataset_label, var_name, start_yyyymm, end_yyyymm):
    """Load observational data"""
    data_files = find_data_files(data_dir)
    
    for filepath in data_files:
        label = get_dataset_label(filepath)
        if label != dataset_label:
            continue
        
        time, variables = load_data(filepath)
        if time is None or variables is None:
            continue
        
        if var_name not in variables:
            continue
        
        mask = (time >= start_yyyymm) & (time <= end_yyyymm)
        if np.sum(mask) == 0:
            continue
        
        time_sel = time[mask]
        var_sel = variables[var_name][mask]
        
        # Compute monthly anomalies
        anomaly = calculate_monthly_anomaly(var_sel)
        
        # if CWV, convert to percent
        if var_name == "cwv":
            anomaly = anomaly * 100.0 / 41.0
        
        return time_sel.astype(int), anomaly.astype(float)
    
    return None, None


def main():
    # datadirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
    if not os.path.isdir(data_dir):
        print("Data directory does not exist: {}".format(data_dir))
        return
    
    # Output directory
    out_dir = "../plot"
    os.makedirs(out_dir, exist_ok=True)
    
    # time period
    start_yyyymm = 200206
    end_yyyymm = 202412
    
    # Minimum time span (months)
    min_periods = [60, 120, 180, 240]
    
    # 2x4 panels, shared x and y per row
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 2, 4 * n_rows), 
                             sharex='row', sharey='col')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    print("Processingrow 1row：USTC TCWV vs SST")
    dataset1 = "USTC"
    
    # Load USTC TCWV and SST
    time_cwv, cwv_anom = load_obs_data(data_dir, dataset1, "cwv", start_yyyymm, end_yyyymm)
    time_sst, sst_anom = load_obs_data(data_dir, dataset1, "sst", start_yyyymm, end_yyyymm)
    
    if time_cwv is None or time_sst is None:
        print("CannotLoadUSTCdata")
        return
    
    # Find common times
    common_times = sorted(set(time_cwv) & set(time_sst))
    if len(common_times) == 0:
        print("USTC TCWV and SST: no common times")
        return
    
    # Extract aligned data
    cwv_common = np.array([cwv_anom[np.where(time_cwv == t)[0][0]] for t in common_times])
    sst_common = np.array([sst_anom[np.where(time_sst == t)[0][0]] for t in common_times])
    
    # Full-period trend (200206-202412)
    time_points_full = np.arange(len(cwv_common)) / 120.0  # time in decades
    cwv_trend_full = linregress(time_points_full, cwv_common)[0]
    sst_trend_full = linregress(time_points_full, sst_common)[0]
    
    last_subplot_c = None
    
    for j, min_period in enumerate(min_periods):
        print("  Row 1 col {}: min period {} months".format(j+1, min_period))
        
        cwv_trends, sst_trends = monte_carlo_trend_analysis(
            cwv_common, sst_common, n_samples=3000, min_period=min_period
        )
        
        if len(sst_trends) == 0:
            continue
        
        # Plotpanel
        title = "({}) USTC".format(chr(97 + j))
        # Y tick labels on leftmost panel only
        xlabel = None
        ylabel = None
        
        c = draw_single_subplot(
            axes[0, j], sst_trends, cwv_trends, title,
            x_min=-0.2, x_max=0.5, y_min=-1, y_max=5,
            xlabel=xlabel, ylabel=ylabel
        )
        
        # Black star on panel for full-period trend point
        axes[0, j].scatter(sst_trend_full, cwv_trend_full, marker='*', s=300, 
                          color='black', edgecolors='none', zorder=4, linewidths=0)
        
        # Hide y ticks except leftmost panel
        if j > 0:
            axes[0, j].set_yticklabels([])
        axes[0, j].set_xticklabels([])
        
        if j == 3 and c is not None:
            last_subplot_c = c
    
    print("\nProcessingrow 2row：ERA5 TCWV vs SST")
    dataset2 = "ERA5"
    
    # Load ERA5 TCWV and SST
    time_cwv2, cwv_anom2 = load_obs_data(data_dir, dataset2, "cwv", start_yyyymm, end_yyyymm)
    time_sst2, sst_anom2 = load_obs_data(data_dir, dataset2, "sst", start_yyyymm, end_yyyymm)
    
    if time_cwv2 is None or time_sst2 is None:
        print("CannotLoadERA5data")
        return
    
    # Find common times
    common_times2 = sorted(set(time_cwv2) & set(time_sst2))
    if len(common_times2) == 0:
        print("ERA5 TCWV and SST: no common times")
        return
    
    # Extract aligned data
    cwv_common2 = np.array([cwv_anom2[np.where(time_cwv2 == t)[0][0]] for t in common_times2])
    sst_common2 = np.array([sst_anom2[np.where(time_sst2 == t)[0][0]] for t in common_times2])
    
    # Full-period trend (200206-202412)
    time_points_full2 = np.arange(len(cwv_common2)) / 120.0  # time in decades
    cwv_trend_full2 = linregress(time_points_full2, cwv_common2)[0]
    sst_trend_full2 = linregress(time_points_full2, sst_common2)[0]
    
    last_subplot_c2 = None
    
    for j, min_period in enumerate(min_periods):
        print("  Row 2 col {}: min period {} months".format(j+1, min_period))
        
        cwv_trends2, sst_trends2 = monte_carlo_trend_analysis(
            cwv_common2, sst_common2, n_samples=3000, min_period=min_period
        )
        
        if len(sst_trends2) == 0:
            continue
        
        # Plotpanel
        title = "({}) ERA5".format(chr(97 + 4 + j))
        # Y tick labels on leftmost panel only
        xlabel = None
        ylabel = None
        
        c = draw_single_subplot(
            axes[1, j], sst_trends2, cwv_trends2, title,
            x_min=-0.2, x_max=0.5, y_min=-1, y_max=5,
            xlabel=xlabel, ylabel=ylabel
        )
        
        # Black star on panel for full-period trend point
        axes[1, j].scatter(sst_trend_full2, cwv_trend_full2, marker='*', s=300, 
                          color='black', edgecolors='none', zorder=4, linewidths=0)
        
        # Hide y ticks except leftmost panel
        if j > 0:
            axes[1, j].set_yticklabels([])
        
        # Redraw x ticks when xlabel=None hid them
        # Get x tick positions
        xticks = axes[1, j].get_xticks()
        # Set x tick labels
        axes[1, j].set_xticklabels([f'{x:.1f}' for x in xticks], fontsize=16)
        for tick in axes[1, j].get_xticklabels():
            tick.set_fontweight('bold')
        
        if j == 3 and c is not None:
            last_subplot_c2 = c
    
    # Shared colorbar right of last panel (row 1 col 4 colormap)
    if last_subplot_c is not None:
        ax_last = axes[0, 3]
        bbox = ax_last.get_position()
        # Colorbar spans both rows
        bbox_bottom = axes[1, 3].get_position().y0
        cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox_bottom, 0.0075, bbox.y1 - bbox_bottom])
        cbar = fig.colorbar(last_subplot_c, cax=cbar_ax, orientation='vertical', extend='max')
        cbar.set_ticks([0, 5, 10, 15, 20])
        cbar.set_label('Count', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight('bold')
    
    # adjustlayout(reducepanelspacing)
    plt.subplots_adjust(left=0.06, right=0.88, top=0.95, bottom=0.1, wspace=0.15, hspace=0.1)
    
    # Period labels above row 1 panels
    period_labels = ['5 years', '10 years', '15 years', '20 years']
    for j in range(n_cols):
        ax = axes[0, j]
        bbox = ax.get_position()
        # Label centered above panel
        fig.text(bbox.x0 + bbox.width / 2, bbox.y1 + 0.02, period_labels[j], 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Unified X and Y labels
    fig.text(0.5, 0.02, 'SST Trend (K/decade)', ha='center', va='bottom', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.5, 'TCWV Trend (%/decade)', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = os.path.join(out_dir, "fig_s12.png")
    save_figure_png_pdf(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
