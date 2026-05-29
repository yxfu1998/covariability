import numpy as np  
from scipy import odr
from scipy.stats import linregress, gaussian_kde
import random
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt plugin issues on WSL
import matplotlib.pyplot as plt
import os
import glob
import re

from supply_plot import save_figure_png_pdf


def odr_linear_regression(x, y):
    """Linear fit using orthogonal distance regression (ODR)
    
    Args:
        x: independent variable array
        y: dependent variable array
    
    Returns:
        tuple: (slope, intercept, r_value, p_value, std_err)
        Return format matches linregress for compatibility
    """
    # Filter NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 2:
        return None, None, None, None, None
    
    def linear_model(B, x):
        """Linear model: y = B[0] * x + B[1]"""
        return B[0] * x + B[1]
    
    model = odr.Model(linear_model)
    
    # If x and y have errors, use sx and sy
    # Assume equal x/y errors (unit weights)
    data = odr.Data(x_valid, y_valid)
    
    # Create ODR object
    # Initial values from least squares
    initial_slope = np.sum((x_valid - np.mean(x_valid)) * (y_valid - np.mean(y_valid))) / np.sum((x_valid - np.mean(x_valid))**2)
    initial_intercept = np.mean(y_valid) - initial_slope * np.mean(x_valid)
    odr_obj = odr.ODR(data, model, beta0=[initial_slope, initial_intercept])
    
    # Run ODR
    output = odr_obj.run()
    
    slope = output.beta[0]
    intercept = output.beta[1]
    
    # Compute correlation coefficient (raw data)
    if len(x_valid) > 1:
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
    else:
        correlation = 0.0
    
    # ODR does not provide p-value directly; use standard errors
    # std_err from covariance matrix
    std_err_slope = output.sd_beta[0]
    std_err_intercept = output.sd_beta[1]
    
    # Return std_err as slope standard error for compatibility
    std_err = std_err_slope
    
    # p-value from t-stat would be needed; None here (ODR)
    p_value = None
    
    return slope, intercept, correlation, p_value, std_err


def calculate_monthly_anomaly(data):
    """Compute monthly anomalies"""
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


# CMIP6 model-ensemble files skipped in rows 3–4 only (other rows still use them)
CMIP6_EXCLUDED_FOR_ROW3_ROW4 = {
    'CESM2-WACCM_r1i1p1f1_s198001_e202412.txt',
    'E3SM-1-1-ECA_r1i1p1f1_s198001_e202412.txt',
    'CESM2_r11i1p1f1_s198001_e202412.txt',
    'E3SM-1-0_r1i1p1f1_s198001_e202412.txt',
    'ACCESS-ESM1-5_r3i1p1f1_s198001_e202412.txt',
    'CIESM_r1i1p1f1_s198001_e202412.txt',
    'ACCESS-ESM1-5_r1i1p1f1_s198001_e202412.txt',
    'CanESM5_r19i1p2f1_s198001_e202412.txt',
    'KACE-1-0-G_r1i1p1f1_s198001_e202412.txt',
    'ACCESS-ESM1-5_r2i1p1f1_s198001_e202412.txt',
    'ACCESS-CM2_r2i1p1f1_s198001_e202412.txt',
    'CESM2_r10i1p1f1_s198001_e202412.txt',
    'CESM2_r4i1p1f1_s198001_e202412.txt',
    'CanESM5_r3i1p1f1_s198001_e202412.txt',
    'CAMS-CSM1-0_r1i1p1f1_s198001_e202412.txt',
}

# Unified obs/reanalysis marker style in cols 1 and 4
OBS_MARKER_SIZE = 180
OBS_MARKER_LW = 1.5


def load_cmip6_data(filepath, start_yyyymm=None, end_yyyymm=None, var_x='sst', var_y='cwv', row_index=None):
    """Read cal_cmip_trend.py output (TXT, space-separated)
    
    File format:time(yyyymm) ts_cwv ts_sst ts_tlt ts_tmt
    
    Args:
        filepath: filepath
        start_yyyymm: Start time (YYYYMM, e.g. 198001)
        end_yyyymm: End time (YYYYMM, e.g. 202412)
        var_x: x-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
        row_index: Row index 0–4; skip CMIP6_EXCLUDED_FOR_ROW3_ROW4 when row is 2 or 3.
    
    Returns:
        tuple: (x_anomaly, y_anomaly, years) or (None, None, None)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None, None
    
    try:
        # Skip listed model-ensemble files only in rows 3–4
        # if row_index in (2, 3):
        #     if os.path.basename(filepath) in CMIP6_EXCLUDED_FOR_ROW3_ROW4:
        #         return None, None, None

        data = np.loadtxt(filepath, skiprows=1)

        # Column order: time(yyyymm), ts_cwv, ts_sst, ts_tlt, ts_tmt
        yyyymm = data[:, 0].astype(int)  # time column
        cwv = data[:, 1]  # ts_cwv
        sst = data[:, 2]  # ts_sst
        tlt = data[:, 3]  # ts_tlt
        tmt = data[:, 4]  # ts_tmt
        
        mask = np.ones(len(yyyymm), dtype=bool)
        
        if start_yyyymm is not None:
            mask = mask & (yyyymm >= start_yyyymm)
        
        if end_yyyymm is not None:
            mask = mask & (yyyymm <= end_yyyymm)
        
        yyyymm = yyyymm[mask]
        cwv = cwv[mask]
        sst = sst[mask]
        tlt = tlt[mask]
        tmt = tmt[mask]
        
        if len(yyyymm) == 0:
            return None, None, None
        
        # Select column by variable name
        var_map = {
            'cwv': cwv,
            'sst': sst,
            'tlt': tlt,
            'tmt': tmt
        }
        
        x_data = var_map.get(var_x.lower())
        y_data = var_map.get(var_y.lower())
        
        if x_data is None or y_data is None:
            return None, None, None
        
        x_anomaly = calculate_monthly_anomaly(x_data)
        y_anomaly = calculate_monthly_anomaly(y_data)
        
        # Convert CWV to percent if y-axis is CWV
        if var_y.lower() == 'cwv':
            y_anomaly = y_anomaly * 100. / 41.
        
        # Extract years for return (from yyyymm)
        years = (yyyymm // 100).astype(float)
        
        return x_anomaly, y_anomaly, years
        
    except Exception as e:
        print(f"Error reading file {filepath}  : {e}")
        return None, None, None


def generate_valid_pairs(range_max, min_diff):
    """Generate valid start/end date pairs"""
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
        
        # Build time axis in decades
        time_points = np.arange(end_idx - start_idx + 1) / 120.0
        
        cwv_segment = cwv[start_idx:end_idx + 1]
        sst_segment = sst[start_idx:end_idx + 1]
        
        # Compute linear trend with ODR
        cwv_trend = linregress(time_points, cwv_segment)[0]
        sst_trend = linregress(time_points, sst_segment)[0]
            
        cwv_trends.append(cwv_trend)
        sst_trends.append(sst_trend)
    
    return np.array(cwv_trends), np.array(sst_trends)


def load_or_compute_cmip6_mc_stats(cmip6_dir, var_combos, start_yyyymm=200206, end_yyyymm=202412,
                                   n_samples=3000, min_period=120):
    """Load or compute CMIP6 Monte Carlo statistics (for rows 4–5)

    Returns dict with:
        - slopes_per_combo: [num_combos][num_models] slope
        - intercepts_per_combo: same for intercept
        - correlations_per_combo: same for correlation
        - model_names_per_combo: same for model-ensemble names
        - filepaths_per_combo: same as above, filepath
        - x_trends_list_per_combo: [num_combos][num_models] x_trends array list per model-ensemble
        - y_trends_list_per_combo: [num_combos][num_models] y_trends array list per model-ensemble
    """
    cache_path = os.path.join("../data", "cmip_monte_carlo.npy")

    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
            if (isinstance(cache, dict)
                    and "var_combos" in cache
                    and len(cache.get("var_combos", [])) == len(var_combos)
                    and "x_trends_list_per_combo" in cache
                    and "y_trends_list_per_combo" in cache):
                print(f"Load CMIP6 Monte Carlo stats from cache: {cache_path}")
                return cache
            else:
                print("Cache format mismatch; will recompute")
        except Exception as e:
            print(f"Cache read error; will recompute: {e}")

    print("Computing CMIP6 Monte Carlo statistics (first run may be slow)...")
    txt_files = glob.glob(os.path.join(cmip6_dir, "*.txt"))

    slopes_per_combo = [[] for _ in range(len(var_combos))]
    intercepts_per_combo = [[] for _ in range(len(var_combos))]
    correlations_per_combo = [[] for _ in range(len(var_combos))]
    model_names_per_combo = [[] for _ in range(len(var_combos))]
    filepaths_per_combo = [[] for _ in range(len(var_combos))]
    x_trends_list_per_combo = [[] for _ in range(len(var_combos))]  # x_trends list per model-ensemble
    y_trends_list_per_combo = [[] for _ in range(len(var_combos))]  # y_trends list per model-ensemble

    for j, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Computing combo {j+1}/{len(var_combos)}: {label}")

        for filepath in txt_files:
            model_name = os.path.basename(filepath).replace('.txt', '')
            x_anom, y_anom, _ = load_cmip6_data(filepath, start_yyyymm, end_yyyymm, var_x, var_y, row_index=j)
            if x_anom is None or y_anom is None:
                continue
    
            # Monte Carlo: random sub-period trends per model-ensemble and variable combo
            y_trends, x_trends = monte_carlo_trend_analysis(y_anom, x_anom,
                                                            n_samples=n_samples,
                                                            min_period=min_period)
            if len(x_trends) == 0:
                continue
            
            # Save MC samples per model-ensemble (for aggregate regression, CC filter)
            valid_mask = ~(np.isnan(x_trends) | np.isnan(y_trends))
            if np.sum(valid_mask) >= 10:
                x_trends_list_per_combo[j].append(x_trends[valid_mask])
                y_trends_list_per_combo[j].append(y_trends[valid_mask])

            x_valid = x_trends[valid_mask]
            y_valid = y_trends[valid_mask]
            slope, intercept, r_value, p_value, std_err = odr_linear_regression(x_valid, y_valid)
            if slope is None or intercept is None or r_value is None:
                continue
    
            slopes_per_combo[j].append(slope)
            intercepts_per_combo[j].append(intercept)
            correlations_per_combo[j].append(r_value)
            model_names_per_combo[j].append(model_name)
            filepaths_per_combo[j].append(filepath)

    stats = {
        "var_combos": var_combos,
        "slopes_per_combo": slopes_per_combo,
        "intercepts_per_combo": intercepts_per_combo,
        "correlations_per_combo": correlations_per_combo,
        "model_names_per_combo": model_names_per_combo,
        "filepaths_per_combo": filepaths_per_combo,
        "x_trends_list_per_combo": x_trends_list_per_combo,
        "y_trends_list_per_combo": y_trends_list_per_combo,
    }

    try:
        np.save(cache_path, stats)
        print(f"Saved CMIP6 Monte Carlo statistics to: {cache_path}")
    except Exception as e:
        print(f"Error saving cache file: {e}")

    return stats

def process_data_for_period_single_trend(data_dir, start_yyyymm, end_yyyymm, var_x='sst', var_y='cwv',
                                         excluded_filenames=None):
    """Process CMIP6 period: one trend per model-ensemble
    
    Args:
        data_dir: Data directory
        start_yyyymm: Start time
        end_yyyymm: End time
        var_x: x-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
    """
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if excluded_filenames is not None:
        excluded_filenames = set(excluded_filenames)
    
    if not txt_files:
        print(f"In directory {data_dir} no txt files found")
        return None, None
    
    all_x_trends = []
    all_y_trends = []
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)

        # Skip member if in exclusion list
        if excluded_filenames is not None and filename in excluded_filenames:
            continue
        
        # Read data with time filter
        x_anomaly, y_anomaly, years = load_cmip6_data(filepath, start_yyyymm, end_yyyymm, var_x, var_y)
        
        if x_anomaly is None or y_anomaly is None:
            continue
        
        if len(x_anomaly) < 180:
            continue
        
        # Build time axis in decades
        time_points = np.arange(len(x_anomaly)) / 120.0
        
        # Check for NaN values
        if np.any(np.isnan(x_anomaly)) or np.any(np.isnan(y_anomaly)):
            continue
        
        # Compute linear trend (full period) with ODR
        try:
            x_trend = linregress(time_points, x_anomaly)[0]
            y_trend = linregress(time_points, y_anomaly)[0]
            
            all_x_trends.append(x_trend)
            all_y_trends.append(y_trend)
        except:
            continue
    
    if len(all_x_trends) == 0:
        return None, None
    
    all_x_trends = np.array(all_x_trends)
    all_y_trends = np.array(all_y_trends)
    
    return all_x_trends, all_y_trends


def draw_single_subplot(ax, sst_trends, cwv_trends, title_text, x_min=None, x_max=None, y_min=None, y_max=None, 
                         xlabel=None, ylabel=None, col_idx=0, show_xticklabels=True):
    """Draw single scatter on given axes
    
    Args:
        ax: matplotlib Axes
        sst_trends: x-axis data
        cwv_trends: y-axis data
        title_text: title text
        x_min, x_max, y_min, y_max: axis limits
        xlabel: x label (col_idx==1 only)
        ylabel: y label (col_idx==0 only)
        col_idx: Column index (0=col1, 1=col2, 2=col3, 3=col4)
        show_xticklabels: Whether to show x tick labels (default True)
    """
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
    
    # Plot density
    H, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[x_min, x_max], [y_min, y_max]])
    H_masked = np.ma.masked_where(H == 0, H)
    
    c = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='jet', shading='auto', vmin=0, vmax=20, zorder=2)
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x[valid_mask], y[valid_mask])
        
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
               color='red', linewidth=2, zorder=3)
        
        # Add statistics text (original format/position)
        rmse = np.sqrt(np.nanmean((x[valid_mask] * slope + intercept - y[valid_mask])**2))
        stats_text = f'CC = {correlation:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        slope_text = f'Slope = {slope:.3f}\nIntercept = {intercept:.3f}'
        ax.text(0.35, 0.18, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Set axis labels on specified columns only
    if col_idx == 1 and xlabel:  # show xlabel on column 2 only
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel('')
    
    if col_idx == 0 and ylabel:  # show ylabel on column 1 only
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    if not show_xticklabels:
        ax.set_xticklabels([])
    
    # y tick labels on row 1, col 1 only
    if col_idx != 0:
        ax.set_yticklabels([])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Set x ticks from x-axis range(consistent with row 3)
    if x_max <= 0.5:
        ax.set_xticks([0, 0.2, 0.4])
    else:
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    
    return c

def get_var_label(var_name):
    """Build label text from variable names"""
    label_map = {
        'sst': 'SST Trend (K/decade)',
        'tmt': 'TMT Trend (K/decade)',
        'tlt': 'TLT Trend (K/decade)',
        'cwv': 'TCWV Trend (%/decade)',
    }
    return label_map.get(var_name, f'{var_name.upper()} Trend')


def load_obs_data_and_calc_trend(filepath, var_name, start_yyyymm=200206, end_yyyymm=202412):
    """Read observational file and compute trend
    
    Args:
        filepath: datafilepath
        var_name: variable name ('cwv', 'sst', 'tmt', 'tlt')
        start_yyyymm: Start time（YYYYMM format）
        end_yyyymm: End time（YYYYMM format）
    
    Returns:
        trend: trend (K/decade or %/decade); None on failure
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        
        time = data[:, 0].astype(int)  # YYYYMM format
        
        # Error reading file header
        with open(filepath, 'r') as f:
            header = f.readline().strip()
        
        column_names = header.split()
        
        # Column for variable from header
        var_data = None
        for i, col_name in enumerate(column_names):
            col_name_lower = col_name.lower()
            if 'time' in col_name_lower or 'yyyymm' in col_name_lower:
                continue
            elif var_name.lower() == 'cwv' and ('cwv' in col_name_lower or 'vapor' in col_name_lower or 'water' in col_name_lower):
                var_data = data[:, i]
                break
            elif var_name.lower() == 'sst' and ('sst' in col_name_lower or 'sea' in col_name_lower or 'temp' in col_name_lower):
                var_data = data[:, i]
                break
            elif var_name.lower() == 'tmt' and 'tmt' in col_name_lower:
                var_data = data[:, i]
                break
            elif var_name.lower() == 'tlt' and 'tlt' in col_name_lower:
                var_data = data[:, i]
                break
        
        if var_data is None:
            print(f"In file {filepath} variable not found {var_name}")
            return None
        
        # Filter time range
        mask = (time >= start_yyyymm) & (time <= end_yyyymm)
        time_filtered = time[mask]
        var_data_filtered = var_data[mask]
        
        if len(var_data_filtered) < 120:  # Need at least 10 years of data
            print(f"Insufficient data length: {len(var_data_filtered)}  months")
            return None
        
        var_anomaly = calculate_monthly_anomaly(var_data_filtered)
        
        # Convert CWV to percent if applicable
        if var_name.lower() == 'cwv':
            var_anomaly = var_anomaly * 100. / 41.
        
        # Build time axis in decades
        time_points = np.arange(len(var_anomaly)) / 120.0
        
        # Check for NaN values
        valid_mask = ~np.isnan(var_anomaly)
        if np.sum(valid_mask) < 120:
            print(f"Insufficient valid points: {np.sum(valid_mask)}")
            return None
        
        var_anomaly_valid = var_anomaly[valid_mask]
        time_points_valid = time_points[valid_mask]
        
        trend = linregress(time_points_valid, var_anomaly_valid)[0]
        
        return trend
        
    except Exception as e:
        print(f"Error reading file {filepath}  : {e}")
        return None


def find_data_file(directory, dataset_name, var_type=None):
    """Find dataset file (consistent with draw_fig3, for row-4 multi-product combos)"""
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    dataset_name_lower = dataset_name.lower()

    if var_type is not None:
        var_type_lower = var_type.lower()
        for file in txt_files:
            filename = os.path.basename(file).lower()
            if dataset_name_lower in filename and var_type_lower in filename:
                if 'cmsaf' not in filename:
                    return file

        for file in txt_files:
            filename = os.path.basename(file).lower()
            if dataset_name_lower in filename and 'cmsaf' not in filename:
                try:
                    with open(file, 'r') as f:
                        header = f.readline().strip()
                    column_names = header.split()
                    for col_name in column_names:
                        if var_type_lower in col_name.lower():
                            return file
                except Exception:
                    continue
    else:
        for file in txt_files:
            filename = os.path.basename(file).lower()
            if dataset_name_lower in filename:
                if 'cmsaf' not in filename:
                    return file

    return None


def draw_scatter_subplot(ax, sst_trends, cwv_trends, title_text, x_min=None, x_max=None, y_min=None, y_max=None, 
                         xlabel=None, ylabel=None, col_idx=0, show_xticklabels=True):
    """Draw scatter with x marker on row-1 subplot
    
    Args:
        ax: matplotlib Axes
        sst_trends: x-axis data
        cwv_trends: y-axis data
        title_text: title text
        x_min, x_max, y_min, y_max: axis limits
        xlabel: x label (col_idx==1 only)
        ylabel: y label (col_idx==0 only)
        col_idx: Column index (0=col1, 1=col2, 2=col3, 3=col4)
        show_xticklabels: Whether to show x tick labels (default True)
    """
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
    
    # Scatter with x marker
    x = sst_trends
    y = cwv_trends
    
    # Filter NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Scatter points with x marker
    ax.scatter(x_valid, y_valid, marker='x', s=50, color='orange', alpha=0.6, zorder=2, linewidths=1.5)
    
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x[valid_mask], y[valid_mask])
        
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
               color='orange', linewidth=2, zorder=3)
        
        # Add statistics text (original format/position)
        rmse = np.sqrt(np.nanmean((x_valid * slope - y_valid)**2))
        stats_text = f'CC = {correlation:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        slope_text = f'Slope = {slope:.3f}\nIntercept = {intercept:.3f}'
        ax.text(0.35, 0.18, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Set axis labels on specified columns only
    if col_idx == 1 and xlabel:  # show xlabel on column 2 only
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel('')
    
    if col_idx == 0 and ylabel:  # show ylabel on column 1 only
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    if not show_xticklabels:
        ax.set_xticklabels([])
    
    # y tick labels on row 1, col 1 only
    if col_idx != 0:
        ax.set_yticklabels([])
    
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
    
    return None


def load_specific_cmip6_model(cmip6_dir, model_name, start_yyyymm=None, end_yyyymm=None, var_x='sst', var_y='cwv'):
    """Load one CMIP6 model-ensemble
    
    Args:
        cmip6_dir: CMIP6Data directory
        model_name: model name，e.g. 'CanESM5_r23i1p2f1'
        start_yyyymm: Start time
        end_yyyymm: End time
        var_x: x-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
    
    Returns:
        tuple: (x_anomaly, y_anomaly) or (None, None)
    """
    # Find matching file
    txt_files = glob.glob(os.path.join(cmip6_dir, "*.txt"))
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        if model_name in filename:
            x_anomaly, y_anomaly, _ = load_cmip6_data(filepath, start_yyyymm, end_yyyymm, var_x, var_y)
            if x_anomaly is not None and y_anomaly is not None:
                return x_anomaly, y_anomaly
    
    print(f"Model not found: {model_name}")
    return None, None


def extract_model_name_from_filename(filename):
    """Extract model name from filename (strip ensemble)
    
    Filename format: {model}_{ensemble}_s{start}_e{end}.txt
    Example: CanESM5_r23i1p2f1_s198001_e202412.txt -> ACCESS-CM2
    """
    basename = os.path.basename(filename)
    # Strip .txt suffix
    basename = basename.replace('.txt', '')
    # Model name before first _r
    match = re.match(r'^([^_]+(?:_[^_]+)*)_r\d+i\d+p\d+f\d+', basename)
    if match:
        return match.group(1)
    # If match fails, split on first underscore
    parts = basename.split('_')
    if len(parts) > 0:
        return parts[0]
    return None


def draw_fit_lines_only(ax, all_sst_trends, all_cwv_trends, title_text, y_min=None, y_max=None):
    """Draw fit lines only (no scatter) for multiple model-ensembles
    
    Args:
        ax: matplotlib Axes
        all_sst_trends: list of SST trend arrays per model-ensemble
        all_cwv_trends: list of CWV trend arrays per model-ensemble
        title_text: title text
        y_min, y_max: yaxis limits
    """
    x_min, x_max = -0.2, 0.5
    if y_min is None:
        y_min = -1
    if y_max is None:
        y_max = 5
    
    ax.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--', zorder=1)
    ax.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--', zorder=1)
    
    line_color = 'blue'
    
    for sst_trends, cwv_trends in zip(all_sst_trends, all_cwv_trends):
        if sst_trends is None or cwv_trends is None:
            continue
        
        # Filter NaN values
        valid_mask = ~(np.isnan(sst_trends) | np.isnan(cwv_trends))
        if np.sum(valid_mask) < 10:
            continue
        
        x_valid = sst_trends[valid_mask]
        y_valid = cwv_trends[valid_mask]
        
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x_valid, y_valid)
        if slope is None:
            continue
        
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
               color=line_color, linewidth=1.5, alpha=0.6, zorder=2)
    
    ax.text(0.05, 0.84, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xticks([0, 0.2, 0.4])
    

def draw_combined_scatter_plot(cmip6_dir, output_path):
    """Draw 4×4 subplots:
    - rows: 4 variable combinations（TCWV vs SST, TCWV vs TMT, TCWV vs TLT, TMT vs SST）
    - cols: 4 statistics panels
    """
    n_rows = 4
    n_cols = 4
    # Subplots; first 3 cols per row share y-axis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 2, 4 * n_rows + 1))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Share y-axis for first 3 cols per row
    for i in range(n_rows):
        for j in range(1, 3):
            axes[i, j].sharey(axes[i, 0])
    
    var_combos = [
        ('sst', 'cwv', 'TCWV vs SST'),
        ('tmt', 'cwv', 'TCWV vs TMT'),
        ('tlt', 'cwv', 'TCWV vs TLT'),
        ('sst', 'tmt', 'TMT vs SST'),
    ]
    
    # Variable combo labels above row 1
    # Add after all panels with precise positions

    def get_axis_limits(var_x, var_y):
        # xaxis limits
        if var_x == 'sst':
            x_min, x_max = -0.2, 0.5
        else:
            x_min, x_max = -0.3, 0.7

        # yaxis limits
        if var_y == 'cwv':
            y_min, y_max = -1.0, 5.0
        else:
            y_min, y_max = -0.3, 0.7

        return x_min, x_max, y_min, y_max

    # consistent with fig3:var datasets, colors (y), markers (x) for row 4 observational slopes
    var_datasets = {
        'cwv': ['rss', 'ustc'],
        'sst': ['cobe', 'ersst', 'hadisst', 'hadsst', 'oisst', 'ustc'],
        'tmt': ['rss', 'star', 'uah'],
        'tlt': ['rss', 'star', 'uah'],
    }
    var_colors = {
        'cwv': {'rss': 'blue', 'ustc': 'red'},
        'sst': {'cobe': 'blue', 'ersst': 'green', 'hadisst': 'orange', 'hadsst': 'purple', 'oisst': 'brown', 'ustc': 'magenta'},
        'tmt': {'rss': 'blue', 'star': 'red', 'uah': 'green'},
        'tlt': {'rss': 'blue', 'star': 'red', 'uah': 'green'},
    }
    var_markers = {
        'sst': {'cobe': 'o', 'ersst': 's', 'hadisst': '^', 'hadsst': 'v', 'oisst': 'D', 'ustc': 'p'},
        'tmt': {'rss': 'o', 'star': 's', 'uah': '^'},
        'tlt': {'rss': 'o', 'star': 's', 'uah': '^'},
        'cwv': {'rss': 'o', 'ustc': 's'},
    }
    dataset_display_names = {
        'rss': 'RSS',
        'ustc': 'USTC',
        'cobe': 'COBE',
        'ersst': 'ERSST',
        'hadisst': 'HADISST',
        'hadsst': 'HADSST',
        'oisst': 'OISST',
        'star': 'STAR',
        'uah': 'UAH',
    }

    print("Processing row 1, col 1: CMIP6 model-ensemble full-period trend scatter (200206-202412)")
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 1: {label}")
        x_trends, y_trends = process_data_for_period_single_trend(
            cmip6_dir, 200206, 202412, var_x, var_y,
            excluded_filenames=[
                'E3SM-1-1-ECA_r1i1p1f1_s198001_e202412.txt',
                'ACCESS-ESM1-5_r2i1p1f1_s198001_e202412.txt',
            ]
        )
        if x_trends is None:
            continue

        if i == 3:
            ind = (x_trends>0.2) & (y_trends>0.8)
            x_trends[ind] = np.nan
            y_trends[ind] = np.nan

        x_min, x_max, y_min, y_max = get_axis_limits(var_x, var_y)
        title = f"({chr(97 + i * 4)})"
        ylabel = get_var_label(var_y)
        draw_scatter_subplot(
            axes[i, 0], x_trends, y_trends, title,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            xlabel=None, ylabel=ylabel, col_idx=0, show_xticklabels=True
        )
    
    # Obs markers on row 1 (same combos as row 4)
    print("\nAdd observational multi-product markers on row 1 (consistent with row 4)...")
    obs_dir = "../data"

    for i, (var_x, var_y, label) in enumerate(var_combos):
        ax = axes[i, 0]
        start_yyyymm_obs = 200206
        end_yyyymm_obs = 202412

        x_datasets_obs = var_datasets.get(var_x, [])
        y_datasets_obs = var_datasets.get(var_y, [])

        for y_ds in y_datasets_obs:
            y_file = find_data_file(obs_dir, y_ds, var_type=var_y)
            if y_file is None:
                continue
            y_tr = load_obs_data_and_calc_trend(y_file, var_y, start_yyyymm_obs, end_yyyymm_obs)
            if y_tr is None or not np.isfinite(float(y_tr)):
                continue
            for x_ds in x_datasets_obs:
                x_file = find_data_file(obs_dir, x_ds, var_type=var_x)
                if x_file is None:
                    continue
                x_tr = load_obs_data_and_calc_trend(x_file, var_x, start_yyyymm_obs, end_yyyymm_obs)
                if x_tr is None or not np.isfinite(float(x_tr)):
                    continue
                y_colors_m = var_colors.get(var_y, {})
                x_markers_m = var_markers.get(var_x, {})
                edge_c = y_colors_m.get(y_ds, 'black')
                mk = x_markers_m.get(x_ds, 'o')
                ax.scatter(
                    x_tr, y_tr, marker=mk, s=OBS_MARKER_SIZE,
                    facecolors='none', edgecolors=edge_c, linewidths=OBS_MARKER_LW, zorder=5
                )
                print(
                    f"  Row {i+1}row：observational combo x={x_ds}({var_x}), y={y_ds}({var_y}) "
                    f"-> ({float(x_tr):.4f}, {float(y_tr):.4f})"
                )

    print("\nProcessing row 2, col 2: CanESM5_r23i1p2f1 (200206-202412) Monte Carlo scatter")
    single_model = 'CanESM5_r23i1p2f1'
    last_subplot_c = None  # Save last subplot pcolormesh object
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 2: {label}")
        x_anom, y_anom = load_specific_cmip6_model(
            cmip6_dir, single_model, 200206, 202412, var_x=var_x, var_y=var_y
        )
        if x_anom is None or y_anom is None:
            continue

        # Monte Carlo: row-1 param for y, row-2 for x
        y_trends, x_trends = monte_carlo_trend_analysis(y_anom, x_anom, n_samples=3000, min_period=120)
        if len(x_trends) == 0:
            continue

        x_min, x_max, y_min, y_max = get_axis_limits(var_x, var_y)
        title = f"({chr(97 + i * 4 + 1)})"
        xlabel = get_var_label(var_x)
        ylabel = get_var_label(var_y)
        c = draw_single_subplot(
            axes[i, 1], x_trends, y_trends, title,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            xlabel=xlabel, ylabel=ylabel, col_idx=1, show_xticklabels=True
        )
        
        if i == 3 and c is not None:
            last_subplot_c = c
    
    if last_subplot_c is not None:
        ax_last = axes[3, 1]
        bbox = ax_last.get_position()
        # Colorbar below panel via fig.add_axes
        cbar_width = bbox.width
        cbar_height = 0.008  # Thinner colorbar height
        cbar_x = bbox.x0 - 0.06  # Shift slightly left
        cbar_y = bbox.y0 - 0.07  # Below panel with margin
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(last_subplot_c, cax=cbar_ax, orientation='horizontal', extend='max')
        cbar.ax.tick_params(labelsize=14)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontweight('bold')
        cbar.ax.text(1.06, 0.5, 'Count', transform=cbar.ax.transAxes, 
                     fontsize=14, fontweight='bold', va='center', ha='left')

    print("\nProcessing row 3, col 3: all CMIP6 models, 200206-202412 Monte Carlo fit lines")

    # Use cached Monte Carlo stats (no recompute here)
    mc_stats = load_or_compute_cmip6_mc_stats(cmip6_dir, var_combos,
                                              start_yyyymm=200206,
                                              end_yyyymm=202412,
                                              n_samples=3000,
                                              min_period=120)
    slopes_per_combo = mc_stats["slopes_per_combo"]
    intercepts_per_combo = mc_stats["intercepts_per_combo"]
    correlations_per_combo = mc_stats["correlations_per_combo"]
    model_names_per_combo = mc_stats["model_names_per_combo"]
    filepaths_per_combo = mc_stats["filepaths_per_combo"]
    x_trends_list_per_combo = mc_stats["x_trends_list_per_combo"]
    y_trends_list_per_combo = mc_stats["y_trends_list_per_combo"]

    for i, (var_x, var_y, label) in enumerate(var_combos):
        slopes = np.array(slopes_per_combo[i])
        intercepts = np.array(intercepts_per_combo[i])
        correlations = np.array(correlations_per_combo[i])
        if slopes.size == 0:
            continue

        # Keep model-ensembles with CC >= 0.95;Rows 3–4 also exclude CMIP6_EXCLUDED_FOR_ROW3_ROW4  model-ensembles
        high_cc_mask = correlations >= 0.95
        filepaths = filepaths_per_combo[i]
        not_excluded = np.array([os.path.basename(filepaths[k]) not in CMIP6_EXCLUDED_FOR_ROW3_ROW4 for k in range(len(filepaths))])
        combined_mask = high_cc_mask & not_excluded
        slopes_filtered = slopes[combined_mask]
        intercepts_filtered = intercepts[combined_mask]
        x_trends_list = x_trends_list_per_combo[i]
        y_trends_list = y_trends_list_per_combo[i]
        
        # Filter x/y_trends lists by CC and exclusion list
        x_trends_list_filtered = [x_trends_list[k] for k in range(len(x_trends_list)) if combined_mask[k]]
        y_trends_list_filtered = [y_trends_list[k] for k in range(len(y_trends_list)) if combined_mask[k]]

        if slopes_filtered.size == 0:
            continue

        x_min, x_max, y_min, y_max = get_axis_limits(var_x, var_y)
        x_vals = np.array([x_min, x_max])
        ax = axes[i, 2]

        # Gray fit lines for filtered ensembles (CC>=0.95; row 3/4 exclusions) CMIP6_EXCLUDED_FOR_ROW3_ROW4)
        for slope, intercept in zip(slopes_filtered, intercepts_filtered):
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color="gray", linewidth=1.0, alpha=0.6, zorder=2)

        # Red line: median slope/intercept of gray lines
        # CC/RMSE from filtered MC samples; RMSE vs median line
        if slopes_filtered.size > 0 and intercepts_filtered.size > 0 and len(x_trends_list_filtered) > 0:
            slope_total = np.nanmedian(slopes_filtered)
            intercept_total = np.nanmedian(intercepts_filtered)
            y_vals_total = slope_total * x_vals + intercept_total
            ax.plot(x_vals, y_vals_total, color='red', linewidth=2, zorder=3)

            # Merge MC samples (CC>=0.95) for CC/RMSE
            all_x_trends = np.concatenate(x_trends_list_filtered) if len(x_trends_list_filtered) > 0 else np.array([])
            all_y_trends = np.concatenate(y_trends_list_filtered) if len(y_trends_list_filtered) > 0 else np.array([])
            if len(all_x_trends) > 10:
                valid_mask = ~(np.isnan(all_x_trends) | np.isnan(all_y_trends))
                x_valid = all_x_trends[valid_mask]
                y_valid = all_y_trends[valid_mask]
                if len(x_valid) > 10:
                    correlation_total = np.corrcoef(x_valid, y_valid)[0, 1] if len(x_valid) > 1 else np.nan
                    rmse_total = np.sqrt(np.nanmean((x_valid * slope_total + intercept_total - y_valid)**2))

                    stats_text = f'CC = {correlation_total:.3f}\nRMSE = {rmse_total:.3f}'
                    ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=4)

            slope_text = f'Slope = {slope_total:.3f}\nIntercept = {intercept_total:.3f}'
            ax.text(0.35, 0.18, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=4)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--', zorder=1)
        ax.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--', zorder=1)
        
        # Panel number top-left (no combo label)
        ax.text(0.05, 0.95, f"({chr(97 + i * 4 + 2)})",  transform=ax.transAxes, 
                fontsize=15, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels([])
        
        if x_max <= 0.5:
            ax.set_xticks([0, 0.2, 0.4])
        else:
            ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
        
        # Set y ticks from y-axis limits
        if y_max <= 0.7:
            ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
        else:
            ax.set_yticks([-1, 0, 1, 2, 3, 4])

    

    print("\nProcessing row 4, col 4: slope distributions from row 3 (weighted histogram + KDE)")
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 4: {label} weighted slope histogram")
        slopes = np.array(slopes_per_combo[i])
        correlations = np.array(correlations_per_combo[i])
        filepaths = filepaths_per_combo[i]
        
        if slopes.size == 0:
            continue

        # Keep model-ensembles with CC >= 0.95(consistent with row 4)
        high_cc_mask = correlations >= 0.95
        not_excluded = np.array([os.path.basename(filepaths[k]) not in CMIP6_EXCLUDED_FOR_ROW3_ROW4 for k in range(len(filepaths))])
        combined_mask = high_cc_mask & not_excluded
        slopes_filtered = slopes[combined_mask]
        filepaths_filtered = [filepaths[i] for i in range(len(filepaths)) if combined_mask[i]]
        
        if slopes_filtered.size == 0:
            continue

        # Weights: count realizations per model
        # weight = 1/#A_m, #A_m = realizations for model m
        model_realization_count = {}
        for filepath in filepaths_filtered:
            model_name = extract_model_name_from_filename(filepath)
            if model_name:
                if model_name not in model_realization_count:
                    model_realization_count[model_name] = 0
                model_realization_count[model_name] += 1
        
        # Compute weight per slope
        weights = []
        for filepath in filepaths_filtered:
            model_name = extract_model_name_from_filename(filepath)
            if model_name and model_name in model_realization_count:
                # weight = 1/#A_m
                weight = 1.0 / model_realization_count[model_name]
                weights.append(weight)
            else:
                weights.append(1.0)  # If model unknown, weight 1
        
        weights = np.array(weights)
        slopes = slopes_filtered  # use filtered slopes
        
        # Set x range by row index; filter data
        x_ranges = [
            (5, 15.5),
            (2.5, 11),
            (2.5, 13.5),
            (0.6, 2.2),
        ]
        x_min, x_max = x_ranges[i]
        
        # Slopes within specified x range only
        mask = (slopes >= x_min) & (slopes <= x_max)
        slopes_filtered_range = slopes[mask]
        weights_filtered_range = weights[mask]
        
        if len(slopes_filtered_range) == 0:
            continue
        
        ax_hist = axes[i, 3]
        
        # Histogram bins from x limits, not data min/max
        # rwidth=0.7 for gaps between bars
        n, bins, patches = ax_hist.hist(
            slopes_filtered_range, bins=20, weights=weights_filtered_range, density=True,
            range=(x_min, x_max), histtype='bar', rwidth=0.6,
            facecolor='none', edgecolor='orange', linewidth=1.5
        )

        q1_val = np.nanpercentile(slopes_filtered_range, 25) if slopes_filtered_range.size > 0 else np.nan
        median_val = np.nanmedian(slopes_filtered_range) if slopes_filtered_range.size > 0 else np.nan
        q3_val = np.nanpercentile(slopes_filtered_range, 75) if slopes_filtered_range.size > 0 else np.nan
        
        y_hist_max = np.max(n) if len(n) > 0 else 0
        
        # KDE (weighted data)
        # Repeat points by weight for KDE
        # Weighted KDE N/A in scipy gaussian_kde
        # Approximate via repeated points
        # Normalize weights to integers and repeat points
        y_kde_max = 0  # store KDE curve maximum
        if len(weights_filtered_range) > 0 and weights_filtered_range.min() > 0:
            weights_normalized = (weights_filtered_range / weights_filtered_range.min() * 10).astype(int)  # scale weights to integers
            slopes_for_kde = np.repeat(slopes_filtered_range, weights_normalized)
            
            if len(slopes_for_kde) > 1:
                try:
                    kde = gaussian_kde(slopes_for_kde)
                    # Larger bandwidth for smoother KDE(default factor 1.0; 1.5–2.0 smoother)
                    kde.set_bandwidth(kde.factor * 2.5)
                    # More KDE points for smoother curve
                    x_kde = np.linspace(x_min, x_max, 500)
                    y_kde = kde(x_kde)
                    y_kde_max = np.max(y_kde)  # KDE curve maximum
                    ax_hist.plot(x_kde, y_kde, color='orange', linewidth=3)
                except:
                    pass  # if KDE fails, skip
        
        ax_hist.set_xlim(x_min, x_max)

        # See fig3 row 4:x/y product combos per var_datasets;Slopes via load_obs_data_and_calc_trend (full-period linregress) as y/x
        dash_line_spacing_ratio = 0.06
        var_x, var_y, _ = var_combos[i]
        start_yyyymm_obs = 200206
        end_yyyymm_obs = 202412

        x_datasets_obs = var_datasets.get(var_x, [])
        y_datasets_obs = var_datasets.get(var_y, [])
        obs_slopes = []
        obs_colors_list = []
        obs_markers_list = []

        for y_ds in y_datasets_obs:
            y_file = find_data_file(obs_dir, y_ds, var_type=var_y)
            if y_file is None:
                continue
            y_tr = load_obs_data_and_calc_trend(y_file, var_y, start_yyyymm_obs, end_yyyymm_obs)
            if y_tr is None or not np.isfinite(float(y_tr)):
                continue
            for x_ds in x_datasets_obs:
                x_file = find_data_file(obs_dir, x_ds, var_type=var_x)
                if x_file is None:
                    continue
                x_tr = load_obs_data_and_calc_trend(x_file, var_x, start_yyyymm_obs, end_yyyymm_obs)
                if x_tr is None or not np.isfinite(float(x_tr)) or float(x_tr) == 0:
                    continue
                slope_obs = y_tr / x_tr
                if x_min <= slope_obs <= x_max:
                    obs_slopes.append(slope_obs)
                    y_colors_m = var_colors.get(var_y, {})
                    x_markers_m = var_markers.get(var_x, {})
                    obs_colors_list.append(y_colors_m.get(y_ds, 'black'))
                    obs_markers_list.append(x_markers_m.get(x_ds, 'o'))

        y_vis = max(y_kde_max, y_hist_max) if max(y_kde_max, y_hist_max) > 0 else ax_hist.get_ylim()[1]
        if len(obs_slopes) > 0 and y_vis > 0:
            color_to_points = {}
            color_order = []
            for slope, color, marker in zip(obs_slopes, obs_colors_list, obs_markers_list):
                key = color if isinstance(color, str) else tuple(np.atleast_1d(color))
                if key not in color_to_points:
                    color_to_points[key] = []
                    color_order.append(key)
                color_to_points[key].append((slope, marker))
            y_ref = y_vis / 2.0
            spacing = y_vis * dash_line_spacing_ratio
            for idx, key in enumerate(color_order):
                y_dash = y_ref - idx * spacing
                ax_hist.axhline(y_dash, color="grey", linestyle='--', linewidth=1.5, zorder=2)
                for slope, marker in color_to_points[key]:
                    ax_hist.scatter(slope, y_dash, facecolors='none', edgecolors=key,
                                    marker=marker, s=OBS_MARKER_SIZE, linewidths=OBS_MARKER_LW, zorder=3)

        ax_hist.axvline(median_val, color='purple', linestyle='--', linewidth=2, )
        ax_hist.axvline(q1_val, color='m', linestyle='--', linewidth=1.8)
        ax_hist.axvline(q3_val, color='m', linestyle='--', linewidth=1.8)
        
        # Median label right of red line with units
        y_max_hist = ax_hist.get_ylim()[1]
        if i < 3:
            unit = " %/K"
        else:
            unit = ""
        ax_hist.text(median_val + (x_max - x_min) * 0.02, y_max_hist * 0.9, 
                    f'{median_val:.2f}{unit}', 
                    fontsize=14, fontweight='bold', color='purple',
                    verticalalignment='top')
        ax_hist.text(q1_val - (x_max - x_min) * 0.02, y_max_hist * 0.82,
                    f'{q1_val:.2f}{unit}',
                    fontsize=12, fontweight='bold', color='m',
                    verticalalignment='top', horizontalalignment='right')
        ax_hist.text(q3_val + (x_max - x_min) * 0.02, y_max_hist * 0.74,
                    f'{q3_val:.2f}{unit}',
                    fontsize=12, fontweight='bold', color='m',
                    verticalalignment='top')

        # Col 4: show ylabel and xlabel on all subplots
        ax_hist.set_ylabel('Possibility density', fontsize=16, fontweight='bold')
        # First three panels: append unit (%/K) to Slope
        if i < 3:
            ax_hist.set_xlabel('Slope (%/K)', fontsize=16, fontweight='bold')
        else:
            ax_hist.set_xlabel('Slope', fontsize=16, fontweight='bold')
        ax_hist.tick_params(axis='both', which='major', labelsize=16)
        for tick in ax_hist.get_xticklabels() + ax_hist.get_yticklabels():
            tick.set_fontweight('bold')

        # Panel number top-left (no combo label)
        ax_hist.text(0.05, 0.95, f"({chr(97 + i * 4 + 3)})", transform=ax_hist.transAxes, 
                     fontsize=14, fontweight='bold', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
        

    # Right margin for row 4 legend(consistent with fig3)
    plt.subplots_adjust(left=0.08, right=0.84, top=0.98, bottom=0.09, wspace=0.1, hspace=0.23)
    
    if n_cols >= 4:        
        for i in range(n_rows):
            ax = axes[i, 3]
            bbox = ax.get_position()
            new_x0 = bbox.x0 + 0.04
            new_x1 = bbox.x1 + 0.04
            ax.set_position([new_x0, bbox.y0, new_x1 - new_x0, bbox.height])
    
    ax_right = axes[0, 3]
    bbox_right = ax_right.get_position()
    right_x = bbox_right.x1
    legend_x_start = right_x + 0.02
    legend_line_height = 0.012
    for i, (var_x, var_y, label) in enumerate(var_combos):
        ax_row = axes[i, 0]
        bbox_row = ax_row.get_position()
        y_colors = var_colors.get(var_y, {})
        x_markers_dict = var_markers.get(var_x, {})
        x_datasets = var_datasets.get(var_x, [])
        y_datasets = var_datasets.get(var_y, [])
        n_color_items = sum(1 for y_ds in y_datasets if y_ds in y_colors)
        color_block_height = (1 + n_color_items) * legend_line_height
        gap_between_blocks = 0.008
        color_row_y = bbox_row.y1 - 0.018
        shape_row_y = color_row_y - color_block_height - gap_between_blocks
        current_x = legend_x_start
        y_var_label = get_var_label(var_y).replace(' Trend (%/decade)', '').replace(' Trend (K/decade)', '')
        fig.text(current_x, color_row_y, f'{y_var_label} (color)',
                 ha='left', va='center', fontsize=13, fontweight='bold')
        color_idx = 0
        for y_ds in y_datasets:
            if y_ds in y_colors:
                color = y_colors[y_ds]
                ds_name = dataset_display_names.get(y_ds, y_ds.upper())
                item_y = color_row_y - (color_idx + 1) * legend_line_height
                fig.text(current_x, item_y, '■',
                         ha='left', va='center', fontsize=14, color=color)
                fig.text(current_x + 0.012, item_y, ds_name,
                         ha='left', va='center', fontsize=12)
                color_idx += 1
        x_var_label = get_var_label(var_x).replace(' Trend (%/decade)', '').replace(' Trend (K/decade)', '')
        fig.text(current_x, shape_row_y, f'{x_var_label} (shape)',
                 ha='left', va='center', fontsize=13, fontweight='bold')
        shape_idx = 0
        for x_ds in x_datasets:
            if x_ds in x_markers_dict:
                marker = x_markers_dict[x_ds]
                ds_name = dataset_display_names.get(x_ds, x_ds.upper())
                item_y = shape_row_y - (shape_idx + 1) * legend_line_height
                temp_ax = fig.add_axes([current_x, item_y - 0.006, 0.012, 0.012])
                temp_ax.scatter([0.5], [0.5], marker=marker, s=80, facecolors='none',
                                edgecolors='black', linewidths=0.8)
                temp_ax.set_xlim(0, 1)
                temp_ax.set_ylim(0, 1)
                temp_ax.axis('off')
                fig.text(current_x + 0.012, item_y, ds_name,
                         ha='left', va='center', fontsize=12)
                shape_idx += 1
    
    # Add variable-pair row titles left of col 1
    for i, (var_x, var_y, label) in enumerate(var_combos):
        # Get row i col 1 subplot position
        ax = axes[i, 0]
        # Get bbox after layout
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.03, bbox.y0 + bbox.height / 2, label, 
                ha='right', va='center', fontsize=16, fontweight='bold', rotation=90)
    
    # Save without bbox_inches=tight to preserve subplots_adjust
    save_figure_png_pdf(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()  # Close figure; avoid plt.show() on non-GUI backend


def main():
    # Data directory
    cmip6_dir = "../data/cmip6"
    out_dir = "../plot"
    
    # Output path
    output_path = os.path.join(out_dir, "fig2.png")
    
    # Plot combined scatter figure(CMIP6 data only)
    draw_combined_scatter_plot(cmip6_dir, output_path)


if __name__ == "__main__":
    main()
