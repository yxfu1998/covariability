import numpy as np  
from scipy import odr
from scipy.stats import linregress, gaussian_kde
import random
import matplotlib
matplotlib.use('Agg')  # non-GUI backend to avoid Qt issues in WSL
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
        Return format matches linregress for API compatibility
    """
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 2:
        return None, None, None, None, None
    
    def linear_model(B, x):
        """Linear model: y = B[0] * x + B[1]"""
        return B[0] * x + B[1]
    
    model = odr.Model(linear_model)
    
    # Optional sx/sy if x and y have errors
    # Assume equal unit weights for x and y
    data = odr.Data(x_valid, y_valid)
    
    # Create ODR object
    # Initial values from least squares
    initial_slope = np.sum((x_valid - np.mean(x_valid)) * (y_valid - np.mean(y_valid))) / np.sum((x_valid - np.mean(x_valid))**2)
    initial_intercept = np.mean(y_valid) - initial_slope * np.mean(x_valid)
    odr_obj = odr.ODR(data, model, beta0=[initial_slope, initial_intercept])
    
    # runODR
    output = odr_obj.run()
    
    slope = output.beta[0]
    intercept = output.beta[1]
    
    # Computecorrelation coefficient(use raw data)
    if len(x_valid) > 1:
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
    else:
        correlation = 0.0
    
    # ODR has no p-value; use standard errors
    # std_err from covariance matrix
    std_err_slope = output.sd_beta[0]
    std_err_intercept = output.sd_beta[1]
    
    # Return std_err as slope SE for compatibility
    std_err = std_err_slope
    
    # p_value None (ODR does not provide)
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


# Rows 3-4 panels skip  CMIP6 ensemble-model(only in these rows; other rows still use)
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

OBS_MARKER_SIZE = 180
OBS_MARKER_LW = 1.5


def load_cmip6_data(filepath, start_yyyymm=None, end_yyyymm=None, var_x='sst', var_y='cwv', row_index=None):
    """Read TXT output from cal_cmip_trend.py (space-separated)
    
    File format:time(yyyymm) ts_cwv ts_sst ts_tlt ts_tmt
    
    Args:
        filepath: filepath
        start_yyyymm: start time (YYYYMM, e.g. 198001)
        end_yyyymm: end time (YYYYMM, e.g. 202412)
        var_x: x-axis variable，'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable，'sst', 'tmt', 'tlt', 'cwv'
        row_index: row index 0-4; skip list only for rows 3-4 (index 2 or 3) CMIP6_EXCLUDED_FOR_ROW3_ROW4 in file。
    
    Returns:
        tuple: (x_anomaly, y_anomaly, years) or (None, None, None)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None, None
    
    try:
        # Only rows 3-4 panels skip listed ensembles
        # if row_index in (2, 3):
        #     if os.path.basename(filepath) in CMIP6_EXCLUDED_FOR_ROW3_ROW4:
        #         return None, None, None

        data = np.loadtxt(filepath, skiprows=1)

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
        
        # Convert CWV to percent(if y is CWV)
        if var_y.lower() == 'cwv':
            y_anomaly = y_anomaly * 100. / 41.
        
        # Extract years for return (from yyyymm)
        years = (yyyymm // 100).astype(float)
        
        return x_anomaly, y_anomaly, years
        
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None, None


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
        
        # Compute linear trend - useODR
        cwv_trend = linregress(time_points, cwv_segment)[0]
        sst_trend = linregress(time_points, sst_segment)[0]
            
        cwv_trends.append(cwv_trend)
        sst_trends.append(sst_trend)
    
    return np.array(cwv_trends), np.array(sst_trends)


def load_or_compute_cmip6_mc_stats(cmip6_dir, var_combos, start_yyyymm=198001, end_yyyymm=202412,
                                   n_samples=3000, min_period=120):
    """Load or compute CMIP6 Monte Carlo statistics (for rows 4–5)

    Returns dict with:
        - slopes_per_combo: [num_combos][num_models] slope
        - intercepts_per_combo: same for intercept
        - correlations_per_combo: same for correlation
        - model_names_per_combo: same, model-ensemble names
        - filepaths_per_combo: same as above, filepath
        - x_trends_list_per_combo: [num_combos][num_models] eachmodel-ensemble x_trendsarray list
        - y_trends_list_per_combo: [num_combos][num_models] eachmodel-ensemble y_trendsarray list
    """
    cache_path = os.path.join("../data", "cmip_monte_carlo_1980-2024.npy")

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
                print("Cache format mismatch; recomputing")
        except Exception as e:
            print(f"Cache read error; recomputing: {e}")

    print("Computing CMIP6 Monte Carlo stats (may take a while)...")
    txt_files = glob.glob(os.path.join(cmip6_dir, "*.txt"))

    slopes_per_combo = [[] for _ in range(len(var_combos))]
    intercepts_per_combo = [[] for _ in range(len(var_combos))]
    correlations_per_combo = [[] for _ in range(len(var_combos))]
    model_names_per_combo = [[] for _ in range(len(var_combos))]
    filepaths_per_combo = [[] for _ in range(len(var_combos))]
    x_trends_list_per_combo = [[] for _ in range(len(var_combos))]  # eachmodel-ensemble x_trendslist
    y_trends_list_per_combo = [[] for _ in range(len(var_combos))]  # eachmodel-ensemble y_trendslist

    for j, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Computing combo {j+1}/{len(var_combos)}: {label}")

        for filepath in txt_files:
            model_name = os.path.basename(filepath).replace('.txt', '')
            x_anom, y_anom, _ = load_cmip6_data(filepath, start_yyyymm, end_yyyymm, var_x, var_y, row_index=j)
            if x_anom is None or y_anom is None:
                continue
    
            # Monte Carlo:Monte Carlo random sub-period trends for model-ensemble and variable combo
            y_trends, x_trends = monte_carlo_trend_analysis(y_anom, x_anom,
                                                            n_samples=n_samples,
                                                            min_period=min_period)
            if len(x_trends) == 0:
                continue
            
            valid_mask = ~(np.isnan(x_trends) | np.isnan(y_trends))
            if np.sum(valid_mask) >= 10:
                x_trends_list_per_combo[j].append(x_trends[valid_mask])
                y_trends_list_per_combo[j].append(y_trends[valid_mask])

            # ODR fit on Monte Carlo samples,for that model-ensemble and combo 
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
        print(f"Save cachefileerror: {e}")

    return stats

def process_data_for_period_single_trend(data_dir, start_yyyymm, end_yyyymm, var_x='sst', var_y='cwv',
                                         excluded_filenames=None):
    """Process CMIP6 period: one trend per model-ensemble
    
    Args:
        data_dir: datadirectory
        start_yyyymm: start time
        end_yyyymm: end time
        var_x: x-axis variable，'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable，'sst', 'tmt', 'tlt', 'cwv'
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
        
        # Time axis in decades
        time_points = np.arange(len(x_anomaly)) / 120.0
        
        if np.any(np.isnan(x_anomaly)) or np.any(np.isnan(y_anomaly)):
            continue
        
        # Compute linear trend(full time period)- useODR
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
        col_idx: col index (0=col1, 1=col2, 2=col3, 3=col4)
        show_xticklabels: show x tick labels (default True)
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
    
    H, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[x_min, x_max], [y_min, y_max]])
    H_masked = np.ma.masked_where(H == 0, H)
    
    c = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='jet', shading='auto', vmin=0, vmax=20, zorder=2)
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x[valid_mask], y[valid_mask])
        
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
               color='red', linewidth=2, zorder=3)
        
        # Add statistics text(original format and position)
        rmse = np.sqrt(np.nanmean((x[valid_mask] * slope + intercept - y[valid_mask])**2))
        stats_text = f'CC = {correlation:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        slope_text = f'Slope = {slope:.3f}\nIntercept = {intercept:.3f}'
        ax.text(0.35, 0.18, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Tick font size and bold
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Axis labels on specified columns only
    if col_idx == 1 and xlabel:  # Show xlabel on row 2 col only
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel('')
    
    if col_idx == 0 and ylabel:  # Show ylabel on row 1 col only
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    # Show x tick labels on all panels
    if not show_xticklabels:
        ax.set_xticklabels([])
    
    if col_idx != 0:
        ax.set_yticklabels([])
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Set x ticks from x limits(consistent with row 3)
    if x_max <= 0.5:
        ax.set_xticks([0, 0.2, 0.4])
    else:
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    
    return c

def get_var_label(var_name):
    """Axis/legend label from variable name"""
    label_map = {
        'sst': 'SST Trend (K/decade)',
        'tmt': 'TMT Trend (K/decade)',
        'tlt': 'TLT Trend (K/decade)',
        'cwv': 'TCWV Trend (%/decade)',
    }
    return label_map.get(var_name, f'{var_name.upper()} Trend')


def load_obs_data_and_calc_trend(filepath, var_name, start_yyyymm=198001, end_yyyymm=202412):
    """Read observational file and compute trend
    
    Args:
        filepath: datafilepath
        var_name: variable name ('cwv', 'sst', 'tmt', 'tlt')
        start_yyyymm: start time（YYYYMM format）
        end_yyyymm: end time（YYYYMM format）
    
    Returns:
        trend: trend (K/decade or %/decade); None on failure
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        
        time = data[:, 0].astype(int)  # YYYYMM format
        
        with open(filepath, 'r') as f:
            header = f.readline().strip()
        
        column_names = header.split()
        
        # Find data column by variable and header
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
        
        # if CWV, convert to percent
        if var_name.lower() == 'cwv':
            var_anomaly = var_anomaly * 100. / 41.
        
        # Time axis in decades
        time_points = np.arange(len(var_anomaly)) / 120.0
        
        valid_mask = ~np.isnan(var_anomaly)
        if np.sum(valid_mask) < 120:
            print(f"Insufficient valid points: {np.sum(valid_mask)}")
            return None
        
        var_anomaly_valid = var_anomaly[valid_mask]
        time_points_valid = time_points[valid_mask]
        
        trend = linregress(time_points_valid, var_anomaly_valid)[0]
        
        return trend
        
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def draw_scatter_subplot(ax, sst_trends, cwv_trends, title_text, x_min=None, x_max=None, y_min=None, y_max=None, 
                         xlabel=None, ylabel=None, col_idx=0, show_xticklabels=True):
    """Draw scatter with x marker on first subplot
    
    Args:
        ax: matplotlib Axes
        sst_trends: x-axis data
        cwv_trends: y-axis data
        title_text: title text
        x_min, x_max, y_min, y_max: axis limits
        xlabel: x label (col_idx==1 only)
        ylabel: y label (col_idx==0 only)
        col_idx: col index (0=col1, 1=col2, 2=col3, 3=col4)
        show_xticklabels: show x tick labels (default True)
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
    
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Scatter points with x marker
    ax.scatter(x_valid, y_valid, marker='x', s=50, color='orange', alpha=0.6, zorder=2, linewidths=1.5)
    
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x[valid_mask], y[valid_mask])
        
        ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept], 
               color='red', linewidth=2, zorder=3)
        
        # Add statistics text(original format and position)
        rmse = np.sqrt(np.nanmean((x_valid * slope - y_valid)**2))
        stats_text = f'CC = {correlation:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        slope_text = f'Slope = {slope:.3f}\nIntercept = {intercept:.3f}'
        ax.text(0.35, 0.18, slope_text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=15, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Tick font size and bold
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # Axis labels on specified columns only
    if col_idx == 1 and xlabel:  # Show xlabel on row 2 col only
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel('')
    
    if col_idx == 0 and ylabel:  # Show ylabel on row 1 col only
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    # Show x tick labels on all panels
    if not show_xticklabels:
        ax.set_xticklabels([])
    
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
        cmip6_dir: CMIP6datadirectory
        model_name: model name，e.g. 'CanESM5_r23i1p2f1'
        start_yyyymm: start time
        end_yyyymm: end time
        var_x: x-axis variable，'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable，'sst', 'tmt', 'tlt', 'cwv'
    
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
    e.g. CanESM5_r23i1p2f1_s198001_e202412.txt -> ACCESS-CM2
    """
    basename = os.path.basename(filename)
    # Strip .txt suffix
    basename = basename.replace('.txt', '')
    # matchmodel name(before first _r)
    match = re.match(r'^([^_]+(?:_[^_]+)*)_r\d+i\d+p\d+f\d+', basename)
    if match:
        return match.group(1)
    # If match fails, split at first _r
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
    - row：four variable combinations（TCWV vs SST, TCWV vs TMT, TCWV vs TLT, TMT vs SST）
    - col：four statistics panels
    """
    n_rows = 4
    n_cols = 4
    # Create panels; first 3 cols per row share y
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 2, 4 * n_rows + 1))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Share y for first 3 cols each row
    for i in range(n_rows):
        for j in range(1, 3):
            axes[i, j].sharey(axes[i, 0])
    
    # (var_x, var_y, label)
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

    print("ProcessingRow1col：allCMIP6model-memberfull-periodtrendscatter (198001-202412)")
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row{i+1}row Row1col：{label}")
        x_trends, y_trends = process_data_for_period_single_trend(
            cmip6_dir, 198001, 202412, var_x, var_y,
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
        # Index only; remove combo labels(combo labels left of row 1 col 1)
        title = f"({chr(97 + i * 4)})"
        # Col 1: ylabel, no xlabel, show x ticks
        ylabel = get_var_label(var_y)
        draw_scatter_subplot(
            axes[i, 0], x_trends, y_trends, title,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            xlabel=None, ylabel=ylabel, col_idx=0, show_xticklabels=True
        )
    
    # Obs markers on row 1: square STAR/CDR, diamond ERA5, circle RSS/OISST
    print("\nAdd obs markers on row 1 col 1...")
    obs_dir = "../data"
    
    # Read obs data and compute trends
    cwv_trend = load_obs_data_and_calc_trend(
        os.path.join(obs_dir, "cdr_s198001_e202412.txt"), 'cwv', 198001, 202412
    )
    print(cwv_trend)
    sst_trend = load_obs_data_and_calc_trend(
        os.path.join(obs_dir, "cdr_s198001_e202412.txt"), 'sst', 198001, 202412
    )
    print(sst_trend)
    tmt_trend = load_obs_data_and_calc_trend(
        os.path.join(obs_dir, "star_tmt_s198101_e202412.txt"), 'tmt', 198001, 202412
    )
    print(tmt_trend)
    tlt_trend = load_obs_data_and_calc_trend(
        os.path.join(obs_dir, "star_tlt_s198101_e202412.txt"), 'tlt', 198001, 202412
    )
    print(tlt_trend)
    
    # Add obs point per panel variable combo (square)
    for i, (var_x, var_y, label) in enumerate(var_combos):
        ax = axes[i, 0]
        
        # x and y trends from variable combo
        if var_x == 'sst':
            x_trend_obs = sst_trend
        elif var_x == 'tmt':
            x_trend_obs = tmt_trend
        elif var_x == 'tlt':
            x_trend_obs = tlt_trend
        else:
            x_trend_obs = None
        
        if var_y == 'cwv':
            y_trend_obs = cwv_trend
        elif var_y == 'tmt':
            y_trend_obs = tmt_trend
        elif var_y == 'tlt':
            y_trend_obs = tlt_trend
        else:
            y_trend_obs = None
        
        # If both trends valid, print only (no symbols on row 1 panel)
        if x_trend_obs is not None and y_trend_obs is not None and not np.isnan(x_trend_obs) and not np.isnan(y_trend_obs):
            print(f"  Row{i+1}row：observational point ({var_x}={x_trend_obs:.4f}, {var_y}={y_trend_obs:.4f})")
        else:
            print(f"  Row{i+1}row：Cannot add observational point (x_trend={x_trend_obs}, y_trend={y_trend_obs})")
    
    # Read ERA5 trends; black diamond markers on row 1
    print("\nAdd ERA5 markers on row 1...")
    era5_file = os.path.join(obs_dir, "era5_s198001_e202412.txt")
    
    # Read ERA5 data; trends per variable 198001-202412
    era5_cwv_trend = load_obs_data_and_calc_trend(era5_file, 'cwv', 198001, 202412)
    era5_sst_trend = load_obs_data_and_calc_trend(era5_file, 'sst', 198001, 202412)
    era5_tmt_trend = load_obs_data_and_calc_trend(era5_file, 'tmt', 198001, 202412)
    era5_tlt_trend = load_obs_data_and_calc_trend(era5_file, 'tlt', 198001, 202412)
    
    print(f"ERA5trend: cwv={era5_cwv_trend}, sst={era5_sst_trend}, tmt={era5_tmt_trend}, tlt={era5_tlt_trend}")
    
    # Add ERA5 point per panel combo (diamond)
    for i, (var_x, var_y, label) in enumerate(var_combos):
        ax = axes[i, 0]
        
        # x and y trends from variable combo
        if var_x == 'sst':
            x_trend_era5 = era5_sst_trend
        elif var_x == 'tmt':
            x_trend_era5 = era5_tmt_trend
        elif var_x == 'tlt':
            x_trend_era5 = era5_tlt_trend
        else:
            x_trend_era5 = None
        
        if var_y == 'cwv':
            y_trend_era5 = era5_cwv_trend
        elif var_y == 'tmt':
            y_trend_era5 = era5_tmt_trend
        elif var_y == 'tlt':
            y_trend_era5 = era5_tlt_trend
        else:
            y_trend_era5 = None
        
        # If both trends valid, print only (no symbols on row 1 panel)
        if x_trend_era5 is not None and y_trend_era5 is not None and not np.isnan(x_trend_era5) and not np.isnan(y_trend_era5):
            print(f"  Row{i+1}row：ERA5 point ({var_x}={x_trend_era5:.4f}, {var_y}={y_trend_era5:.4f})")
        else:
            print(f"  Row{i+1}row：Cannot add ERA5 point (x_trend={x_trend_era5}, y_trend={y_trend_era5})")

    # Read RSS/OISST data:TCWV/TMT/TLT use RSS,SST use OISST;black circle markers on row 1
    print("\ninrow 1colpanelinAdd RSS/OISST data markers (circle)...")
    rss_cwv_trend = load_obs_data_and_calc_trend(os.path.join(obs_dir, "rss_cwv_s198801_e202412.txt"), 'cwv', 198001, 202412)
    rss_tmt_trend = load_obs_data_and_calc_trend(os.path.join(obs_dir, "rss_tmt_s198101_e202412.txt"), 'tmt', 198001, 202412)
    rss_tlt_trend = load_obs_data_and_calc_trend(os.path.join(obs_dir, "rss_tlt_s198101_e202412.txt"), 'tlt', 198001, 202412)
    oisst_sst_trend = load_obs_data_and_calc_trend(os.path.join(obs_dir, "oisst_s198109_e202507.txt"), 'sst', 198001, 202412)
    print(f"RSS/OISST trend: cwv={rss_cwv_trend}, sst={oisst_sst_trend}, tmt={rss_tmt_trend}, tlt={rss_tlt_trend}")

    for i, (var_x, var_y, label) in enumerate(var_combos):
        ax = axes[i, 0]
        if var_x == 'sst':
            x_trend_rss_oisst = oisst_sst_trend
        elif var_x == 'tmt':
            x_trend_rss_oisst = rss_tmt_trend
        elif var_x == 'tlt':
            x_trend_rss_oisst = rss_tlt_trend
        else:
            x_trend_rss_oisst = None
        if var_y == 'cwv':
            y_trend_rss_oisst = rss_cwv_trend
        elif var_y == 'tmt':
            y_trend_rss_oisst = rss_tmt_trend
        elif var_y == 'tlt':
            y_trend_rss_oisst = rss_tlt_trend
        else:
            y_trend_rss_oisst = None
        if x_trend_rss_oisst is not None and y_trend_rss_oisst is not None and not np.isnan(x_trend_rss_oisst) and not np.isnan(y_trend_rss_oisst):
            print(f"  Row{i+1}row：RSS/OISST point ({var_x}={x_trend_rss_oisst:.4f}, {var_y}={y_trend_rss_oisst:.4f})")
        else:
            print(f"  Row{i+1}row：Cannot add RSS/OISST point (x_trend={x_trend_rss_oisst}, y_trend={y_trend_rss_oisst})")

    print("\nProcessingRow2col：CanESM5_r23i1p2f1 (198001-202412) Monte Carlo scatter")
    single_model = 'CanESM5_r23i1p2f1'
    last_subplot_c = None  # Save last panel pcolormesh for colorbar
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row{i+1}row Row2col：{label}")
        x_anom, y_anom = load_specific_cmip6_model(
            cmip6_dir, single_model, 198001, 202412, var_x=var_x, var_y=var_y
        )
        if x_anom is None or y_anom is None:
            continue

        # Monte Carlo: first row param for y, second for x
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
        cbar_x = bbox.x0 - 0.02  # Shift slightly left
        cbar_y = bbox.y0 - 0.07  # Below panel with margin
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(last_subplot_c, cax=cbar_ax, orientation='horizontal', extend='max')
        cbar.ax.tick_params(labelsize=14)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontweight('bold')
        cbar.ax.text(1.06, 0.5, 'Count', transform=cbar.ax.transAxes, 
                     fontsize=14, fontweight='bold', va='center', ha='left')

    print("\nProcessingRow3col：allCMIP6model，198001-202412 Monte Carlo fit lines")

    # Use cached Monte Carlo stats (no recompute here)
    mc_stats = load_or_compute_cmip6_mc_stats(cmip6_dir, var_combos,
                                              start_yyyymm=198001,
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

    # Per row-3 panel: print top/bottom 3 slopes and top 3 |intercept| (filtered, same as plot)
    print("\nRow 3 slope/intercept extremes per combo (CC>=0.95, rows 3-4 exclusions):")
    for i, (var_x, var_y, label) in enumerate(var_combos):
        slopes = np.array(slopes_per_combo[i])
        intercepts = np.array(intercepts_per_combo[i])
        correlations = np.array(correlations_per_combo[i])
        filepaths = filepaths_per_combo[i]
        model_names = model_names_per_combo[i]
        if slopes.size == 0:
            continue
        high_cc_mask = correlations >= 0.95
        not_excluded = np.array([os.path.basename(filepaths[k]) not in CMIP6_EXCLUDED_FOR_ROW3_ROW4 for k in range(len(filepaths))])
        combined_mask = high_cc_mask & not_excluded
        n_ok = int(np.sum(combined_mask))
        if n_ok == 0:
            continue
        slopes_f = slopes[combined_mask]
        intercepts_f = intercepts[combined_mask]
        names_f = [model_names[k] for k in range(len(model_names)) if combined_mask[k]]
        # top 3 slopes
        n3 = min(3, len(slopes_f))
        idx_slope_desc = np.argsort(slopes_f)[::-1][:n3]
        top3_slope_names = [names_f[k] for k in idx_slope_desc]
        top3_slope_vals = [float(slopes_f[k]) for k in idx_slope_desc]
        # bottom 3 slopes
        idx_slope_asc = np.argsort(slopes_f)[:n3]
        bot3_slope_names = [names_f[k] for k in idx_slope_asc]
        bot3_slope_vals = [float(slopes_f[k]) for k in idx_slope_asc]
        # top 3 |intercept|
        abs_int = np.abs(intercepts_f)
        idx_abs_int = np.argsort(abs_int)[::-1][:n3]
        top3_abs_int_names = [names_f[k] for k in idx_abs_int]
        top3_abs_int_vals = [float(intercepts_f[k]) for k in idx_abs_int]
        print(f"  Row{i+1}row {label}:")
        print(f"    top 3 slopes: {list(zip(top3_slope_names, top3_slope_vals))}")
        print(f"    bottom 3 slopes: {list(zip(bot3_slope_names, bot3_slope_vals))}")
        print(f"    top 3 |intercept|: {list(zip(top3_abs_int_names, top3_abs_int_vals))}")

    # Row3col:Plotallmodel-ensemble fit lines(no repeat Monte Carlo)
    for i, (var_x, var_y, label) in enumerate(var_combos):
        slopes = np.array(slopes_per_combo[i])
        intercepts = np.array(intercepts_per_combo[i])
        correlations = np.array(correlations_per_combo[i])
        if slopes.size == 0:
            continue

        # Filter:keep only CC >= 0.95 model-ensemble;rows 3-4 also exclude CMIP6_EXCLUDED_FOR_ROW3_ROW4 in ensemble-model
        high_cc_mask = correlations >= 0.95
        filepaths = filepaths_per_combo[i]
        not_excluded = np.array([os.path.basename(filepaths[k]) not in CMIP6_EXCLUDED_FOR_ROW3_ROW4 for k in range(len(filepaths))])
        combined_mask = high_cc_mask & not_excluded
        slopes_filtered = slopes[combined_mask]
        intercepts_filtered = intercepts[combined_mask]
        x_trends_list = x_trends_list_per_combo[i]
        y_trends_list = y_trends_list_per_combo[i]
        
        # Filter x_trends_list and y_trends_list; keep ensembles passing CC and exclusion
        x_trends_list_filtered = [x_trends_list[k] for k in range(len(x_trends_list)) if combined_mask[k]]
        y_trends_list_filtered = [y_trends_list[k] for k in range(len(y_trends_list)) if combined_mask[k]]

        if slopes_filtered.size == 0:
            continue

        x_min, x_max, y_min, y_max = get_axis_limits(var_x, var_y)
        x_vals = np.array([x_min, x_max])
        ax = axes[i, 2]

        # Plot filtered ensemble fit lines(gray: CC>=0.95; rows 3-4 exclude CMIP6_EXCLUDED_FOR_ROW3_ROW4)
        for slope, intercept in zip(slopes_filtered, intercepts_filtered):
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color="gray", linewidth=1.0, alpha=0.6, zorder=2)

        # Red line: median of gray line slopes/intercepts (fig2)
        # CC/RMSE from filtered Monte Carlo samples;RMSE vs that median line
        if slopes_filtered.size > 0 and intercepts_filtered.size > 0 and len(x_trends_list_filtered) > 0:
            slope_total = np.nanmedian(slopes_filtered)
            intercept_total = np.nanmedian(intercepts_filtered)
            y_vals_total = slope_total * x_vals + intercept_total
            ax.plot(x_vals, y_vals_total, color='red', linewidth=2, zorder=3)

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
        
        # Tick font size and bold
        ax.tick_params(axis='both', which='major', labelsize=16)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        
        # Set axis labels(row 3: hide x/y labels, show x ticks)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Row3colnotshowyticklabels(because y is shared)
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

    

    print("\nProcessingRow4col：Row3colslope distribution per variable combo (weighted histogram + KDE)")
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row{i+1}row Row4col：{label} weighted slope histogram")
        slopes = np.array(slopes_per_combo[i])
        correlations = np.array(correlations_per_combo[i])
        filepaths = filepaths_per_combo[i]
        
        if slopes.size == 0:
            continue

        # Filter:keep only CC >= 0.95 model-ensemble(consistent with row 4)
        high_cc_mask = correlations >= 0.95
        not_excluded = np.array([os.path.basename(filepaths[k]) not in CMIP6_EXCLUDED_FOR_ROW3_ROW4 for k in range(len(filepaths))])
        combined_mask = high_cc_mask & not_excluded
        slopes_filtered = slopes[combined_mask]
        filepaths_filtered = [filepaths[i] for i in range(len(filepaths)) if combined_mask[i]]
        
        if slopes_filtered.size == 0:
            continue

        # Weights: count realizations per model
        # weight = 1/#A_m,#A_m is number of realizations per model
        model_realization_count = {}
        for filepath in filepaths_filtered:
            model_name = extract_model_name_from_filename(filepath)
            if model_name:
                if model_name not in model_realization_count:
                    model_realization_count[model_name] = 0
                model_realization_count[model_name] += 1
        
        weights = []
        for filepath in filepaths_filtered:
            model_name = extract_model_name_from_filename(filepath)
            if model_name and model_name in model_realization_count:
                # weight = 1/#A_m
                weight = 1.0 / model_realization_count[model_name]
                weights.append(weight)
            else:
                weights.append(1.0)  # ifCannotidentifymodel,useweight1
        
        weights = np.array(weights)
        slopes = slopes_filtered  # Use filtered slopes
        
        # Set x limits by row index, and filter data
        x_ranges = [
            (0, 15),
            (0, 10),
            (0, 11),
            (0, 3),
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

        # Unweighted quantiles on filtered data (fig2)
        q1_val = np.nanpercentile(slopes_filtered_range, 25) if slopes_filtered_range.size > 0 else np.nan
        median_val = np.nanmedian(slopes_filtered_range) if slopes_filtered_range.size > 0 else np.nan
        q3_val = np.nanpercentile(slopes_filtered_range, 75) if slopes_filtered_range.size > 0 else np.nan
        
        y_hist_max = np.max(n) if len(n) > 0 else 0
        
        # KDE (weighted data)
        # Repeat points by weight for KDE
        # Weighted KDE N/A in scipy gaussian_kde
        # Approximate via repeated points
        # Normalize weights to integers and repeat points
        y_kde_max = 0  # Store KDE curve maximum
        if len(weights_filtered_range) > 0 and weights_filtered_range.min() > 0:
            weights_normalized = (weights_filtered_range / weights_filtered_range.min() * 10).astype(int)  # scale weights to integers
            slopes_for_kde = np.repeat(slopes_filtered_range, weights_normalized)
            
            if len(slopes_for_kde) > 1:
                try:
                    kde = gaussian_kde(slopes_for_kde)
                    # Larger bandwidth for smoother KDE(default factor 1.0; 1.5–2.0 smoother)
                    kde.set_bandwidth(kde.factor * 2.5)
                    # More points for KDE curve,smoother curve
                    x_kde = np.linspace(x_min, x_max, 500)
                    y_kde = kde(x_kde)
                    y_kde_max = np.max(y_kde)  # KDE curve maximum
                    ax_hist.plot(x_kde, y_kde, color='orange', linewidth=3)
                except:
                    pass  # if KDE fails, skip
        
        ax_hist.set_xlim(x_min, x_max)
        
        # Half-peak height for marker placement
        if y_kde_max > 0:
            y_dashed_line = y_kde_max / 2.0
        else:
            # if no KDE, use histogram peak
            y_dashed_line = y_hist_max / 2.0 if y_hist_max > 0 else ax_hist.get_ylim()[1] / 2.0
        
        # y/x ratios for row-1 obs and ERA5 at marker height (not plotted on row 4)
        var_x, var_y, _ = var_combos[i]
        
        # Obs trend values (black star)
        if var_x == 'sst':
            x_trend_obs = sst_trend
        elif var_x == 'tmt':
            x_trend_obs = tmt_trend
        elif var_x == 'tlt':
            x_trend_obs = tlt_trend
        else:
            x_trend_obs = None
        
        if var_y == 'cwv':
            y_trend_obs = cwv_trend
        elif var_y == 'tmt':
            y_trend_obs = tmt_trend
        elif var_y == 'tlt':
            y_trend_obs = tlt_trend
        else:
            y_trend_obs = None
        
        # ERA5 trend values (black circle)
        if var_x == 'sst':
            x_trend_era5 = era5_sst_trend
        elif var_x == 'tmt':
            x_trend_era5 = era5_tmt_trend
        elif var_x == 'tlt':
            x_trend_era5 = era5_tlt_trend
        else:
            x_trend_era5 = None
        
        if var_y == 'cwv':
            y_trend_era5 = era5_cwv_trend
        elif var_y == 'tmt':
            y_trend_era5 = era5_tmt_trend
        elif var_y == 'tlt':
            y_trend_era5 = era5_tlt_trend
        else:
            y_trend_era5 = None
        
        # RSS/OISST trend values (circle)
        if var_x == 'sst':
            x_trend_rss_oisst = oisst_sst_trend
        elif var_x == 'tmt':
            x_trend_rss_oisst = rss_tmt_trend
        elif var_x == 'tlt':
            x_trend_rss_oisst = rss_tlt_trend
        else:
            x_trend_rss_oisst = None
        if var_y == 'cwv':
            y_trend_rss_oisst = rss_cwv_trend
        elif var_y == 'tmt':
            y_trend_rss_oisst = rss_tmt_trend
        elif var_y == 'tlt':
            y_trend_rss_oisst = rss_tlt_trend
        else:
            y_trend_rss_oisst = None
        
        # y/x ratios for row-1 markers (not drawn on row 4)
        if x_trend_obs is not None and y_trend_obs is not None and not np.isnan(x_trend_obs) and not np.isnan(y_trend_obs) and x_trend_obs != 0:
            slope_obs = y_trend_obs / x_trend_obs
            if x_min <= slope_obs <= x_max:
                pass
        
        if x_trend_era5 is not None and y_trend_era5 is not None and not np.isnan(x_trend_era5) and not np.isnan(y_trend_era5) and x_trend_era5 != 0:
            slope_era5 = y_trend_era5 / x_trend_era5
            if x_min <= slope_era5 <= x_max:
                pass
        
        if x_trend_rss_oisst is not None and y_trend_rss_oisst is not None and not np.isnan(x_trend_rss_oisst) and not np.isnan(y_trend_rss_oisst) and x_trend_rss_oisst != 0:
            slope_rss_oisst = y_trend_rss_oisst / x_trend_rss_oisst
            if x_min <= slope_rss_oisst <= x_max:
                pass
        
        # Purple dashed (median), magenta dashed (lower quartile; fig2)
        ax_hist.axvline(median_val, color='purple', linestyle='--', linewidth=2, )
        ax_hist.axvline(q1_val, color='m', linestyle='--', linewidth=1.8)
        ax_hist.axvline(q3_val, color='m', linestyle='--', linewidth=1.8)

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

        ax_hist.set_ylabel('Possibility density', fontsize=16, fontweight='bold')
        # Panels 0-2: add units after Slope (%/K)
        if i < 3:
            ax_hist.set_xlabel('Slope (%/K)', fontsize=16, fontweight='bold')
        else:
            ax_hist.set_xlabel('Slope', fontsize=16, fontweight='bold')
        ax_hist.tick_params(axis='both', which='major', labelsize=16)
        for tick in ax_hist.get_xticklabels() + ax_hist.get_yticklabels():
            tick.set_fontweight('bold')
        # Row4colshow all tick labels

        # Panel number top-left (no combo label)
        ax_hist.text(0.05, 0.95, f"({chr(97 + i * 4 + 3)})", transform=ax_hist.transAxes, 
                     fontsize=14, fontweight='bold', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
        

    # Simple layout adjust(reduce horizontal spacing for first 3 cols)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.09, wspace=0.1, hspace=0.23)
    
    # Increase gap between cols 3 and 4
    # Positions of cols 3-4 for spacing
    if n_cols >= 4:        
        # Shift all row 4 panels
        for i in range(n_rows):
            ax = axes[i, 3]
            bbox = ax.get_position()
            # Shift panel right
            new_x0 = bbox.x0 + 0.04
            new_x1 = bbox.x1 + 0.04
            ax.set_position([new_x0, bbox.y0, new_x1 - new_x0, bbox.height])
    
    # Variable combo labels left of row 1 col 1
    for i, (var_x, var_y, label) in enumerate(var_combos):
        # Row i row-1 col-1 panel position
        ax = axes[i, 0]
        # Panel bbox after layout adjust
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.03, bbox.y0 + bbox.height / 2, label, 
                ha='right', va='center', fontsize=16, fontweight='bold', rotation=90)
    
    # Save without bbox_inches=tight to preserve subplots_adjust
    save_figure_png_pdf(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()  # Close figure; avoid plt.show() on non-GUI backend


def main():
    # datadirectory
    cmip6_dir = "../data/cmip6"
    out_dir = "../plot"
    
    # Output path
    output_path = os.path.join(out_dir, "fig_s6.png")
    
    # Plot combined scatter figure(CMIP6 data only)
    draw_combined_scatter_plot(cmip6_dir, output_path)


if __name__ == "__main__":
    main()
