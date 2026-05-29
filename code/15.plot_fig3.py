#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw fig3: probability density plots for observational data."""

import numpy as np  
from scipy import odr
from scipy.stats import linregress, gaussian_kde
import random
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt plugin issues on WSL
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import glob
import re

from supply_plot import save_figure_png_pdf


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
    correlation = np.corrcoef(x_valid, y_valid)[0, 1] if len(x_valid) > 1 else 0.0
    std_err = output.sd_beta[0]
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


def load_data(filepath, start_yyyymm=None, end_yyyymm=None):
    """Read variable time series from file for given start/end YYYYMM"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        
        time = data[:, 0].astype(int)  # YYYYMM format
        
        # Error reading file header
        with open(filepath, 'r') as f:
            header = f.readline().strip()
        
        column_names = header.split()
        
        # Infer variable type from column names (legacy logic, extended for more variables)
        variables = {}
        for i, col_name in enumerate(column_names):
            col_name_lower = col_name.lower()
            if 'time' in col_name_lower or 'yyyymm' in col_name_lower:
                continue  # skip time column
            # Exact match: prefer ts_cwv, ts_sst format (CDR file format)
            # CDR format: time(yyyymm) ts_cwv ts_sst
            if 'ts_cwv' in col_name_lower or ('cwv' in col_name_lower and 'ts_' in col_name_lower):
                if 'cwv' not in variables:
                    variables['cwv'] = data[:, i]
            elif 'ts_sst' in col_name_lower or ('sst' in col_name_lower and 'ts_' in col_name_lower):
                if 'sst' not in variables:
                    variables['sst'] = data[:, i]
            elif 'ts_tlt' in col_name_lower or ('tlt' in col_name_lower and 'ts_' in col_name_lower):
                if 'tlt' not in variables:
                    variables['tlt'] = data[:, i]
            elif 'ts_tmt' in col_name_lower or ('tmt' in col_name_lower and 'ts_' in col_name_lower):
                if 'tmt' not in variables:
                    variables['tmt'] = data[:, i]
            elif 'tlt' in col_name_lower and 'tlt' not in variables:
                variables['tlt'] = data[:, i]
            elif 'tmt' in col_name_lower and 'tmt' not in variables:
                variables['tmt'] = data[:, i]
            elif ('cwv' in col_name_lower or 'vapor' in col_name_lower or 'water' in col_name_lower) and 'cwv' not in variables:
                    variables['cwv'] = data[:, i]
            elif 'sst' in col_name_lower and 'sst' not in variables:
                variables['sst'] = data[:, i]
            elif 'sea' in col_name_lower and 'sst' not in variables:
                # Use sea as sst only when sst not found (legacy)
                variables['sst'] = data[:, i]
        
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
        print(f"Error reading file {filepath}  : {e}")
        return None, None


def find_data_file(directory, dataset_name, var_type=None):
    """Find file for dataset (see legacy draw_fig3.py)
    
    Args:
        directory: Data directory
        dataset_name: Dataset name (e.g. 'star', 'ustc', 'rss')
        var_type: Variable type 'cwv', 'tmt', 'tlt', or 'sst' to distinguish separate files
    """
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    dataset_name_lower = dataset_name.lower()
    
    # If var_type given, prefer filenames containing that variable
    # e.g. star_tmt_s198101_e202412.txt or star_tlt_s198101_e202412.txt
    if var_type is not None:
        var_type_lower = var_type.lower()
        # First search filenames that explicitly contain var_type
        for file in txt_files:
            filename = os.path.basename(file).lower()
            if dataset_name_lower in filename and var_type_lower in filename:
                # exclude CMSAF
                if 'cmsaf' not in filename:
                    return file
        
        # If not found by filename, inspect file contents
        for file in txt_files:
            filename = os.path.basename(file).lower()
            if dataset_name_lower in filename and 'cmsaf' not in filename:
                # Check header for required variables
                try:
                    with open(file, 'r') as f:
                        header = f.readline().strip()
                    column_names = header.split()
                    for col_name in column_names:
                        if var_type_lower in col_name.lower():
                            return file
                except:
                    continue
    else:
        # If var_type not given, return first matching file
        for file in txt_files:
            filename = os.path.basename(file).lower()
            if dataset_name_lower in filename:
                # exclude CMSAF
                if 'cmsaf' not in filename:
                    return file
    
    return None


def generate_valid_pairs(range_max, min_diff):
    """Generate valid start/end date pairs"""
    while True:
        a = random.randint(0, range_max)
        b = random.randint(0, range_max)
        if a < b and (b - a) >= min_diff:
            return a, b
        elif b < a and (a - b) >= min_diff:
            return b, a


def monte_carlo_trend_analysis(var1, var2, n_samples=3000, min_period=120):
    """Monte Carlo trend analysis: random periods, trends for two variables"""
    data_length = len(var1)
    var1_trends = []
    var2_trends = []
    
    for _ in range(n_samples):
        start_idx, end_idx = generate_valid_pairs(data_length-1, min_period)
        
        # Build time axis in decades
        time_points = np.arange(end_idx - start_idx + 1) / 120.0
        
        var1_segment = var1[start_idx:end_idx + 1]
        var2_segment = var2[start_idx:end_idx + 1]
        
        # Check for NaN values
        if np.any(np.isnan(var1_segment)) or np.any(np.isnan(var2_segment)):
            continue
        
        try:
            var1_trend = linregress(time_points, var1_segment)[0]
            var2_trend = linregress(time_points, var2_segment)[0]
            
            var1_trends.append(var1_trend)
            var2_trends.append(var2_trend)
        except:
            continue
    
    return np.array(var1_trends), np.array(var2_trends)


def process_obs_data_pair(obs_dir, dataset1, dataset2, start_yyyymm=None, end_yyyymm=None, var_x='sst', var_y='cwv'):
    """Process observational pair; return trends (legacy draw_fig3 read logic)
    
    Args:
        obs_dir: observational data directory
        dataset1: First dataset name (x-axis variable)
        dataset2: Second dataset name (y-axis variable)
        start_yyyymm: Start time
        end_yyyymm: End time
        var_x: x-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
        var_y: y-axis variable: 'sst', 'tmt', 'tlt', 'cwv'
    """
    var1_file = None
    var2_file = None
    var1_time = None
    var1_data = None
    var2_time = None
    var2_data = None
    
    # If same dataset, try reading both variables from one file (legacy)
    if dataset1.lower() == dataset2.lower():
        # First find file with x-axis variable
        var1_file = find_data_file(obs_dir, dataset1, var_type=var_x)
        if var1_file is not None:
            var1_time, var1_vars = load_data(var1_file, start_yyyymm, end_yyyymm)
            if var1_time is not None:
                # Check whether both variables are present
                has_var1 = var_x in var1_vars
                has_var2 = var_y in var1_vars
                
                if has_var1 and has_var2:
                    # Read from same file
                    var1_data = var1_vars[var_x]
                    var2_time = var1_time
                    var2_data = var1_vars[var_y]
                else:
                    # File lacks required variables; search separately
                    var1_file = None
                    var1_time = None
                    var1_vars = None
    
    # If not found, search separately (legacy)
    if var1_file is None or var1_time is None:
        # Find x-axis variable file
        var1_file = find_data_file(obs_dir, dataset1, var_type=var_x)
        if var1_file is None:
            print(f"x-axis dataset not found: {dataset1} (var_type={var_x})")
            return None, None
        print(f"Found x-axis file: {var1_file}")
        
        # Find y-axis variable file
        var2_file = find_data_file(obs_dir, dataset2, var_type=var_y)
        if var2_file is None:
            print(f"y-axis dataset not found: {dataset2} (var_type={var_y})")
            return None, None
        print(f"Found y-axis file: {var2_file}")
        
        # Read x-axis variable data
        var1_time, var1_vars = load_data(var1_file, start_yyyymm, end_yyyymm)
        if var1_time is None or var_x not in var1_vars:
            print(f"Failed to read x-axis variable or variable not found: {var1_file}, variable: {list(var1_vars.keys()) if var1_vars else 'None'}")
            return None, None
        var1_data = var1_vars[var_x]
        
        # Read y-axis variable data
        var2_time, var2_vars = load_data(var2_file, start_yyyymm, end_yyyymm)
        if var2_time is None or var_y not in var2_vars:
            print(f"Failed to read y-axis variable or variable not found: {var2_file}, variable: {list(var2_vars.keys()) if var2_vars else 'None'}")
            return None, None
        var2_data = var2_vars[var_y]
    
    if var2_time is None:
        var2_time = var1_time
    
    common_times = np.intersect1d(var1_time, var2_time)
    
    if len(common_times) < 120:
        print(f"Common period too short: {len(common_times)}  months")
        return None, None
    
    # Extract data for common period
    var1_matched = []
    var2_matched = []

    
    for t in common_times:
        var1_idx = np.where(var1_time == t)[0]
        var2_idx = np.where(var2_time == t)[0]
        
        if len(var1_idx) > 0 and len(var2_idx) > 0:
            var1_matched.append(var1_data[var1_idx[0]])
            var2_matched.append(var2_data[var2_idx[0]])
    
    var1_matched = np.array(var1_matched)
    var2_matched = np.array(var2_matched)
    
    var1_anomaly = calculate_monthly_anomaly(var1_matched)
    var2_anomaly = calculate_monthly_anomaly(var2_matched)

    # Convert CWV to percent if y-axis is cwv (legacy handling)
    if var_y == 'cwv':
        var2_anomaly = var2_anomaly * 100. / 41.
    
    # Run Monte Carlo (legacy: n_samples=3000, min_period=120)
    var2_trends, var1_trends = monte_carlo_trend_analysis(var2_anomaly, var1_anomaly, n_samples=3000, min_period=120)
    
    return var1_trends, var2_trends


def draw_single_subplot(ax, x_trends, y_trends, title_text, x_min=None, x_max=None, y_min=None, y_max=None,
                         xlabel=None, ylabel=None, col_idx=0, show_xticklabels=True):
    """Density scatter on axes; col_idx controls which axis labels/ticks show."""
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
    
    x = x_trends
    y = y_trends
    
    # Filter NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) == 0:
        return None
    
    # Plot density
    H, xedges, yedges = np.histogram2d(x_valid, y_valid, bins=30, range=[[x_min, x_max], [y_min, y_max]])
    H_masked = np.ma.masked_where(H == 0, H)
    
    c = ax.pcolormesh(xedges, yedges, H_masked.T, cmap='jet', shading='auto', vmin=0, vmax=20, zorder=2)
    
    if len(x_valid) > 10:
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
        slope, intercept, r_value, p_value, std_err = odr_linear_regression(x_valid, y_valid)
        
        if slope is not None:
            ax.plot([x_min, x_max], [slope * x_min + intercept, slope * x_max + intercept],
                    color='red', linewidth=2, zorder=3)
            rmse = np.sqrt(np.nanmean((x_valid * slope + intercept - y_valid) ** 2))
            ax.text(0.05, 0.82, f'CC = {correlation:.3f}\nRMSE = {rmse:.3f}',
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.text(0.35, 0.18, f'Slope = {slope:.3f}\nIntercept = {intercept:.3f}',
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
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
    
    if col_idx == 1:  # shared y: hide col2 labels via tick_params
        ax.tick_params(axis='y', labelleft=False)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
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


def extract_model_name_from_filename(filename):
    """Extract model name from filename (see fig2)"""
    # Filename format: ModelName_rXiYpZfW_sYYYYMM_eYYYYMM.txt
    # Extract ModelName portion
    basename = os.path.basename(filename)
    match = re.match(r'^([^_]+)_', basename)
    
    if match:
        return match.group(1)
    # If match fails, split on first underscore
    parts = basename.split('_')
    if len(parts) > 0:
        return parts[0]
    
    return None


def load_cmip6_data(filepath, start_yyyymm=None, end_yyyymm=None, var_x='sst', var_y='cwv'):
    """Read CMIP6 data (see fig2)"""
    if not os.path.exists(filepath):
        return None, None, None
    
    try:
        data = np.loadtxt(filepath, skiprows=1)
        yyyymm = data[:, 0].astype(int)
        cwv = data[:, 1]
        sst = data[:, 2]
        tlt = data[:, 3]
        tmt = data[:, 4]
        
        var_map = {'cwv': cwv, 'sst': sst, 'tlt': tlt, 'tmt': tmt}
        x_data = var_map.get(var_x, sst)
        y_data = var_map.get(var_y, cwv)
        
        # Time filtering
        mask = np.ones(len(yyyymm), dtype=bool)
        if start_yyyymm is not None:
            mask = mask & (yyyymm >= start_yyyymm)
        if end_yyyymm is not None:
            mask = mask & (yyyymm <= end_yyyymm)
        
        yyyymm = yyyymm[mask]
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) == 0:
            return None, None, None
        
        x_anomaly = calculate_monthly_anomaly(x_data)
        y_anomaly = calculate_monthly_anomaly(y_data)
        
        if var_y == 'cwv':
            y_anomaly = y_anomaly * 100. / 41.
        if var_x == 'cwv':
            x_anomaly = x_anomaly * 100. / 41.
        
        return x_anomaly, y_anomaly, yyyymm
        
    except Exception as e:
        print(f"Error reading CMIP6 file {filepath}  : {e}")
        return None, None, None


# Column 4 slope distribution matches fig2: rows 3–4 exclude these CMIP6 model-ensemble files
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


def load_or_compute_cmip6_mc_stats(cmip6_dir, var_combos, start_yyyymm=200206, end_yyyymm=202412,
                                   n_samples=3000, min_period=120):
    """Load or compute CMIP6 Monte Carlo statistics (see fig2)"""
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
    x_trends_list_per_combo = [[] for _ in range(len(var_combos))]
    y_trends_list_per_combo = [[] for _ in range(len(var_combos))]
    
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        
        for j, (var_x, var_y, label) in enumerate(var_combos):
            x_anomaly, y_anomaly, _ = load_cmip6_data(filepath, start_yyyymm, end_yyyymm, var_x, var_y)
            
            if x_anomaly is None or y_anomaly is None:
                continue
            
            y_trends, x_trends = monte_carlo_trend_analysis(y_anomaly, x_anomaly, n_samples=n_samples, min_period=min_period)
            
            if len(x_trends) == 0 or len(y_trends) == 0:
                continue
            
            valid_mask = ~(np.isnan(x_trends) | np.isnan(y_trends))
            if np.sum(valid_mask) < 10:
                continue
            
            x_trends_valid = x_trends[valid_mask]
            y_trends_valid = y_trends[valid_mask]
            
            slope, intercept, correlation, _, _ = odr_linear_regression(x_trends_valid, y_trends_valid)
            
            if slope is not None:
                slopes_per_combo[j].append(slope)
                intercepts_per_combo[j].append(intercept)
                correlations_per_combo[j].append(correlation)
                model_names_per_combo[j].append(extract_model_name_from_filename(filepath))
                filepaths_per_combo[j].append(filepath)
                x_trends_list_per_combo[j].append(x_trends_valid)
                y_trends_list_per_combo[j].append(y_trends_valid)
    
    cache_data = {
        "var_combos": var_combos,
        "slopes_per_combo": slopes_per_combo,
        "intercepts_per_combo": intercepts_per_combo,
        "correlations_per_combo": correlations_per_combo,
        "model_names_per_combo": model_names_per_combo,
        "filepaths_per_combo": filepaths_per_combo,
        "x_trends_list_per_combo": x_trends_list_per_combo,
        "y_trends_list_per_combo": y_trends_list_per_combo,
    }
    np.save(cache_path, cache_data)
    print(f"CMIP6 Monte Carlo statistics saved to: {cache_path}")
    
    return cache_data


def draw_combined_scatter_plot(obs_dir, cmip6_dir, output_path):
    """Plot 4x4 probability-density grid (transposed layout)"""
    n_rows = 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 2, 4 * n_rows + 1))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Share y only between cols 1–2 per row; col3 uses independent y limits
    for i in range(n_rows):
        axes[i, 1].sharey(axes[i, 0])
    
    # Define 5 variable pairs (rows, transposed)
    var_combos = [
        ('sst', 'cwv', 'TCWV vs SST'),
        ('tmt', 'cwv', 'TCWV vs TMT'),
        ('tlt', 'cwv', 'TCWV vs TLT'),
        ('sst', 'tmt', 'TMT vs SST'),
    ]
    
    ROW_AXIS_LIMITS_COL0_COL1 = [
        (-0.2, 0.5, -1.0, 5.0),
        (-0.3, 0.7, -1.0, 5.0),
        (-0.3, 0.7, -1.0, 5.0),
        (-0.2, 0.5, -0.3, 0.7),
    ]

    def get_axis_limits(var_x, var_y):
        # Cols 3–4 limits still by variable type
        if var_x == 'sst':
            x_min, x_max = -0.2, 0.5
        else:
            x_min, x_max = -0.3, 0.7
        if var_y == 'cwv':
            y_min, y_max = -1.0, 5.0
        else:
            y_min, y_max = -0.3, 0.7
        return x_min, x_max, y_min, y_max
    
    print("Processing row 1, col 1: USTC/STAR PDF (200206-202412)")
    
    cwv_dataset = 'ustc'  # USTC
    sst_dataset = 'ustc'  # USTC 
    temp_dataset = 'star'  # STAR (for TMT and TLT)
    
    last_subplot_c = None
    
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 1: {label}")
        
        # Period: rows 2,3,5 -> 200206-202012; else 200206-202412
        if i in [1, 2, 3]:
            start_yyyymm = 200206
            end_yyyymm = 202012
        else:
            start_yyyymm = 200206
            end_yyyymm = 202412
        
        if var_x == 'cwv':
            x_dataset = cwv_dataset
        elif var_x == 'sst':
            x_dataset = sst_dataset  # USTC SST
        else:  # tmt, tlt
            x_dataset = temp_dataset  # STAR
        
        if var_y == 'cwv':
            y_dataset = cwv_dataset
        elif var_y == 'sst':
            y_dataset = sst_dataset  # USTC SST
        else:  # tmt, tlt
            y_dataset = temp_dataset  # STAR
        
        x_trends, y_trends = process_obs_data_pair(
            obs_dir, x_dataset, y_dataset, start_yyyymm, end_yyyymm, var_x=var_x, var_y=var_y
        )
        
        if x_trends is None or len(x_trends) == 0:
            print(f"    Warning: no data; skipping")
            continue
        
        x_min, x_max, y_min, y_max = ROW_AXIS_LIMITS_COL0_COL1[i]
        
        if var_x == 'cwv':
            x_source = 'USTC'
        elif var_x == 'sst':
            x_source = 'USTC'
        else:  # tmt, tlt
            x_source = 'STAR'
        
        if var_y == 'cwv':
            y_source = 'USTC'
        elif var_y == 'sst':
            y_source = 'USTC'
        else:  # tmt, tlt
            y_source = 'STAR'
        
        period_str = f"{start_yyyymm}-{end_yyyymm}"
        
        title = f"({chr(97 + i * 4)}) {y_source} vs {x_source}"
        
        ylabel = get_var_label(var_y)
        c = draw_single_subplot(
            axes[i, 0], x_trends, y_trends, title,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            xlabel=None, ylabel=ylabel, col_idx=0, show_xticklabels=True
        )
        
        if i == 3 and c is not None:
            last_subplot_c = c
    
    
    print("\nProcessing row 2, col 2: PDFs for different data combos")
    
    col2_configs = [
        ('cobe', 'rss', 'sst', 'cwv', 200206, 202412, 'RSS TCWV vs COBE SST'),
        ('rss', 'rss', 'tmt', 'cwv', 200206, 202012, 'RSS TCWV vs RSS TMT'),
        ('uah', 'rss', 'tlt', 'cwv', 200206, 202012, 'RSS TCWV vs UAH TLT'),
        ('hadsst', 'uah', 'sst', 'tmt', 200206, 202012, 'UAH TMT vs HADSST'),
    ]
    
    last_subplot_c_col2 = None
    col2_xlabels = [get_var_label(c[2]) for c in col2_configs]

    for i, (x_dataset, y_dataset, var_x, var_y, start_yyyymm, end_yyyymm, label) in enumerate(col2_configs):
        print(f"  Row {i+1} col 2: {label} ({start_yyyymm}-{end_yyyymm})")
        
        x_trends, y_trends = process_obs_data_pair(
            obs_dir, x_dataset, y_dataset, start_yyyymm, end_yyyymm, var_x=var_x, var_y=var_y
        )

        if x_dataset == 'hadsst':
            print(x_trends)
        
        if x_trends is None or len(x_trends) == 0:
            print(f"    Warning: no data; skipping")
            continue
        
        # Col 2: same unified limits as col 1
        x_min, x_max, y_min, y_max = ROW_AXIS_LIMITS_COL0_COL1[i]
        
        # Data source labels(from col2_configs dataset names)
        dataset_display_names = {
            'rss': 'RSS',
            'cobe': 'COBE',
            'uah': 'UAH',
            'star': 'STAR',
            'oisst': 'OISST',
            'hadsst': 'HADSST',
        }
        x_source = dataset_display_names.get(x_dataset.lower(), x_dataset.upper())
        y_source = dataset_display_names.get(y_dataset.lower(), y_dataset.upper())
        
        period_str = f"{start_yyyymm}-{end_yyyymm}"
        
        title = f"({chr(97 + i * 4 + 1)}) {y_source} vs {x_source}"
        
        # Col 2: xlabel placed between cols 1–2 later; no ylabel
        c = draw_single_subplot(
            axes[i, 1], x_trends, y_trends, title,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            xlabel='', ylabel=None, col_idx=1, show_xticklabels=True
        )
        
        if i == 3 and c is not None:
            last_subplot_c_col2 = c
    
    
    print("\nColumn 3: CC and RMSE scatter by variable pair")
    
    # Time ranges by row index (same as col 1)
    # rows 1,4 (i=0,3): 200206-202412
    
    # Data sources per variable (exclude ERA5/MERRA2)
    var_datasets = {
        'cwv': ['rss', 'ustc'],  # TCWV sources
        'sst': ['cobe', 'ersst', 'hadisst', 'hadsst', 'oisst', 'ustc'],  # SST sources
        'tmt': ['rss', 'star', 'uah'],  # TMT sources
        'tlt': ['rss', 'star', 'uah'],  # TLT sources
    }
    
    # Color map for y-axis variable
    var_colors = {
        'cwv': {'rss': 'blue', 'ustc': 'red'},
        'sst': {'cobe': 'blue', 'ersst': 'green', 'hadisst': 'orange', 'hadsst': 'purple', 'oisst': 'brown', 'ustc': 'magenta'},
        'tmt': {'rss': 'blue', 'star': 'red', 'uah': 'green'},
        'tlt': {'rss': 'blue', 'star': 'red', 'uah': 'green'},
    }
    
    # Marker map for x-axis variable
    var_markers = {
        'sst': {'cobe': 'o', 'ersst': 's', 'hadisst': '^', 'hadsst': 'v', 'oisst': 'D', 'ustc': 'p'},
        'tmt': {'rss': 'o', 'star': 's', 'uah': '^'},
        'tlt': {'rss': 'o', 'star': 's', 'uah': '^'},
        'cwv': {'rss': 'o', 'ustc': 's'},
    }
    
    # Step 1: collect CC from row 3 panels for unified y limits
    all_cc_values = []
    all_col3_data = []  # Per-row data for plotting
    
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 3: {label} (collect data)")
        
        # Set time range by row (same as col 1)
        if i in [1, 2, 3]:
            start_yyyymm = 200206
            end_yyyymm = 202012
        else:
            start_yyyymm = 200206
            end_yyyymm = 202412
        
        # Get x- and y-axis data source lists
        x_datasets = var_datasets.get(var_x, [])
        y_datasets = var_datasets.get(var_y, [])
        
        if len(x_datasets) == 0 or len(y_datasets) == 0:
            all_col3_data.append(None)
            continue
        
        # Read x-axis variable data
        x_data_dict = {}
        for x_ds in x_datasets:
            x_file = find_data_file(obs_dir, x_ds, var_type=var_x)
            if x_file is not None:
                x_time, x_vars = load_data(x_file, start_yyyymm, end_yyyymm)
                if x_time is not None and var_x in x_vars:
                    x_anomaly = calculate_monthly_anomaly(x_vars[var_x])
                    # Convert CWV to percent if applicable
                    if var_x == 'cwv':
                        x_anomaly = x_anomaly * 100. / 41.
                    x_data_dict[x_ds] = (x_time, x_anomaly)
        
        # Read y-axis variable data
        y_data_dict = {}
        for y_ds in y_datasets:
            y_file = find_data_file(obs_dir, y_ds, var_type=var_y)
            if y_file is not None:
                y_time, y_vars = load_data(y_file, start_yyyymm, end_yyyymm)
                if y_time is not None and var_y in y_vars:
                    y_anomaly = calculate_monthly_anomaly(y_vars[var_y])
                    # Convert CWV to percent if applicable
                    if var_y == 'cwv':
                        y_anomaly = y_anomaly * 100. / 41.
                    y_data_dict[y_ds] = (y_time, y_anomaly)
        
        # Collect CC and RMSE for all combinations
        x_names = []
        y_names = []
        cc_values = []
        rmse_values = []
        
        # Compute CC and RMSE for all combinations
        for y_ds, (y_time, y_data) in y_data_dict.items():
            for x_ds, (x_time, x_data) in x_data_dict.items():
                common_times = np.intersect1d(y_time, x_time)
                
                if len(common_times) < 120:
                    continue
                
                # Extract data for common period
                y_matched = []
                x_matched = []
                
                for t in common_times:
                    y_idx = np.where(y_time == t)[0]
                    x_idx = np.where(x_time == t)[0]
                    
                    if len(y_idx) > 0 and len(x_idx) > 0:
                        y_matched.append(y_data[y_idx[0]])
                        x_matched.append(x_data[x_idx[0]])
                
                if len(y_matched) < 120:
                    continue
                
                y_matched = np.array(y_matched)
                x_matched = np.array(x_matched)
                
                # Monte Carlo trends (same as row-1 subplots)
                y_trends, x_trends = monte_carlo_trend_analysis(y_matched, x_matched, n_samples=3000, min_period=120)
                
                if len(x_trends) == 0 or len(y_trends) == 0:
                    continue
                
                # Regress trends (same as row-1 subplots)
                valid_mask = ~(np.isnan(x_trends) | np.isnan(y_trends))
                if np.sum(valid_mask) < 10:
                    continue
                
                x_trends_valid = x_trends[valid_mask]
                y_trends_valid = y_trends[valid_mask]
                
                # Compute correlation coefficient
                correlation = np.corrcoef(x_trends_valid, y_trends_valid)[0, 1]
                
                slope, intercept, _, _, _ = odr_linear_regression(x_trends_valid, y_trends_valid)
                if slope is None:
                    continue
                
                # RMSE after regression (same as row-1 subplots)
                rmse = np.sqrt(np.nanmean((x_trends_valid * slope + intercept - y_trends_valid)**2))
                
                x_names.append(x_ds)
                y_names.append(y_ds)
                cc_values.append(correlation)
                rmse_values.append(rmse)
        
        if len(cc_values) > 0:
            all_cc_values.extend(cc_values)
            all_col3_data.append({
                'x_names': x_names,
                'y_names': y_names,
                'cc_values': cc_values,
                'rmse_values': rmse_values,
                'var_x': var_x,
                'var_y': var_y,
                'x_datasets': x_datasets,
                'y_datasets': y_datasets
            })
        else:
            all_col3_data.append(None)
    
    y_min_col3 = 0.957
    y_max_col3 = 1.005
    y_ticks_col3 = np.array([0.96, 0.98, 1.00])

    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 3: {label} (plot)")
        
        if all_col3_data[i] is None:
            continue
        
        data = all_col3_data[i]
        x_names = data['x_names']
        y_names = data['y_names']
        cc_values = data['cc_values']
        rmse_values = data['rmse_values']
        x_datasets = data['x_datasets']
        y_datasets = data['y_datasets']
        
        if len(cc_values) > 0:
            ax = axes[i, 2]
            
            y_colors = var_colors.get(var_y, {})
            x_markers_dict = var_markers.get(var_x, {})
            
            # Plot points per y-axis source (color)
            for y_ds in y_datasets:
                # Combos for this y source (saved y_names)
                y_indices = [i for i, y_name in enumerate(y_names) if y_name == y_ds]
                
                if len(y_indices) == 0:
                    continue
                
                # Plot points per x-axis source (marker)
                for x_ds in x_datasets:
                    # Index for combo (saved x_names)
                    combo_indices = [i for i in y_indices if x_names[i] == x_ds]
                    
                    if len(combo_indices) == 0:
                        continue
                    
                    # CC and RMSE for combo
                    combo_cc = [cc_values[i] for i in combo_indices]
                    combo_rmse = [rmse_values[i] for i in combo_indices]
                    
                    # Hollow markers: color=y variable, marker=x variable, edge matches color
                    ax.scatter(combo_rmse, combo_cc, 
                              facecolors='none',
                              edgecolors=y_colors.get(y_ds, 'black'),
                              marker=x_markers_dict.get(x_ds, 'o'),
                              s=150, linewidths=1.5, zorder=3)
            
            rmse_label = 'RMSE (%)' if var_y == 'cwv' else 'RMSE (K)'
            ax.set_xlabel(rmse_label, fontsize=16, fontweight='bold')
            ax.set_ylabel('CC', fontsize=16, fontweight='bold')
            ax.set_ylim(y_min_col3, y_max_col3)
            ax.set_yticks(y_ticks_col3)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # Panel index inside upper-left (fig2)
            ax.text(0.05, 0.95, f"({chr(97 + i * 4 + 2)})", transform=ax.transAxes, 
                    fontsize=15, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
            
            # Set tick font
            ax.tick_params(labelsize=16)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
        else:
            print(f"    Warning: no valid data; skip plot")
    
    print("\nColumn 4: CMIP6 slope KDE from fig2 (orange solid)")
    
    # Use fig2 CMIP6 data (same as fig2 col4)
    mc_stats = load_or_compute_cmip6_mc_stats(cmip6_dir, var_combos,
                                              start_yyyymm=200206,
                                              end_yyyymm=202012,
                                              n_samples=3000,
                                              min_period=120)
    slopes_per_combo = mc_stats["slopes_per_combo"]
    correlations_per_combo = mc_stats["correlations_per_combo"]
    filepaths_per_combo = mc_stats["filepaths_per_combo"]
    
    # Plot col4: orange KDE (same as fig2 col4)
    for i, (var_x, var_y, label) in enumerate(var_combos):
        print(f"  Row {i+1} col 4: {label} slope KDE")
        slopes = np.array(slopes_per_combo[i])
        correlations = np.array(correlations_per_combo[i])
        filepaths = filepaths_per_combo[i]
        
        if slopes.size == 0:
            continue
        
        # Filter: CC >= 0.95 and exclude CMIP6_EXCLUDED_FOR_ROW3_ROW4 (fig2 col4)
        high_cc_mask = correlations >= 0.95
        not_excluded = np.array([os.path.basename(filepaths[k]) not in CMIP6_EXCLUDED_FOR_ROW3_ROW4 for k in range(len(filepaths))])
        combined_mask = high_cc_mask & not_excluded
        slopes_filtered = slopes[combined_mask]
        filepaths_filtered = [filepaths[k] for k in range(len(filepaths)) if combined_mask[k]]
        
        if slopes_filtered.size == 0:
            continue
        
        # Weights: count realizations per model (fig2)
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
        
        ax_kde = axes[i, 3]

        # Compute quartiles; gray vertical dashed lines (no values labeled)
        q1_val = np.nanpercentile(slopes_filtered_range, 25)
        q3_val = np.nanpercentile(slopes_filtered_range, 75)
        
        # Weighted KDE (fig2)
        # Repeat points by weight for KDE
        y_kde_max = 0  # store KDE curve maximum
        if len(weights_filtered_range) > 0 and weights_filtered_range.min() > 0:
            weights_normalized = (weights_filtered_range / weights_filtered_range.min() * 10).astype(int)  # scale weights to integers
            slopes_for_kde = np.repeat(slopes_filtered_range, weights_normalized)
            
            if len(slopes_for_kde) > 1:
                try:
                    kde = gaussian_kde(slopes_for_kde)
                    # Larger bandwidth for smoother curve (fig2)
                    kde.set_bandwidth(kde.factor * 2.5)
                    x_kde = np.linspace(x_min, x_max, 500)
                    y_kde = kde(x_kde)
                    y_kde_max = np.max(y_kde)  # KDE curve maximum
                    ax_kde.plot(x_kde, y_kde, color='orange', linewidth=3)
                except:
                    pass  # if KDE fails, skip
        
        ax_kde.set_xlim(x_min, x_max)
        ax_kde.axvline(q1_val, color='m', linestyle='--', linewidth=1.8, zorder=1.8)
        ax_kde.axvline(q3_val, color='m', linestyle='--', linewidth=1.8, zorder=1.8)
        
        # Dashed line spacing (fraction of KDE max); one dashed line per color
        dash_line_spacing_ratio = 0.06
        
        # Observational slopes (same method as row 3)
        # Period by row (consistent with row 3)
        if i in [1, 2, 3]:
            start_yyyymm = 200206
            end_yyyymm = 202012
        else:
            start_yyyymm = 200206
            end_yyyymm = 202412
        
        # Get x- and y-axis data source lists
        x_datasets_obs = var_datasets.get(var_x, [])
        y_datasets_obs = var_datasets.get(var_y, [])
        
        # Computeobservationaldata slope
        obs_slopes = []  # store slope values
        obs_colors = []  # store colors
        obs_markers = []  # store markers
        
        for y_ds in y_datasets_obs:
            y_file = find_data_file(obs_dir, y_ds, var_type=var_y)
            if y_file is None:
                continue
            y_time, y_vars = load_data(y_file, start_yyyymm, end_yyyymm)
            if y_time is None or var_y not in y_vars:
                continue
            
            y_anomaly = calculate_monthly_anomaly(y_vars[var_y])
            if var_y == 'cwv':
                y_anomaly = y_anomaly * 100. / 41.
            
            for x_ds in x_datasets_obs:
                x_file = find_data_file(obs_dir, x_ds, var_type=var_x)
                if x_file is None:
                    continue
                x_time, x_vars = load_data(x_file, start_yyyymm, end_yyyymm)
                if x_time is None or var_x not in x_vars:
                    continue
                
                x_anomaly = calculate_monthly_anomaly(x_vars[var_x])
                if var_x == 'cwv':
                    x_anomaly = x_anomaly * 100. / 41.
                
                common_times = np.intersect1d(y_time, x_time)
                if len(common_times) < 120:
                    continue
                
                # Extract data for common period
                y_matched = []
                x_matched = []
                for t in common_times:
                    y_idx = np.where(y_time == t)[0]
                    x_idx = np.where(x_time == t)[0]
                    if len(y_idx) > 0 and len(x_idx) > 0:
                        y_matched.append(y_anomaly[y_idx[0]])
                        x_matched.append(x_anomaly[x_idx[0]])
                
                if len(y_matched) < 120:
                    continue
                
                y_matched = np.array(y_matched)
                x_matched = np.array(x_matched)
                
                # Monte Carlo trends
                y_trends, x_trends = monte_carlo_trend_analysis(y_matched, x_matched, n_samples=3000, min_period=120)
                
                if len(x_trends) == 0 or len(y_trends) == 0:
                    continue
                
                # Regress trends for slope
                valid_mask = ~(np.isnan(x_trends) | np.isnan(y_trends))
                if np.sum(valid_mask) < 10:
                    continue
                
                x_trends_valid = x_trends[valid_mask]
                y_trends_valid = y_trends[valid_mask]
                
                # ODR regressionslope
                slope, intercept, _, _, _ = odr_linear_regression(x_trends_valid, y_trends_valid)
                if slope is not None:
                    # Keep slopes within x limits only
                    if x_min <= slope <= x_max:
                        obs_slopes.append(slope)
                        # Matching color and marker
                        y_colors = var_colors.get(var_y, {})
                        x_markers_dict = var_markers.get(var_x, {})
                        obs_colors.append(y_colors.get(y_ds, 'black'))
                        obs_markers.append(x_markers_dict.get(x_ds, 'o'))
        
        # Same-color points on one dashed line; spacing between dashed lines
        if len(obs_slopes) > 0 and y_kde_max > 0:
            # Group by color; preserve color order
            color_to_points = {}  # color -> list of (slope, marker)
            color_order = []      # color order for y placement
            for slope, color, marker in zip(obs_slopes, obs_colors, obs_markers):
                # Hash colors as strings (avoid RGB tuple vs string duplicate keys)
                key = color if isinstance(color, str) else tuple(np.atleast_1d(color))
                if key not in color_to_points:
                    color_to_points[key] = []
                    color_order.append(key)
                color_to_points[key].append((slope, marker))
            y_ref = y_kde_max / 2.0
            spacing = y_kde_max * dash_line_spacing_ratio
            for idx, key in enumerate(color_order):
                y_dash = y_ref - idx * spacing
                ax_kde.axhline(y_dash, color="grey", linestyle='--', linewidth=1.5, zorder=2)
                for slope, marker in color_to_points[key]:
                    ax_kde.scatter(slope, y_dash, facecolors='none', edgecolors=key,
                                  marker=marker, s=150, linewidths=1.5, zorder=3)
        
        # Col 4: show ylabel and xlabel on all subplots
        ax_kde.set_ylabel('Possibility density', fontsize=16, fontweight='bold')
        # First three panels: append unit (%/K) to Slope
        if i < 3:
            ax_kde.set_xlabel('Slope (%/K)', fontsize=16, fontweight='bold')
        else:
            ax_kde.set_xlabel('Slope', fontsize=16, fontweight='bold')
        
        # Panel index inside upper-left (fig2)
        ax_kde.text(0.05, 0.95, f"({chr(97 + i * 4 + 3)})", transform=ax_kde.transAxes, 
                    fontsize=14, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
        
        # Set tick font
        ax_kde.tick_params(labelsize=16)
        for label in ax_kde.get_xticklabels() + ax_kde.get_yticklabels():
            label.set_fontweight('bold')
    
    # Adjust layout; leave right margin for col gaps
    plt.subplots_adjust(left=0.08, right=0.8, top=0.98, bottom=0.07, wspace=0.1, hspace=0.23)

    # Increase gap between cols 2–3 and 3–4 (shift cols 3–4 right)
    gap_2_3 = 0.04   # extra gap cols 2–3 (figure coords)
    gap_3_4 = 0.04   # extra gap cols 3–4
    for i in range(n_rows):
        b2 = axes[i, 2].get_position()
        axes[i, 2].set_position([b2.x0 + gap_2_3, b2.y0, b2.width, b2.height])
        b3 = axes[i, 3].get_position()
        axes[i, 3].set_position([b3.x0 + gap_2_3 + gap_3_4, b3.y0, b3.width, b3.height])

    # Center col2 xlabel between cols 1–2 per row
    for i in range(n_rows):
        if i >= len(col2_xlabels):
            break
        ax0, ax1 = axes[i, 0], axes[i, 1]
        bbox0, bbox1 = ax0.get_position(), ax1.get_position()
        center_x = (bbox0.x1 + bbox1.x0) / 2
        y_bottom = bbox0.y0 - 0.02
        fig.text(center_x, y_bottom, col2_xlabels[i], ha='center', va='top', fontsize=16, fontweight='bold')
    
    # if n_cols >= 4:        
    #     # Shift all col4 subplots
    #     for i in range(n_rows):
    #         ax = axes[i, 3]
    #         bbox = ax.get_position()
    #         # Shift subplot right
    #         new_x0 = bbox.x0 + 0.05
    #         ax.set_position([new_x0, bbox.y0, bbox.width, bbox.height])
    #     bbox_row1 = ax_row1.get_position()
    #     bbox_row2 = ax_row2.get_position()
        
        
    #     # Compute downward shift
    #     move_down = current_gap - new_gap
        
    #     # Shift row 1 down
    #     new_y0_row1 = bbox_row1.y0 - move_down
    #     ax_row1.set_position([bbox_row1.x0, new_y0_row1, bbox_row1.width, bbox_row1.height])
    
    
    # # Get adjusted row 2 positions
    # ax_row2_ref = axes[1, 0]
    # bbox_row2_ref = ax_row2_ref.get_position()
    
    # # Current gap between rows 2 and 3
    # current_gap_row2_row3 = bbox_row2_ref.y0 - bbox_row3.y1
    
    # # Reduce gap to 50%
    # new_gap_row2_row3 = current_gap_row2_row3 * 0.5
    
    # # Compute downward shift
    # move_down = current_gap_row2_row3 - new_gap_row2_row3
    
        
    #     # Move row 1 down
    #     ax_row1.set_position([bbox_row1.x0, bbox_row1.y0 - move_down, 
    #                           bbox_row1.width, bbox_row1.height])
        
    #     # Move row 2 down
    #     ax_row2.set_position([bbox_row2.x0, bbox_row2.y0 - move_down, 
    #                           bbox_row2.width, bbox_row2.height])
    
    # Colorbar below row 5 spanning cols 1–2
    if last_subplot_c is not None:
        bbox0 = axes[3, 0].get_position()
        bbox1 = axes[3, 1].get_position()
        cbar_x = bbox0.x0 + 0.02
        cbar_width = bbox1.x1 - bbox0.x0 - 0.07  # from col1 left to col2 right
        cbar_height = 0.008
        cbar_y = bbox0.y0 - 0.05
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = fig.colorbar(last_subplot_c, cax=cbar_ax, orientation='horizontal', extend='max')
        cbar.set_ticks([0, 5, 10, 15, 20])
        cbar.ax.tick_params(labelsize=14)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontweight('bold')
        cbar.ax.text(1.05, 0.5, 'Count', transform=cbar.ax.transAxes, 
                     fontsize=14, fontweight='bold', va='center', ha='left')
    
    # Add variable-pair row titles left of col 1
    # After layout, get final subplot positions (fig2)
    for i, (var_x, var_y, label) in enumerate(var_combos):
        # Get row i col 1 subplot position
        ax = axes[i, 0]
        # Get bbox after layout
        bbox = ax.get_position()
        # Add label left of subplot at center y
        fig.text(bbox.x0 - 0.06, bbox.y0 + bbox.height / 2, label, 
                ha='right', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    
    # Display names for data sources
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
    
    
    # Col4 bbox for legend (after layout)
    ax_right = axes[0, 3]
    bbox_right = ax_right.get_position()
    right_x = bbox_right.x1  # right edge of col4 subplots
    
    # Legend on right, products listed below title
    legend_x_start = right_x + 0.02  # legend start (right of subplot)
    legend_line_height = 0.012  # legend row height (tighter spacing)
    
    for i, (var_x, var_y, label) in enumerate(var_combos):
        # Get row col1 subplot position
        ax_row = axes[i, 0]
        bbox_row = ax_row.get_position()
        
        # Get color and marker maps for row
        y_colors = var_colors.get(var_y, {})
        x_markers_dict = var_markers.get(var_x, {})
        x_datasets = var_datasets.get(var_x, [])
        y_datasets = var_datasets.get(var_y, [])
        
        # Legend near top of subplot
        n_color_items = sum(1 for y_ds in y_datasets if y_ds in y_colors)
        color_block_height = (1 + n_color_items) * legend_line_height
        gap_between_blocks = 0.008
        color_row_y = bbox_row.y1 - 0.018   # title near subplot top
        shape_row_y = color_row_y - color_block_height - gap_between_blocks
        
        # align with title left
        current_x = legend_x_start

        # Legend title for y-axis variable (color)
        y_var_label = get_var_label(var_y).replace(' Trend (%/decade)', '').replace(' Trend (K/decade)', '')
        fig.text(current_x, color_row_y, f'{y_var_label} (color)', 
                ha='left', va='center', fontsize=13, fontweight='bold')
        
        # Color legend (■ aligned with title left)
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
        
        # Legend title for x-axis variable (marker)
        x_var_label = get_var_label(var_x).replace(' Trend (%/decade)', '').replace(' Trend (K/decade)', '')
        fig.text(current_x, shape_row_y, f'{x_var_label} (shape)', 
                ha='left', va='center', fontsize=13, fontweight='bold')
        
        # Marker legend (aligned with title left)
        shape_idx = 0
        for x_ds in x_datasets:
            if x_ds in x_markers_dict:
                marker = x_markers_dict[x_ds]
                ds_name = dataset_display_names.get(x_ds, x_ds.upper())
                item_y = shape_row_y - (shape_idx + 1) * legend_line_height
                # mini-axes left edge at current_x
                temp_ax = fig.add_axes([current_x, item_y - 0.006, 0.012, 0.012])
                temp_ax.scatter([0.5], [0.5], marker=marker, s=80, facecolors='none', 
                               edgecolors='black', linewidths=0.8)
                temp_ax.set_xlim(0, 1)
                temp_ax.set_ylim(0, 1)
                temp_ax.axis('off')
                fig.text(current_x + 0.012, item_y, ds_name, 
                        ha='left', va='center', fontsize=12)
                shape_idx += 1
    save_figure_png_pdf(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
    plt.close()


def main():
    # Data directory
    obs_dir = "../data"
    cmip6_dir = "../data/cmip6"
    out_dir = "../plot"
    
    # Output path
    output_path = os.path.join(out_dir, "fig3.png")
    
    # Plot combined scatter figure
    draw_combined_scatter_plot(obs_dir, cmip6_dir, output_path)


if __name__ == "__main__":
    main()
