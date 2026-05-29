import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend to avoid Qt issues in WSL
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import odr

from supply_plot import save_figure_png_pdf


def odr_linear_regression(x, y):
    """Linear fit with ODR on given data
    
    Return values match linregress where possible:
        (slope, intercept, r_value, p_value, std_err)
    p_value is None here.
    """
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) < 2:
        return None, None, None, None, None

    # Linear model: y = B0 * x + B1
    def linear_model(B, x_in):
        return B[0] * x_in + B[1]

    model = odr.Model(linear_model)
    data = odr.Data(x_valid, y_valid)

    # Least-squares initial guess for ODR
    x_mean = np.mean(x_valid)
    y_mean = np.mean(y_valid)
    denom = np.sum((x_valid - x_mean) ** 2)
    if denom == 0:
        return None, None, None, None, None
    init_slope = np.sum((x_valid - x_mean) * (y_valid - y_mean)) / denom
    init_intercept = y_mean - init_slope * x_mean

    odr_obj = odr.ODR(data, model, beta0=[init_slope, init_intercept])
    out = odr_obj.run()

    slope = out.beta[0]
    intercept = out.beta[1]

    # correlation coefficient
    if len(x_valid) > 1:
        r_value = np.corrcoef(x_valid, y_valid)[0, 1]
    else:
        r_value = 0.0

    std_err_slope = out.sd_beta[0]

    return slope, intercept, r_value, None, std_err_slope


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


def load_cmip6_data(filepath, start_yyyymm=None, end_yyyymm=None):
    """Read TXT output from cal_cmip_trend.py (space-separated)
    
    File format:time(yyyymm) ts_cwv ts_sst ts_tlt ts_tmt
    
    Args:
        filepath: filepath
        start_yyyymm: start time (YYYYMM, e.g. 198001)
        end_yyyymm: end time (YYYYMM, e.g. 202412)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None, None

    try:
        if filepath == '../data/cmip6/CESM2-WACCM_r1i1p1f1_s198001_e202412.txt':
            print(filepath)
            return None, None, None

        if filepath == '../data/cmip6/E3SM-1-1-ECA_r1i1p1f1_s198001_e202412.txt':
            print(filepath)
            return None, None, None

        data = np.loadtxt(filepath, skiprows=1)

        yyyymm = data[:, 0].astype(int)  # time column
        cwv = data[:, 1]  # ts_cwv
        sst = data[:, 2]  # ts_sst

        mask = np.ones(len(yyyymm), dtype=bool)

        if start_yyyymm is not None:
            mask &= (yyyymm >= start_yyyymm)

        if end_yyyymm is not None:
            mask &= (yyyymm <= end_yyyymm)

        yyyymm = yyyymm[mask]
        sst = sst[mask]
        cwv = cwv[mask]

        if len(sst) == 0:
            return None, None, None

        sst_anomaly = calculate_monthly_anomaly(sst)
        cwv_anomaly = calculate_monthly_anomaly(cwv)

        # Convert CWV to percent(Same processing as original script)
        cwv_anomaly = cwv_anomaly * 100. / 41.

        # Extract years for return (from yyyymm)
        years = (yyyymm // 100).astype(float)

        return sst_anomaly, cwv_anomaly, years

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None, None


def calculate_sliding_trend_pairs(sst_anomaly, cwv_anomaly, window_length):
    """Sliding-window trend pairs for given L"""
    n = min(len(sst_anomaly), len(cwv_anomaly))
    if n < window_length:
        return np.array([]), np.array([])

    sst_slopes = []
    cwv_slopes = []

    time_axis = np.arange(window_length) / 120.0  # Time axis in decades

    for start_idx in range(0, n - window_length + 1):
        sst_segment = sst_anomaly[start_idx:start_idx + window_length]
        cwv_segment = cwv_anomaly[start_idx:start_idx + window_length]

        if np.any(np.isnan(sst_segment)) or np.any(np.isnan(cwv_segment)):
            continue

        sst_slope = linregress(time_axis, sst_segment)[0]
        cwv_slope = linregress(time_axis, cwv_segment)[0]

        sst_slopes.append(sst_slope)
        cwv_slopes.append(cwv_slope)

    if not sst_slopes:
        return np.array([]), np.array([])

    return np.array(sst_slopes), np.array(cwv_slopes)


def collect_trend_pairs_by_L(cmip6_dir, start_yyyymm, end_yyyymm, L_values):
    """Loop models; collect trend pairs per L"""
    txt_files = glob.glob(os.path.join(cmip6_dir, "*.txt"))

    if not txt_files:
        print(f"In directory {cmip6_dir} no txt files found")
        return {L: {"sst": np.array([]), "cwv": np.array([])} for L in L_values}

    results = {L: {"sst": [], "cwv": []} for L in L_values}
    for filepath in txt_files:
        sst_anomaly, cwv_anomaly, _ = load_cmip6_data(filepath, start_yyyymm, end_yyyymm)
        if sst_anomaly is None or cwv_anomaly is None:
            continue

        for L in L_values:
            window_length = L * 12
            sst_slopes, cwv_slopes = calculate_sliding_trend_pairs(sst_anomaly, cwv_anomaly, window_length)
            if len(sst_slopes) == 0:
                continue
            results[L]["sst"].append(sst_slopes)
            results[L]["cwv"].append(cwv_slopes)

    for L in L_values:
        if results[L]["sst"]:
            results[L]["sst"] = np.concatenate(results[L]["sst"])
            results[L]["cwv"] = np.concatenate(results[L]["cwv"])
        else:
            results[L]["sst"] = np.array([])
            results[L]["cwv"] = np.array([])

    return results

def compute_regression_stats(sst_trends, cwv_trends):
    """Regression slope, RMSE (CWV %), RMSE on SST scale (K), and CC from trend pairs.

    Regression CWV_trend = slope * SST_trend + intercept; RMSE is RMS of residuals (%).
    SST-scale RMSE: rmse_sst = rmse / |slope| (% to K).
    """
    if len(sst_trends) < 2 or len(cwv_trends) < 2:
        return np.nan, np.nan, np.nan, np.nan

    valid_mask = ~(np.isnan(sst_trends) | np.isnan(cwv_trends))
    if np.sum(valid_mask) < 2:
        return np.nan, np.nan, np.nan, np.nan

    sst_valid = sst_trends[valid_mask]
    cwv_valid = cwv_trends[valid_mask]

    # ODR regression on (sst_trends, cwv_trends)
    slope, intercept, r_value, _, _ = odr_linear_regression(sst_valid, cwv_valid)
    if slope is None or intercept is None or r_value is None:
        return np.nan, np.nan, np.nan, np.nan

    predictions = slope * sst_valid + intercept
    rmse = np.sqrt(np.mean((predictions - cwv_valid) ** 2))
    cc = r_value

    if abs(slope) < 1e-15:
        rmse_sst = np.nan
    else:
        rmse_sst = rmse / abs(slope)

    return slope, rmse, rmse_sst, cc


def draw_metrics_vs_L(cmip6_dir, output_path, start_yyyymm=200206, end_yyyymm=202412):
    """Plot L vs regression statistics"""
    L_values = list(range(5, 26))
    trend_pairs = collect_trend_pairs_by_L(cmip6_dir, start_yyyymm, end_yyyymm, L_values)

    slopes = []
    rmses = []
    rmses_sst = []
    ccs = []

    for L in L_values:
        sst_trends = trend_pairs[L]["sst"]
        cwv_trends = trend_pairs[L]["cwv"]
        slope, rmse, rmse_sst, cc = compute_regression_stats(sst_trends, cwv_trends)
        slopes.append(slope)
        rmses.append(rmse)
        rmses_sst.append(rmse_sst)
        ccs.append(cc)

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(L_values, slopes, marker='o')
    axes[0].set_ylabel('Slope (%/K)', fontsize=14, fontweight='bold')
    axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes, fontsize=14, fontweight='bold', verticalalignment='top')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(L_values, rmses, marker='o', color='tab:orange')
    axes[1].set_ylabel('RMSE (%)', fontsize=14, fontweight='bold')
    axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes, fontsize=14, fontweight='bold', verticalalignment='top')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].plot(L_values, rmses_sst, marker='o', color='tab:orange')
    axes[2].set_ylabel('RMSE (K)', fontsize=14, fontweight='bold')
    axes[2].text(
        0.02, 0.98, '(c)', transform=axes[2].transAxes, fontsize=14, fontweight='bold', verticalalignment='top'
    )
    axes[2].grid(True, linestyle='--', alpha=0.5)

    axes[3].plot(L_values, ccs, marker='o', color='tab:green')
    axes[3].set_ylabel('CC', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Years', fontsize=14, fontweight='bold')
    axes[3].text(0.02, 0.98, '(d)', transform=axes[3].transAxes, fontsize=14, fontweight='bold', verticalalignment='top')
    axes[3].grid(True, linestyle='--', alpha=0.5)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticks([5, 10, 15, 20, 25])
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

    plt.tight_layout()
    save_figure_png_pdf(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure; avoid plt.show() on non-GUI backend


def main():
    cmip6_dir = "../data/cmip6"
    out_dir = "../plot"
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "fig_s5.png")

    draw_metrics_vs_L(cmip6_dir, output_path, start_yyyymm=198001, end_yyyymm=202412)


if __name__ == "__main__":
    main()

