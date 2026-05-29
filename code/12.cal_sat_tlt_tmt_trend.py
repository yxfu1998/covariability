import os
import argparse
import numpy as np
from netCDF4 import Dataset, num2date
from global_land_mask import globe
from scipy.stats import linregress


def ensure_2d_latlon(latitude: np.ndarray, longitude: np.ndarray, shape_2d):
    if latitude.ndim == 2 and longitude.ndim == 2:
        return latitude, longitude
    if latitude.ndim == 1 and longitude.ndim == 1:
        lat2d = np.repeat(latitude.reshape(-1, 1), longitude.shape[0], axis=1)
        lon2d = np.repeat(longitude.reshape(1, -1), latitude.shape[0], axis=0)
        if lat2d.shape != shape_2d:
            lat2d_t = np.repeat(latitude.reshape(1, -1), longitude.shape[0], axis=0)
            lon2d_t = np.repeat(longitude.reshape(-1, 1), latitude.shape[0], axis=1)
            if lat2d_t.shape == shape_2d:
                return lat2d_t, lon2d_t
            return lat2d, lon2d
        return lat2d, lon2d
    raise ValueError("Latitude/longitude arrays must be 1D or 2D")


def calc_area_weights(lat2d: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat2d)).astype(np.float64)


def spatial_weighted_mean(field: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(field) & mask
    if not np.any(valid):
        return np.nan
    w = weights * valid
    return np.nansum(field * w) / np.nansum(w)


def to_yyyymm_from_time_var(time_var, time_vals) -> np.ndarray:
    units = getattr(time_var, 'units', None)
    calendar = getattr(time_var, 'calendar', 'standard')
    if units is None:
        # Assume time_vals are already YYYYMM integers
        return time_vals.astype(np.int32)
    
    # Handle RSS "months since 1978-1-1" format
    if units and "months since 1978" in units:
        # Convert months since 1978-1-1 to YYYYMM
        yyyymm = np.array([], dtype=np.int32)
        for month_offset in time_vals:
            # Calculate year and month from months since 1978-1-1
            total_months = int(month_offset)
            year = 1978 + (total_months // 12)
            month = (total_months % 12) + 1
            yyyymm = np.append(yyyymm, year * 100 + month)
        return yyyymm
    
    try:
        dates = num2date(time_vals, units=units, calendar=calendar)
        yyyymm = np.array([d.year * 100 + d.month for d in dates], dtype=np.int32)
        return yyyymm
    except:
        # Fallback: assume time_vals are YYYYMM integers
        return time_vals.astype(np.int32)


def calculate_monthly_anomaly(data: np.ndarray) -> np.ndarray:
    n = len(data)
    full_years = n // 12
    remaining = n % 12
    reshaped_data = data[: full_years * 12].reshape(full_years, 12)
    climatological_mean = np.nanmean(reshaped_data, axis=0)
    anomaly = reshaped_data - climatological_mean
    anomaly_flattened = anomaly.flatten()
    if remaining > 0:
        remaining_data = data[full_years * 12 :]
        remaining_anomaly = remaining_data - climatological_mean[:remaining]
        anomaly_flattened = np.concatenate((anomaly_flattened, remaining_anomaly))
    return anomaly_flattened


def series_regression(y: np.ndarray, x_years: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(x_years)
    if np.sum(ok) < 3:
        return np.nan, np.nan
    res = linregress(x_years[ok], y[ok])
    return res.slope, res.intercept


def load_satellite_data(filepath: str, var_name: str, lat_name: str, lon_name: str, time_name: str):
    """Load satellite TLT/TMT data from NetCDF file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    
    with Dataset(filepath, 'r') as ds:
        if var_name not in ds.variables:
            raise KeyError(f"Variable '{var_name}' not found in {filepath}")
        if lat_name not in ds.variables or lon_name not in ds.variables:
            raise KeyError(f"Latitude/longitude variables not found in {filepath}")
        if time_name not in ds.variables:
            raise KeyError(f"Time variable '{time_name}' not found in {filepath}")
        
        lat = ds.variables[lat_name][:]
        lon = ds.variables[lon_name][:]
        data = np.array(ds.variables[var_name][:], dtype=np.float32)
        time_var = ds.variables[time_name]
        time_vals = np.array(time_var[:])

        # Replace fill values and filter valid range
        fill_val = getattr(time_var, '_FillValue', None)
        if fill_val is not None:
            data = np.where(data == float(fill_val), np.nan, data)
        data = np.where(~np.isfinite(data) | (data < -100.0) | (data > 400.0), np.nan, data)
        
        # Convert time to YYYYMM
        yyyymm = to_yyyymm_from_time_var(time_var, time_vals)
        
    return yyyymm, lat, lon, data


def process_satellite_dataset(name: str, filepath: str, var_name: str, lat_name: str, lon_name: str, 
                            time_name: str, start_yyyymm: int, end_yyyymm: int, outdir: str,
                            tls_filepath: str = None, tls_var_name: str = None):
    """Process a single satellite dataset and save results.
    
    Parameters
    ----------
    tls_filepath : str, optional
        Path to TLS NetCDF file. If provided, TMT will be recalculated using formula: tmt = 1.1*tmt - 0.1*tls
    tls_var_name : str, optional
        Variable name for TLS data in the TLS file.
    """
    try:
        yyyymm, lat, lon, data = load_satellite_data(filepath, var_name, lat_name, lon_name, time_name)
        
        # Filter time range
        time_mask = (yyyymm >= start_yyyymm) & (yyyymm <= end_yyyymm)
        if not np.any(time_mask):
            print(f"[{name}] No data in time range {start_yyyymm}-{end_yyyymm}")
            return
        
        yyyymm = yyyymm[time_mask]
        data = data[time_mask]
        
        # If TMT dataset and TLS file provided, recalculate TMT
        if 'tmt' in name and tls_filepath is not None and tls_var_name is not None:
            print(f"[{name}] Loading TLS data from {tls_filepath}...")
            try:
                yyyymm_tls, lat_tls, lon_tls, tls_data = load_satellite_data(
                    tls_filepath, tls_var_name, lat_name, lon_name, time_name
                )
                
                # Filter TLS time range
                time_mask_tls = (yyyymm_tls >= start_yyyymm) & (yyyymm_tls <= end_yyyymm)
                if not np.any(time_mask_tls):
                    print(f"[{name}] Warning: No TLS data in time range {start_yyyymm}-{end_yyyymm}, skipping recalculation")
                else:
                    yyyymm_tls = yyyymm_tls[time_mask_tls]
                    tls_data = tls_data[time_mask_tls]
                    
                    # Find common time points
                    common_time = np.intersect1d(yyyymm, yyyymm_tls)
                    if len(common_time) == 0:
                        print(f"[{name}] Warning: No common time points between TMT and TLS, skipping recalculation")
                    else:
                        tmt_indices = np.where(np.isin(yyyymm, common_time))[0]
                        tls_indices = np.where(np.isin(yyyymm_tls, common_time))[0]
                        
                        # Ensure spatial dimensions match
                        if data.shape[1:] == tls_data.shape[1:]:
                            # Recalculate TMT: tmt = 1.1*tmt - 0.1*tls
                            print(f"[{name}] Recalculating TMT using formula: tmt = 1.1*tmt - 0.1*tls...")
                            data[tmt_indices] = 1.1 * data[tmt_indices] - 0.1 * tls_data[tls_indices]
                            
                            # Update yyyymm to common time
                            yyyymm = common_time
                            data = data[tmt_indices]
                        else:
                            print(f"[{name}] Warning: Spatial dimensions don't match (TMT: {data.shape[1:]}, TLS: {tls_data.shape[1:]}), skipping recalculation")
            except Exception as e:
                print(f"[{name}] Warning: Failed to load TLS data: {e}, using original TMT data")
        
        # Grid and mask
        sample_shape = data.shape[1:]
        lat2d, lon2d = ensure_2d_latlon(np.array(lat), np.array(lon), sample_shape)
        lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
        ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)
        lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
        region_mask = lat_band & ocean_mask
        weights = calc_area_weights(lat2d)
        
        # Time series
        ntime = data.shape[0]
        ts = np.full(ntime, np.nan, dtype=np.float64)
        for i in range(ntime):
            ts[i] = spatial_weighted_mean(data[i], region_mask, weights)
        
        # Save to txt
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, f"{name}_s{int(yyyymm[0]):06d}_e{int(yyyymm[-1]):06d}.txt")
        data_to_save = np.column_stack([yyyymm.astype(np.int32), ts.astype(float)])
        
        # Determine variable name based on dataset name
        if 'tlt' in name:
            var_name_header = "ts_tlt"
        elif 'tmt' in name:
            var_name_header = "ts_tmt"
        else:
            var_name_header = "ts_temp"
        
        np.savetxt(
            out_path,
            data_to_save,
            fmt=["%06d", "%.6f"],
            header=f"time(yyyymm) {var_name_header}",
            comments=""
        )
        
        # Trend on anomalies
        x_years = (yyyymm // 100) + ((yyyymm % 100) - 0.5) / 12.0
        ts_anom = calculate_monthly_anomaly(ts)
        slope, _ = series_regression(ts_anom, x_years)
        
        # Print mean TMT/TLT value
        mean_value = np.nanmean(ts)
        print(f"[{name}] {int(yyyymm[0])}-{int(yyyymm[-1])} n={len(yyyymm)}: mean={mean_value:.4f} K, trend={slope:.6g} K/yr; saved {out_path}")
                
        return
        
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return 


def main():
    parser = argparse.ArgumentParser(
        description="Compute 20S-20N ocean-only area-mean TLT/TMT trends for satellite datasets (STAR/RSS/UAH)"
    )
    parser.add_argument("--outdir", default="../data")
    parser.add_argument("--start", type=int, default=198101)
    parser.add_argument("--end", type=int, default=202412)
    args = parser.parse_args()
    
    # Dataset configurations based on NCL script
    # Format: (name, filepath, var_name, lat_name, lon_name, time_name, tls_filepath, tls_var_name)
    datasets = [
        # RSS TLT
        ("rss_tlt", "../cdr/RSS_Tb_Maps_ch_TLT_V4_0.nc", 
         "brightness_temperature", "latitude", "longitude", "months", None, None),
        # RSS TMT (with TLS recalculation)
        ("rss_tmt", "../cdr/RSS_Tb_Maps_ch_TMT_V4_0.nc",
         "brightness_temperature", "latitude", "longitude", "months",
         "../cdr/RSS_Tb_Maps_ch_TLS_V4_0.nc", "brightness_temperature"),
        # STAR TLT
        ("star_tlt", "../cdr/Mean-Layer-Temperature-NOAA_v05r00_TLT_S198101_E202507_C20250805.nc",
         "tcdr_MSU_AMSUA_ATMS_TLT_anomaly", "latitude", "longitude", "time", None, None),
        # STAR TMT (with TLS recalculation)
        ("star_tmt", "../cdr/Mean-Layer-Temperature-NOAA_v05r00_TMT_S197811_E202507_C20250805.nc", 
         "tcdr_MSU_AMSUA_ATMS_TMT_anomaly", "latitude", "longitude", "time",
         "../cdr/Mean-Layer-Temperature-NOAA_v05r00_TLS_S197812_E202507_C20250805.nc",
         "tcdr_MSU_AMSUA_ATMS_TLS_anomaly"),
        # UAH TLT
        ("uah_tlt", "../cdr/uah_tlt_tmt_tls_s197801_e202412.nc",
         "tlt", "latitude", "longitude", "time", None, None),
        # UAH TMT
        ("uah_tmt", "../cdr/uah_tlt_tmt_tls_s197801_e202412.nc",
         "tmt", "latitude", "longitude", "time", "../cdr/uah_tlt_tmt_tls_s197801_e202412.nc", "tls"),
    ]
    
    # Collect trends for 200301-202012 period
    trend_results = []
    
    for name, filepath, var_name, lat_name, lon_name, time_name, tls_filepath, tls_var_name in datasets:
        process_satellite_dataset(name, filepath, var_name, lat_name, lon_name, time_name, 
                                args.start, args.end, args.outdir, tls_filepath, tls_var_name)

if __name__ == "__main__":
    main()
