import os
import numpy as np
from netCDF4 import Dataset
from scipy.stats import linregress
from global_land_mask import globe
import argparse


def ensure_2d_latlon(lat: np.ndarray, lon: np.ndarray, sample_shape: tuple) -> tuple:
    """Ensure lat/lon are 2D arrays matching the sample shape."""
    if lat.ndim == 1 and lon.ndim == 1:
        if len(lat) == sample_shape[0] and len(lon) == sample_shape[1]:
            lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
        else:
            lat2d, lon2d = np.meshgrid(lat, lon)
    else:
        lat2d, lon2d = lat, lon
    return lat2d, lon2d


def calc_area_weights(lat2d: np.ndarray) -> np.ndarray:
    """Calculate area weights based on cosine of latitude."""
    return np.cos(np.radians(lat2d))


def spatial_weighted_mean(field: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    """Calculate spatial weighted mean over masked region."""
    valid = np.isfinite(field) & mask
    if not np.any(valid):
        return np.nan
    w = weights * valid
    return np.nansum(field * w) / np.nansum(w)


def fractional_year_to_yyyymm(fractional_years: np.ndarray) -> np.ndarray:
    """Convert fractional years to YYYYMM format."""
    yyyymm = np.array([], dtype=np.int32)
    for fy in fractional_years:
        year = int(fy)
        month = int((fy - year) * 12) + 1
        yyyymm = np.append(yyyymm, year * 100 + month)
    return yyyymm


def calculate_monthly_anomaly(data: np.ndarray) -> np.ndarray:
    """Calculate monthly anomalies by subtracting climatological mean."""
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
    """Calculate linear regression using scipy.stats.linregress."""
    ok = np.isfinite(y) & np.isfinite(x_years)
    if np.sum(ok) < 3:
        return np.nan, np.nan
    res = linregress(x_years[ok], y[ok])
    return res.slope, res.intercept


def load_rss_cwv_data(filepath: str):
    """Load RSS CWV data from NetCDF file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    
    with Dataset(filepath, 'r') as ds:
        if 'merged_vapor' not in ds.variables:
            raise KeyError(f"Variable 'merged_vapor' not found in {filepath}")
        if 'latitude' not in ds.variables or 'longitude' not in ds.variables:
            raise KeyError(f"Latitude/longitude variables not found in {filepath}")
        if 'time' not in ds.variables:
            raise KeyError(f"Time variable not found in {filepath}")
        
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        cwv = np.array(ds.variables['merged_vapor'][:], dtype=np.float32)
        time_vals = np.array(ds.variables['time'][:])
        
        # Handle missing values
        fill_val = getattr(ds.variables['merged_vapor'], '_FillValue', None)
        if fill_val is not None:
            cwv = np.where(cwv == float(fill_val), np.nan, cwv)
        
        # Filter valid range (CWV should be positive and reasonable)
        cwv = np.where(~np.isfinite(cwv) | (cwv < 0.0) | (cwv > 100.0), np.nan, cwv)
        
        # Transpose from (lat, lon, time) to (time, lat, lon)
        if cwv.ndim == 3 and cwv.shape[0] == len(lat) and cwv.shape[1] == len(lon):
            cwv = np.transpose(cwv, (2, 0, 1))  # (time, lat, lon)
        
        # Convert fractional year to YYYYMM
        yyyymm = fractional_year_to_yyyymm(time_vals)
        
    return yyyymm, lat, lon, cwv


def main():
    parser = argparse.ArgumentParser(
        description="Compute 20S-20N ocean-only area-mean CWV trend for RSS data"
    )
    parser.add_argument("--input", default="../cdr/merged_vapor_1988-2024.nc")
    parser.add_argument("--outdir", default="../data")
    parser.add_argument("--start", type=int, default=198801)
    parser.add_argument("--end", type=int, default=202412)
    args = parser.parse_args()
    
    try:
        print(f"Loading RSS CWV data from {args.input}...")
        yyyymm, lat, lon, cwv = load_rss_cwv_data(args.input)
        
        # Filter time range
        time_mask = (yyyymm >= args.start) & (yyyymm <= args.end)
        if not np.any(time_mask):
            print(f"No data in time range {args.start}-{args.end}")
            return
        
        yyyymm = yyyymm[time_mask]
        cwv = cwv[time_mask]
        
        print(f"Time range: {int(yyyymm[0])} to {int(yyyymm[-1])}")
        print(f"Data shape: {cwv.shape}")
        
        # Grid and mask
        sample_shape = cwv.shape[1:]
        lat2d, lon2d = ensure_2d_latlon(np.array(lat), np.array(lon), sample_shape)
        lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
        ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)
        lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
        region_mask = lat_band & ocean_mask
        weights = calc_area_weights(lat2d)
        
        # Calculate time series
        ntime = cwv.shape[0]
        ts_cwv = np.full(ntime, np.nan, dtype=np.float64)
        for i in range(ntime):
            ts_cwv[i] = spatial_weighted_mean(cwv[i], region_mask, weights)
        
        # Save time series to txt
        os.makedirs(args.outdir, exist_ok=True)
        output_filename = os.path.join(
            args.outdir, f"rss_cwv_s{int(yyyymm[0]):06d}_e{int(yyyymm[-1]):06d}.txt"
        )
        data_to_save = np.column_stack([
            yyyymm.astype(np.int32),
            ts_cwv.astype(float)
        ])
        np.savetxt(
            output_filename,
            data_to_save,
            fmt=["%06d", "%.6f"],
            header="time(yyyymm) ts_cwv",
            comments=""
        )
        
        # Calculate trend on anomalies
        x_years = (yyyymm // 100) + ((yyyymm % 100) - 0.5) / 12.0
        ts_cwv_anom = calculate_monthly_anomaly(ts_cwv)
        slope_cwv, _ = series_regression(ts_cwv_anom, x_years)
        
        # Calculate percentage trend
        mean_cwv = np.nanmean(ts_cwv)
        slope_cwv_pct = slope_cwv * 100.0 / mean_cwv if mean_cwv > 0 else np.nan
        
        print(f"RSS CWV trend: {slope_cwv:.6g} mm/yr ({slope_cwv_pct:.6g} %/yr)")
        print(f"Mean CWV: {mean_cwv:.6g} mm")
        print(f"Saved time series to: {output_filename}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
