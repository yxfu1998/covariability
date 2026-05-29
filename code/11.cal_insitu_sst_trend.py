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


def to_yyyymm_from_time_var(time_var) -> np.ndarray:
    units = getattr(time_var, 'units', None)
    calendar = getattr(time_var, 'calendar', 'standard')
    time_vals = np.array(time_var[:])
    if units is None:
        # Fallback: assume already YYYYMM
        return time_vals.astype(np.int32)
    dates = num2date(time_vals, units=units, calendar=calendar)
    yyyymm = np.array([d.year * 100 + d.month for d in dates], dtype=np.int32)
    return yyyymm


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


def normalize_grid_and_mask(lat, lon, field3d):
    sample_shape = field3d.shape[1:]
    lat2d, lon2d = ensure_2d_latlon(np.array(lat), np.array(lon), sample_shape)
    # If field is (time, lon, lat), transpose
    if field3d.shape[1] == len(lon) and field3d.shape[2] == len(lat):
        field3d = np.transpose(field3d, (0, 2, 1))
        sample_shape = field3d.shape[1:]
        lat2d, lon2d = ensure_2d_latlon(np.array(lat), np.array(lon), sample_shape)
    lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)
    lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
    region_mask = lat_band & ocean_mask
    weights = calc_area_weights(lat2d)
    return field3d, region_mask, weights


def read_sst_series(nc_path: str, var_candidates, lat_candidates, lon_candidates, time_name='time'):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)
    with Dataset(nc_path, 'r') as ds:
        var_name = next((v for v in var_candidates if v in ds.variables), None)
        if var_name is None:
            raise KeyError(f"No SST variable found in {nc_path}")
        lat_name = next((v for v in lat_candidates if v in ds.variables), None)
        lon_name = next((v for v in lon_candidates if v in ds.variables), None)
        if lat_name is None or lon_name is None:
            raise KeyError(f"Latitude/longitude not found in {nc_path}")

        lat = ds.variables[lat_name][:]
        lon = ds.variables[lon_name][:]
        sst = np.array(ds.variables[var_name][:], dtype=float)
        # Replace common fill values
        fill_val = getattr(ds.variables[var_name], '_FillValue', None)
        if fill_val is not None:
            sst = np.where(sst == float(fill_val), np.nan, sst)
        sst = np.where(~np.isfinite(sst) | (sst < -100.0) | (sst > 400.0), np.nan, sst)

        time_var = ds.variables[time_name] if time_name in ds.variables else None
        if time_var is None:
            raise KeyError(f"time variable not found in {nc_path}")
        yyyymm = to_yyyymm_from_time_var(time_var)

    # Grid and mask
    sst, region_mask, weights = normalize_grid_and_mask(lat, lon, sst)

    # Area mean series
    ntime = sst.shape[0]
    ts = np.full(ntime, np.nan, dtype=np.float64)
    for i in range(ntime):
        ts[i] = spatial_weighted_mean(sst[i], region_mask, weights)
    return yyyymm, ts


def save_txt(outdir, name, yyyymm, ts):
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{name}_s{int(yyyymm[0]):06d}_e{int(yyyymm[-1]):06d}.txt")
    data_to_save = np.column_stack([yyyymm.astype(np.int32), ts.astype(float)])
    np.savetxt(
        out_path,
        data_to_save,
        fmt=["%06d", "%.6f"],
        header="time(yyyymm) ts_sst",
        comments=""
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compute 20S-20N ocean mean SST trends for COBE/ERSST/HadSST/HadISST and export longest series")
    parser.add_argument("--outdir", default="../data")
    parser.add_argument("--cobe", default="../cdr/COBE.nc")
    parser.add_argument("--ersst", default="../cdr/ERSST.nc")
    parser.add_argument("--hadisst", default="../cdr/HadISST.nc")
    parser.add_argument("--hadsst", default="../cdr/HadSST.nc")
    parser.add_argument("--oisst", default="../cdr/OISST.nc")
    args = parser.parse_args()

    datasets = [
        ("cobe", args.cobe, ["sst", "SST"], ["lat", "latitude"], ["lon", "longitude"], "time"),
        ("ersst", args.ersst, ["sst", "SST"], ["lat", "latitude"], ["lon", "longitude"], "time"),
        ("hadisst", args.hadisst, ["sst", "SST"], ["lat", "latitude"], ["lon", "longitude"], "time"),
        ("hadsst", args.hadsst, ["tos", "sst", "SST"], ["lat", "latitude"], ["lon", "longitude"], "time"),
        ("oisst", args.oisst, ["sst"], ["lat"], ["lon"], "time"),
    ]

    for name, path, var_cands, lat_cands, lon_cands, tname in datasets:
        try:
            yyyymm, ts = read_sst_series(path, var_cands, lat_cands, lon_cands, time_name=tname)
        except Exception as e:
            print(f"[{name}] skipped: {e}")
            continue

        # Save full available series
        out_txt = save_txt(args.outdir, name, yyyymm, ts)

        # Trend on anomalies over full period
        x_years = (yyyymm // 100) + ((yyyymm % 100) - 0.5) / 12.0
        ts_anom = calculate_monthly_anomaly(ts)
        slope, _ = series_regression(ts_anom, x_years)
        print(f"[{name}] {int(yyyymm[0])}-{int(yyyymm[-1])} n={len(yyyymm)}: trend={slope:.6g} K/yr; saved {out_txt}")


if __name__ == "__main__":
    main()


