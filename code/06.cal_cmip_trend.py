import os
import argparse
import numpy as np
from netCDF4 import Dataset
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


def to_decimal_year(yyyymm: np.ndarray) -> np.ndarray:
    years = yyyymm // 100
    months = yyyymm % 100
    return years + (months - 0.5) / 12.0


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


def load_cmip_fields(cmip_dir: str, combination: str):
    """
    Read four fields output by process_cmip_1980_2024.py:
      prw->cwv, ts->sst, tlt, tmt
    File naming convention:
      {combination}_prw1980-2024.nc
      {combination}_ts1980-2024.nc
      {combination}_tlt1980-2024.nc
      {combination}_tmt1980-2024.nc
    """
    fname_map = {
        "cwv": f"{combination}_prw1980-2024.nc",
        "sst": f"{combination}_ts1980-2024.nc",
        "tlt": f"{combination}_tlt1980-2024.nc",
        "tmt": f"{combination}_tmt1980-2024.nc",
    }
    paths = {k: os.path.join(cmip_dir, v) for k, v in fname_map.items()}

    # Check file existence
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{k} file not found: {p}")

    with Dataset(paths["cwv"], mode="r") as ds_cwv:
        time = np.array(ds_cwv.variables["time"][:], dtype=np.int32)
        lat = np.array(ds_cwv.variables["lat"][:])
        lon = np.array(ds_cwv.variables["lon"][:])
        cwv = np.array(ds_cwv.variables["prw"][:], dtype=np.float32)

    with Dataset(paths["sst"], mode="r") as ds_sst:
        sst = np.array(ds_sst.variables["ts"][:], dtype=np.float32)

    with Dataset(paths["tlt"], mode="r") as ds_tlt:
        tlt = np.array(ds_tlt.variables["tlt"][:], dtype=np.float32)

    with Dataset(paths["tmt"], mode="r") as ds_tmt:
        tmt = np.array(ds_tmt.variables["tmt"][:], dtype=np.float32)

    # Return data and paths for per-variable coordinate reads
    return time, lat, lon, cwv, sst, tlt, tmt, paths


def build_region_and_weights(nc_path: str, var_name: str):
    """
    Build region_mask and area weights for a single variable file.
    Note: variables may use different lat/lon grids within the same model-ensemble.
    """
    with Dataset(nc_path, mode="r") as ds:
        lat = np.array(ds.variables["lat"][:])
        lon = np.array(ds.variables["lon"][:])
        data = np.array(ds.variables[var_name][:], dtype=np.float32)

    sample_shape = data.shape[1:]
    lat2d, lon2d = ensure_2d_latlon(np.array(lat), np.array(lon), sample_shape)
    lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)
    lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
    region_mask = lat_band & ocean_mask
    weights = calc_area_weights(lat2d)
    return region_mask, weights


def main():
    parser = argparse.ArgumentParser(description="Compute 20S-20N ocean-only area-mean CMIP trends (CWV/SST/TLT/TMT)")
    parser.add_argument("--cmip_dir", default="../cmip6", help="Directory of process_cmip_1980_2024.py output files")
    parser.add_argument("--combo_file", default="../cmip6/cmip_name.txt", help="Model-ensemble list file")
    parser.add_argument("--out_dir", default="../data/cmip6", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.combo_file, "r") as f:
        combinations = [line.strip() for line in f if line.strip()]

    print(f"Total combinations: {len(combinations)}")

    for combo in combinations:
        try:
            time, lat, lon, cwv, sst, tlt, tmt, paths = load_cmip_fields(
                args.cmip_dir, combo
            )
        except Exception as e:
            print(f"[{combo}] skip due to error: {e}")
            continue

        # Grid and mask: compute per variable on its own grid
        mask_cwv, w_cwv = build_region_and_weights(paths["cwv"], "prw")
        mask_sst, w_sst = build_region_and_weights(paths["sst"], "ts")
        mask_tlt, w_tlt = build_region_and_weights(paths["tlt"], "tlt")
        mask_tmt, w_tmt = build_region_and_weights(paths["tmt"], "tmt")

        ntime = cwv.shape[0]
        ts_cwv = np.full(ntime, np.nan, dtype=np.float64)
        ts_sst = np.full(ntime, np.nan, dtype=np.float64)
        ts_tlt = np.full(ntime, np.nan, dtype=np.float64)
        ts_tmt = np.full(ntime, np.nan, dtype=np.float64)

        for i in range(ntime):
            ts_cwv[i] = spatial_weighted_mean(cwv[i], mask_cwv, w_cwv)
            ts_sst[i] = spatial_weighted_mean(sst[i], mask_sst, w_sst)
            ts_tlt[i] = spatial_weighted_mean(tlt[i], mask_tlt, w_tlt)
            ts_tmt[i] = spatial_weighted_mean(tmt[i], mask_tmt, w_tmt)

        # Save time series
        out_path = os.path.join(
            args.out_dir,
            f"{combo}_s{int(time[0]):06d}_e{int(time[-1]):06d}.txt",
        )
        data_to_save = np.column_stack(
            [
                time.astype(np.int32),
                ts_cwv.astype(float),
                ts_sst.astype(float),
                ts_tlt.astype(float),
                ts_tmt.astype(float),
            ]
        )
        np.savetxt(
            out_path,
            data_to_save,
            fmt=["%06d", "%.6f", "%.6f", "%.6f", "%.6f"],
            header="time(yyyymm) ts_cwv ts_sst ts_tlt ts_tmt",
            comments="",
        )

        # Compute trends and print
        x_years = to_decimal_year(time)
        ts_cwv_anom = calculate_monthly_anomaly(ts_cwv)
        ts_sst_anom = calculate_monthly_anomaly(ts_sst)
        ts_tlt_anom = calculate_monthly_anomaly(ts_tlt)
        ts_tmt_anom = calculate_monthly_anomaly(ts_tmt)

        slope_cwv_abs, _ = series_regression(ts_cwv_anom, x_years)
        slope_sst, _ = series_regression(ts_sst_anom, x_years)
        slope_tlt, _ = series_regression(ts_tlt_anom, x_years)
        slope_tmt, _ = series_regression(ts_tmt_anom, x_years)

        cwv_mean = np.nanmean(ts_cwv)
        slope_cwv = (
            np.nan
            if not np.isfinite(cwv_mean) or abs(cwv_mean) < 1e-12
            else slope_cwv_abs * 100.0 / cwv_mean
        )

        def fmt(val):
            return "nan" if not np.isfinite(val) else f"{val:.6g}"

        print(f"[{combo}] time {int(time[0])}-{int(time[-1])} n={len(time)}")
        print(
            f"  CWV slope: {fmt(slope_cwv)} (%/yr), {fmt(slope_cwv*10)} per decade"
        )
        print(
            f"  SST slope: {fmt(slope_sst)} (K/yr), {fmt(slope_sst*10)} per decade"
        )
        print(
            f"  TLT slope: {fmt(slope_tlt)} (K/yr), {fmt(slope_tlt*10)} per decade"
        )
        print(
            f"  TMT slope: {fmt(slope_tmt)} (K/yr), {fmt(slope_tmt*10)} per decade"
        )


if __name__ == "__main__":
    main()


