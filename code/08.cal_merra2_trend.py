import os
import re
import argparse
import numpy as np
import h5py
from netCDF4 import Dataset
from global_land_mask import globe
from scipy.stats import linregress


def month_iter(start_yyyymm: int, end_yyyymm: int):
    start_year, start_month = divmod(start_yyyymm, 100)
    end_year, end_month = divmod(end_yyyymm, 100)
    months = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        months.append(year * 100 + month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def list_existing_merra2_2d_files(directory: str, yyyymm: int):
    pattern = f"MERRA2_*.instM_2d_asm_Nx.{yyyymm:06d}.nc4"
    results = []
    for filename in os.listdir(directory):
        if re.match(pattern.replace("*", r"\d+"), filename):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                results.append(path)
    return results


def to_decimal_year(yyyymm: np.ndarray) -> np.ndarray:
    years = yyyymm // 100
    months = yyyymm % 100
    return years + (months - 0.5) / 12.0


def build_ocean_mask_from_2d_file(filepath: str, prefer_vars=None) -> np.ndarray:
    if prefer_vars is None:
        prefer_vars = []
    with Dataset(filepath, mode="r") as ds:
        # Try fractional land first
        land_candidates = [
            "FRLAND", "frland", "land_fraction", "sftlf", "LANDMASK", "LSMASK"
        ]
        for var in land_candidates:
            if var in ds.variables:
                data = ds.variables[var][:]
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
                if data.ndim == 2:
                    # FRLAND [0..1] or [%]
                    arr = np.array(data, dtype=np.float32)
                    if np.nanmax(arr) > 1.5:
                        arr = arr / 100.0
                    return arr < 0.5
        # Fall back to SST-like variable that is NaN/masked over land
        sst_candidates = ["SST", "sst", "SEA_SURFACE_TEMPERATURE"] + list(prefer_vars)
        for var in sst_candidates:
            if var in ds.variables:
                data = ds.variables[var][:]
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
                if data.ndim == 2:
                    arr = np.array(data, dtype=np.float32)
                    return np.isfinite(arr) & (arr > 200.0) & (arr < 350.0)
    raise RuntimeError(f"Cannot derive ocean mask from file: {filepath}")


def ensure_2d_latlon(latitude: np.ndarray, longitude: np.ndarray, shape_2d):
    if latitude.ndim == 2 and longitude.ndim == 2:
        return latitude, longitude
    if latitude.ndim == 1 and longitude.ndim == 1:
        lat2d = np.repeat(latitude.reshape(-1, 1), longitude.shape[0], axis=1)
        lon2d = np.repeat(longitude.reshape(1, -1), latitude.shape[0], axis=0)
        if lat2d.shape != shape_2d:
            # try transposed
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


def series_regression(y: np.ndarray, x_years: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(x_years)
    if np.sum(ok) < 3:
        return np.nan, np.nan
    res = linregress(x_years[ok], y[ok])
    return res.slope, res.intercept


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


def load_merged_fields(merra2_dir: str, start: int, end: int):
    file_cwv_sst = os.path.join(
        merra2_dir, f"merra2_cwv_sst_s{start:06d}_e{end:06d}.nc"
    )

    if not os.path.exists(file_cwv_sst):
        raise FileNotFoundError(file_cwv_sst)

    with Dataset(file_cwv_sst, mode="r") as ds1:
        time = np.array(ds1.variables["time"][:], dtype=np.int32)
        latitude = ds1.variables["latitude"][:]
        longitude = ds1.variables["longitude"][:]
        cwv = np.array(ds1.variables["cwv"][:], dtype=np.float32)
        sst = np.array(ds1.variables["sst"][:], dtype=np.float32)
    return time, latitude, longitude, cwv, sst


def load_equiv_temps(equiv_dir: str, time: np.ndarray, target_shape):
    ntime = len(time)
    tmt = np.full((ntime, *target_shape), np.nan, dtype=np.float32)
    tlt = np.full((ntime, *target_shape), np.nan, dtype=np.float32)

    for i, yyyymm in enumerate(time.astype(np.int32)):
        file_path = os.path.join(equiv_dir, f"MERRA2_BT_Ch6_Ch8_Ch10_{yyyymm:06d}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with h5py.File(file_path, mode="r") as f:
            bt_ch6 = np.array(f["bt_ch6"][:], dtype=np.float32)   # tmt
            bt_ch8 = np.array(f["bt_ch8"][:], dtype=np.float32)   # tut
            bt_ch10 = np.array(f["bt_ch10"][:], dtype=np.float32) # tls

        # Keep orientation consistent with CWV/SST grid.
        if bt_ch6.shape != target_shape:
            if bt_ch6.T.shape == target_shape:
                bt_ch6 = bt_ch6.T
                bt_ch8 = bt_ch8.T
                bt_ch10 = bt_ch10.T
            else:
                raise ValueError(
                    f"Shape mismatch in {file_path}: "
                    f"got {bt_ch6.shape}, expected {target_shape}"
                )

        tmt_i = 1.1 * bt_ch6 - 0.1 * bt_ch10
        tlt_i = 1.430 * tmt_i - 0.462 * bt_ch8 + 0.032 * bt_ch10

        tmt[i] = tmt_i.astype(np.float32)
        tlt[i] = tlt_i.astype(np.float32)

    return tlt, tmt


def main():
    parser = argparse.ArgumentParser(
        description="Compute 20S-20N ocean-only area-mean trends (CWV, SST, TLT, TMT) from MERRA2 merged outputs"
    )
    parser.add_argument("--dir", dest="merra2_dir", default="../reanalysis/merra2")
    parser.add_argument("--start", dest="start", type=int, default=198001)
    parser.add_argument("--end", dest="end", type=int, default=202412)
    parser.add_argument(
        "--equiv_dir",
        dest="equiv_dir",
        default="../reanalysis/merra2",
        help="Directory containing monthly MERRA2_BT_Ch6_Ch8_Ch10_YYYYMM.h5 files",
    )
    parser.add_argument("--mask_yyyymm", dest="mask_yyyymm", type=int, default=None,
                        help="YYYYMM to pick a 2D asm file for deriving ocean mask; default=first time step")
    args = parser.parse_args()

    time, latitude, longitude, cwv, sst = load_merged_fields(
        args.merra2_dir, args.start, args.end
    )
    tlt, tmt = load_equiv_temps(args.equiv_dir, time, cwv.shape[1:])

    # Ensure 2D lat/lon matching data slices
    sample_shape = cwv.shape[1:]
    lat2d, lon2d = ensure_2d_latlon(np.array(latitude), np.array(longitude), sample_shape)

    # Normalize longitude to [-180, 180] for global_land_mask
    lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

    # Ocean mask from global_land_mask
    ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)

    # Latitude band 20S–20N
    lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
    region_mask = lat_band & ocean_mask

    # Weights
    weights = calc_area_weights(lat2d)

    # Build time series (area means)
    ntime = cwv.shape[0]
    ts_cwv = np.full(ntime, np.nan, dtype=np.float64)
    ts_sst = np.full(ntime, np.nan, dtype=np.float64)
    ts_tlt = np.full(ntime, np.nan, dtype=np.float64)
    ts_tmt = np.full(ntime, np.nan, dtype=np.float64)

    for i in range(ntime):
        ts_cwv[i] = spatial_weighted_mean(cwv[i], region_mask, weights)
        ts_sst[i] = spatial_weighted_mean(sst[i], region_mask, weights)
        ts_tlt[i] = spatial_weighted_mean(tlt[i], region_mask, weights)
        ts_tmt[i] = spatial_weighted_mean(tmt[i], region_mask, weights)

    # Save time series to txt (n x 5)
    out_dir = "../data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"merra2_s{args.start:06d}_e{args.end:06d}.txt")
    data_to_save = np.column_stack([
        time.astype(np.int32), ts_cwv.astype(float), ts_sst.astype(float), ts_tlt.astype(float), ts_tmt.astype(float)
    ])
    np.savetxt(
        out_path,
        data_to_save,
        fmt=["%06d", "%.6f", "%.6f", "%.6f", "%.6f"],
        header="time(yyyymm) ts_cwv ts_sst ts_tlt ts_tmt",
        comments=""
    )

    # Anomalies per calendar month
    x_years = to_decimal_year(time)
    ts_cwv_anom = calculate_monthly_anomaly(ts_cwv)
    ts_sst_anom = calculate_monthly_anomaly(ts_sst)
    ts_tlt_anom = calculate_monthly_anomaly(ts_tlt)
    ts_tmt_anom = calculate_monthly_anomaly(ts_tmt)

    # Trends on anomalies (per year)
    slope_cwv_abs, _ = series_regression(ts_cwv_anom, x_years)
    slope_sst, _ = series_regression(ts_sst_anom, x_years)
    slope_tlt, _ = series_regression(ts_tlt_anom, x_years)
    slope_tmt, _ = series_regression(ts_tmt_anom, x_years)

    # CWV percent relative to original mean (not anomaly mean)
    cwv_mean = np.nanmean(ts_cwv)
    slope_cwv = np.nan if not np.isfinite(cwv_mean) or abs(cwv_mean) < 1e-12 else slope_cwv_abs * 100.0 / cwv_mean

    # Report: per year and per decade
    print("MERRA2 20S–20N ocean-only trends (area-weighted):")
    print(f"Time range: {int(time[0])}–{int(time[-1])}  (n={len(time)} months)")
    def fmt(val):
        return "nan" if not np.isfinite(val) else f"{val:.6g}"
    print(f"  CWV  slope: {fmt(slope_cwv)} (% / year), {fmt(slope_cwv*10)} per decade")
    print(f"  SST  slope: {fmt(slope_sst)} (K / year), {fmt(slope_sst*10)} per decade")
    print(f"  TLT  slope: {fmt(slope_tlt)} (K / year), {fmt(slope_tlt*10)} per decade")
    print(f"  TMT  slope: {fmt(slope_tmt)} (K / year), {fmt(slope_tmt*10)} per decade")


if __name__ == "__main__":
    main()


