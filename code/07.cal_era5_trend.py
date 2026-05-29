import os
import argparse
import numpy as np
import h5py
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


def load_merged_fields(era5_dir: str, start: int, end: int):
    file_cwv_sst = os.path.join(
        era5_dir, f"era5_cwv_sst_s{start:06d}_e{end:06d}.nc"
    )
    if not os.path.exists(file_cwv_sst):
        raise FileNotFoundError(file_cwv_sst)

    with Dataset(file_cwv_sst, mode="r") as ds1:
        time = np.array(ds1.variables["time"][:], dtype=np.int32)
        latitude = ds1.variables["latitude"][:]
        longitude = ds1.variables["longitude"][:]
        sst = np.array(ds1.variables["sst"][:], dtype=np.float32)
        cwv = np.array(ds1.variables["cwv"][:], dtype=np.float32)
    return time, latitude, longitude, cwv, sst


def load_era5_aquiv_temps(aquiv_dir: str, time: np.ndarray, target_shape):
    """
    Load monthly ERA5 aquiv H5 files and compute TMT/TLT.

    Files: ERA5_BT_Ch6_Ch8_Ch10_YYYYMM.h5
    Variables:
      - bt_ch6: tmt (raw)
      - bt_ch8: tut
      - bt_ch10: tls

    Formulas (same as merra2_equiv usage):
      - tmt = 1.1*tmt_raw - 0.1*tls
      - tlt = 1.430*tmt - 0.462*tut + 0.032*tls
    """
    ntime = len(time)
    tmt = np.full((ntime, *target_shape), np.nan, dtype=np.float32)
    tlt = np.full((ntime, *target_shape), np.nan, dtype=np.float32)

    for i, yyyymm in enumerate(time.astype(np.int32)):
        file_path = os.path.join(aquiv_dir, f"ERA5_BT_Ch6_Ch8_Ch10_{yyyymm:06d}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with h5py.File(file_path, mode="r") as f:
            bt_ch6 = np.array(f["bt_ch6"][:], dtype=np.float32)   # tmt raw
            bt_ch8 = np.array(f["bt_ch8"][:], dtype=np.float32)   # tut
            bt_ch10 = np.array(f["bt_ch10"][:], dtype=np.float32) # tls

        if bt_ch6.shape != target_shape:
            if bt_ch6.T.shape == target_shape:
                bt_ch6 = bt_ch6.T
                bt_ch8 = bt_ch8.T
                bt_ch10 = bt_ch10.T
            else:
                raise ValueError(
                    f"Shape mismatch in {file_path}: got {bt_ch6.shape}, expected {target_shape}"
                )

        tmt_i = 1.1 * bt_ch6 - 0.1 * bt_ch10
        tlt_i = 1.430 * tmt_i - 0.462 * bt_ch8 + 0.032 * bt_ch10

        tmt[i] = tmt_i.astype(np.float32)
        tlt[i] = tlt_i.astype(np.float32)

    return tlt, tmt


def main():
    parser = argparse.ArgumentParser(
        description="Compute 20S-20N ocean-only area-mean trends (CWV, SST, TLT, TMT) from ERA5 merged outputs"
    )
    parser.add_argument("--dir", dest="era5_dir", default="../reanalysis/era5")
    parser.add_argument("--start", dest="start", type=int, default=198001)
    parser.add_argument("--end", dest="end", type=int, default=202412)
    parser.add_argument(
        "--aquiv_dir",
        dest="aquiv_dir",
        default="../reanalysis/era5",
        help="Directory containing monthly ERA5_BT_Ch6_Ch8_Ch10_YYYYMM.h5 files",
    )
    args = parser.parse_args()

    time, latitude, longitude, cwv, sst = load_merged_fields(
        args.era5_dir, args.start, args.end
    )
    tlt, tmt = load_era5_aquiv_temps(args.aquiv_dir, time, cwv.shape[1:])

    # Grid and mask
    sample_shape = cwv.shape[1:]
    lat2d, lon2d = ensure_2d_latlon(np.array(latitude), np.array(longitude), sample_shape)
    lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)

    # Region 20S–20N
    lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
    region_mask = lat_band & ocean_mask

    # Weights
    weights = calc_area_weights(lat2d)

    # Time series
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

    # Save to txt (n x 5)
    out_dir = "../data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"era5_s{args.start:06d}_e{args.end:06d}.txt")
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

    # Anomaly and trends
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
    slope_cwv = np.nan if not np.isfinite(cwv_mean) or abs(cwv_mean) < 1e-12 else slope_cwv_abs * 100.0 / cwv_mean

    print("ERA5 20S–20N ocean-only trends (area-weighted):")
    print(f"Time range: {int(time[0])}–{int(time[-1])}  (n={len(time)} months)")
    def fmt(val):
        return "nan" if not np.isfinite(val) else f"{val:.6g}"
    print(f"  CWV  slope: {fmt(slope_cwv)} (% / year), {fmt(slope_cwv*10)} per decade")
    print(f"  SST  slope: {fmt(slope_sst)} (K / year), {fmt(slope_sst*10)} per decade")
    print(f"  TLT  slope: {fmt(slope_tlt)} (K / year), {fmt(slope_tlt*10)} per decade")
    print(f"  TMT  slope: {fmt(slope_tmt)} (K / year), {fmt(slope_tmt*10)} per decade")


if __name__ == "__main__":
    main()


