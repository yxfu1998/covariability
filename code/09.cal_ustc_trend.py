import os
import re
import argparse
import numpy as np
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


def ensure_2d_latlon(latitude: np.ndarray, longitude: np.ndarray, shape_2d):
    if latitude is None or longitude is None:
        return None, None
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
    return None, None


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


def guess_var(ds: Dataset, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute 20S-20N ocean-only area-mean trends for CWV/SST from a CDR file, and export time series to txt"
    )
    parser.add_argument("--file", dest="infile", default="../cdr/USTC_mean_TCWV_SST_CDR_S200206_E202412_C20260101.nc")
    parser.add_argument("--outdir", dest="outdir", default="../data")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise FileNotFoundError(args.infile)

    # Parse S/E from filename for output naming and time reconstruction
    base = os.path.basename(args.infile)
    m = re.search(r"S(\d{6}).*E(\d{6})", base)
    start_yyyymm = None
    end_yyyymm = None
    if m:
        start_yyyymm = int(m.group(1))
        end_yyyymm = int(m.group(2))

    with Dataset(args.infile, mode="r") as ds:
        # Identify variables
        lat_name = guess_var(ds, ["latitude", "lat", "LAT", "Latitude"])
        lon_name = guess_var(ds, ["longitude", "lon", "LON", "Longitude"])
        cwv_name = guess_var(ds, ["cwv", "tcwv", "Mean_CWV", "mean_cwv", "CWV", "TCWV"]) 
        sst_name = guess_var(ds, ["sst", "Mean_SST", "mean_sst", "SST"]) 
        time_name = guess_var(ds, ["time", "Time", "TIME", "yyyymm"]) 

        if cwv_name is None or sst_name is None:
            raise KeyError("Cannot find CWV/SST variables in the input file")

        cwv = np.array(ds.variables[cwv_name][:], dtype=float)
        sst = np.array(ds.variables[sst_name][:], dtype=float)
        # Replace missing flags with NaN
        cwv = np.where(cwv == -999.0, np.nan, cwv)
        sst = np.where(sst == -999.0, np.nan, sst)

        # Construct time (yyyymm)
        if start_yyyymm is not None and end_yyyymm is not None:
            time = np.array(month_iter(start_yyyymm, end_yyyymm), dtype=np.int32)
        elif time_name is not None:
            time_raw = np.array(ds.variables[time_name][:])
            # If already YYYYMM ints
            if np.issubdtype(time_raw.dtype, np.integer) and np.nanmax(time_raw) > 100000:
                time = time_raw.astype(np.int32)
            else:
                # Fallback: try length-based with unknown start
                if time_raw.ndim == 1:
                    n = time_raw.shape[0]
                    if start_yyyymm is not None:
                        time = np.array(month_iter(start_yyyymm, start_yyyymm + (n - 1)), dtype=np.int32)
                    else:
                        raise ValueError("Unable to construct YYYYMM time from file")
                else:
                    raise ValueError("Unsupported time variable format")
        else:
            raise ValueError("No time information available in filename or variables")

        # Shapes and grid
        latitude = ds.variables[lat_name][:] if lat_name else None
        longitude = ds.variables[lon_name][:] if lon_name else None

    # Normalize array order to (time, lat, lon) if input is (time, lon, lat)
    if (
        cwv.ndim == 3
        and sst.ndim == 3
        and latitude is not None
        and longitude is not None
        and latitude.ndim == 1
        and longitude.ndim == 1
    ):
        nlat = int(latitude.shape[0])
        nlon = int(longitude.shape[0])
        # If matches (time, lon, lat), transpose to (time, lat, lon)
        if cwv.shape[1] == nlon and cwv.shape[2] == nlat:
            cwv = np.transpose(cwv, (0, 2, 1))
        if sst.shape[1] == nlon and sst.shape[2] == nlat:
            sst = np.transpose(sst, (0, 2, 1))

    # If 3D time-lat-lon, compute 20S–20N ocean-only area mean; else use as-is
    ntime = cwv.shape[0]
    if cwv.ndim == 3 and sst.ndim == 3 and latitude is not None and longitude is not None:
        sample_shape = cwv.shape[1:]
        lat2d, lon2d = ensure_2d_latlon(np.array(latitude), np.array(longitude), sample_shape)
        lon2d_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
        ocean_mask = globe.is_ocean(lat2d, lon2d_norm).astype(bool)
        lat_band = (lat2d >= -20.0) & (lat2d <= 20.0)
        region_mask = lat_band & ocean_mask
        weights = calc_area_weights(lat2d)
        # weights[:] = 1.0

        ts_cwv = np.full(ntime, np.nan, dtype=np.float64)
        ts_sst = np.full(ntime, np.nan, dtype=np.float64)
        for i in range(ntime):
            ts_cwv[i] = spatial_weighted_mean(cwv[i], region_mask, weights)
            ts_sst[i] = spatial_weighted_mean(sst[i], region_mask, weights)
    elif cwv.ndim == 1 and sst.ndim == 1:
        ts_cwv = cwv.astype(np.float64)
        ts_sst = sst.astype(np.float64)
    else:
        raise ValueError("Unsupported variable shapes for CWV/SST in the CDR file")

    # Save txt (n x 3)
    os.makedirs(args.outdir, exist_ok=True)
    out_base = f"ustc_s{int(time[0]):06d}_e{int(time[-1]):06d}.txt"
    out_path = os.path.join(args.outdir, out_base)
    data_to_save = np.column_stack([
        time.astype(np.int32), ts_cwv.astype(float), ts_sst.astype(float)
    ])
    np.savetxt(
        out_path,
        data_to_save,
        fmt=["%06d", "%.6f", "%.6f"],
        header="time(yyyymm) ts_cwv ts_sst",
        comments=""
    )

    # Trends on anomalies
    x_years = to_decimal_year(time)
    ts_cwv_anom = calculate_monthly_anomaly(ts_cwv)
    ts_sst_anom = calculate_monthly_anomaly(ts_sst)
    slope_cwv_abs, _ = series_regression(ts_cwv_anom[7:223], x_years[7:223])
    slope_sst, _ = series_regression(ts_sst_anom[7:223], x_years[7:223])
    cwv_mean = np.nanmean(ts_cwv)
    slope_cwv = np.nan if not np.isfinite(cwv_mean) or abs(cwv_mean) < 1e-12 else slope_cwv_abs * 100.0 / cwv_mean

    print("CDR 20S–20N ocean-only trends (area-weighted if gridded):")
    print(f"Time range: {int(time[0])}–{int(time[-1])}  (n={len(time)} months)")
    def fmt(val):
        return "nan" if not np.isfinite(val) else f"{val:.6g}"
    print(f"  CWV  slope: {fmt(slope_cwv)} (% / year), {fmt(slope_cwv*10)} per decade")
    print(f"  SST  slope: {fmt(slope_sst)} (K / year), {fmt(slope_sst*10)} per decade")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()


