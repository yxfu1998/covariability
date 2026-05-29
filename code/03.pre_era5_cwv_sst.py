from netCDF4 import Dataset
import os
import re
from typing import List, Tuple

import numpy as np


def extract_yyyymm_from_filename(filename: str) -> int:
    """Extract YYYYMM as int from filename like 'ERA5_2d_196001.nc'.

    Returns
    -------
    int
        YYYYMM as integer (e.g., 196001). Raises ValueError if not found.
    """
    match = re.search(r"(\d{6})", os.path.basename(filename))
    if not match:
        raise ValueError(f"Cannot find YYYYMM in filename: {filename}")
    return int(match.group(1))


def month_iter(start_yyyymm: int, end_yyyymm: int) -> List[int]:
    """Generate list of months in YYYYMM between inclusive bounds."""
    start_year, start_month = divmod(start_yyyymm, 100)
    end_year, end_month = divmod(end_yyyymm, 100)

    months: List[int] = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        months.append(year * 100 + month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def list_existing_month_files(
    directory: str,
    start_yyyymm: int,
    end_yyyymm: int,
) -> List[Tuple[int, str]]:
    """List (YYYYMM, filepath) for files existing in directory in the range.

    Expected filename pattern: ERA5_2d_YYYYMM.nc
    """
    results: List[Tuple[int, str]] = []
    for yyyymm in month_iter(start_yyyymm, end_yyyymm):
        filename = f"ERA5_2d_{yyyymm:06d}.nc"
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            results.append((yyyymm, path))
    results.sort(key=lambda x: x[0])
    return results


def load_era5_series(
    directory: str,
    start_yyyymm: int = 198001,
    end_yyyymm: int = 202412,
):
    """Load ERA5 monthly fields and stack by time.

    Parameters
    ----------
    directory : str
        Directory containing files like 'ERA5_2d_YYYYMM.nc'.
    start_yyyymm : int
        Start month (inclusive) in YYYYMM.
    end_yyyymm : int
        End month (inclusive) in YYYYMM.

    Returns
    -------
    time : np.ndarray
        1D array of YYYYMM integers for loaded files.
    latitude : np.ndarray
        1D or 2D latitude array from files (copied from the first file).
    longitude : np.ndarray
        1D or 2D longitude array from files (copied from the first file).
    sst : np.ndarray
        3D array with shape (time, lat, lon).
    tcwv : np.ndarray
        3D array with shape (time, lat, lon).
    """
    month_files = list_existing_month_files(directory, start_yyyymm, end_yyyymm)
    if not month_files:
        raise FileNotFoundError(
            f"No files found in {directory} for months {start_yyyymm}-{end_yyyymm}"
        )

    time_list: List[int] = []
    sst_slices: List[np.ndarray] = []
    tcwv_slices: List[np.ndarray] = []
    latitude = None
    longitude = None

    for yyyymm, filepath in month_files:
        with Dataset(filepath, mode="r") as ds:
            if latitude is None:
                if "latitude" not in ds.variables or "longitude" not in ds.variables:
                    raise KeyError(
                        f"Missing 'latitude' or 'longitude' in file: {filepath}"
                    )
                latitude = ds.variables["latitude"][:]
                longitude = ds.variables["longitude"][:]

            if "sst" not in ds.variables or "tcwv" not in ds.variables:
                raise KeyError(f"Missing 'sst' or 'tcwv' in file: {filepath}")

            sst_data = ds.variables["sst"][:]
            tcwv_data = ds.variables["tcwv"][:]

            # Normalize shapes to 2D (lat, lon) per file, then stack as (time, lat, lon)
            if sst_data.ndim == 3:
                if sst_data.shape[0] == 1:
                    sst_data = sst_data[0]
                else:
                    raise ValueError(
                        f"Variable 'sst' has unexpected shape {sst_data.shape} in {filepath}. "
                        f"Expected 2D (lat, lon) or 3D with leading size 1."
                    )
            elif sst_data.ndim != 2:
                raise ValueError(
                    f"Variable 'sst' must be 2D or 3D (with leading size 1), got shape {sst_data.shape}"
                )

            if tcwv_data.ndim == 3:
                if tcwv_data.shape[0] == 1:
                    tcwv_data = tcwv_data[0]
                else:
                    raise ValueError(
                        f"Variable 'tcwv' has unexpected shape {tcwv_data.shape} in {filepath}. "
                        f"Expected 2D (lat, lon) or 3D with leading size 1."
                    )
            elif tcwv_data.ndim != 2:
                raise ValueError(
                    f"Variable 'tcwv' must be 2D or 3D (with leading size 1), got shape {tcwv_data.shape}"
                )

            # Basic consistency check across months
            if sst_slices:
                if sst_data.shape != sst_slices[0].shape:
                    raise ValueError(
                        "Inconsistent 'sst' spatial shape across files: "
                        f"{sst_data.shape} vs {sst_slices[0].shape} (file: {filepath})"
                    )
                if tcwv_data.shape != tcwv_slices[0].shape:
                    raise ValueError(
                        "Inconsistent 'tcwv' spatial shape across files: "
                        f"{tcwv_data.shape} vs {tcwv_slices[0].shape} (file: {filepath})"
                    )

            time_list.append(yyyymm)
            sst_slices.append(np.asarray(sst_data))
            tcwv_slices.append(np.asarray(tcwv_data))

    time = np.array(time_list, dtype=np.int32)
    sst = np.stack(sst_slices, axis=0)
    tcwv = np.stack(tcwv_slices, axis=0)

    return time, latitude, longitude, sst, tcwv


def save_era5_5vars_to_netcdf(
    output_directory: str,
    start_yyyymm: int,
    end_yyyymm: int,
    time: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    sst: np.ndarray,
    tcwv: np.ndarray,
):
    """Save the five arrays into a single NetCDF file in the same directory.

    Notes
    -----
    - Filename: era5_cwv_sst_s{start}_e{end}.nc (all lowercase)
    - Variable names: time, latitude, longitude, sst, cwv (lowercase; tcwv -> cwv)
    - Supports either 1D (lat, lon) or 2D lat/lon grids
    """
    os.makedirs(output_directory, exist_ok=True)
    outfile = os.path.join(
        output_directory, f"era5_cwv_sst_s{start_yyyymm:06d}_e{end_yyyymm:06d}.nc"
    )

    time_len = int(time.shape[0])

    # Infer spatial dims and choose dimension names
    if latitude.ndim == 1 and longitude.ndim == 1:
        lat_name, lon_name = "latitude", "longitude"
        lat_len = int(latitude.shape[0])
        lon_len = int(longitude.shape[0])
        data_dims = ("time", lat_name, lon_name)
        lat_dims = (lat_name,)
        lon_dims = (lon_name,)
    else:
        # Treat as rectified 2D grid
        y_name, x_name = "y", "x"
        grid_shape = sst.shape[1:]
        if latitude.shape != grid_shape or longitude.shape != grid_shape:
            raise ValueError(
                "When latitude/longitude are 2D, their shapes must match each SST slice"
            )
        data_dims = ("time", y_name, x_name)
        lat_dims = (y_name, x_name)
        lon_dims = (y_name, x_name)
        lat_len, lon_len = grid_shape

    with Dataset(outfile, mode="w") as ds:
        # Dimensions
        ds.createDimension("time", time_len)
        if len(lat_dims) == 1:
            ds.createDimension(lat_dims[0], lat_len)
            ds.createDimension(lon_dims[0], lon_len)
        else:
            ds.createDimension(lat_dims[0], lat_len)
            ds.createDimension(lat_dims[1], lon_len)

        # Variables
        v_time = ds.createVariable("time", "i4", ("time",))
        v_lat = ds.createVariable("latitude", np.float32, lat_dims)
        v_lon = ds.createVariable("longitude", np.float32, lon_dims)
        v_sst = ds.createVariable("sst", np.float32, data_dims)
        v_cwv = ds.createVariable("cwv", np.float32, data_dims)

        # Minimal attributes
        v_time.long_name = "month in YYYYMM"
        v_lat.units = "degrees_north"
        v_lon.units = "degrees_east"
        v_sst.long_name = "sea_surface_temperature"
        v_cwv.long_name = "total_column_water_vapor"

        # Data
        v_time[:] = np.asarray(time, dtype=np.int32)
        v_lat[:] = np.asarray(latitude, dtype=np.float32)
        v_lon[:] = np.asarray(longitude, dtype=np.float32)
        v_sst[:] = np.asarray(sst, dtype=np.float32)
        v_cwv[:] = np.asarray(tcwv, dtype=np.float32)

    return outfile


if __name__ == "__main__":
    era5_dir = "../reanalysis/era5"
    start = 198001
    end = 202412

    print(f"Loading ERA5 monthly files from {era5_dir} ({start}-{end})...")
    time, latitude, longitude, sst, tcwv = load_era5_series(era5_dir, start, end)

    print("Loaded months:", time.shape[0])
    print("First/last YYYYMM:", int(time[0]), int(time[-1]))
    print("Latitude shape:", np.shape(latitude))
    print("Longitude shape:", np.shape(longitude)) 
    print("SST shape (time, lat, lon):", sst.shape)
    print("TCWV shape (time, lat, lon):", tcwv.shape)

    # Save five arrays to a single NetCDF file (tcwv -> cwv, lowercase)
    outfile = save_era5_5vars_to_netcdf(
        output_directory=era5_dir,
        start_yyyymm=start,
        end_yyyymm=end,
        time=time,
        latitude=latitude,
        longitude=longitude,
        sst=sst,
        tcwv=tcwv,
    )
    print(f"Written file: {outfile}")


