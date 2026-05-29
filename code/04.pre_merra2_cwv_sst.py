from netCDF4 import Dataset
import os
import re
from typing import List, Tuple
import numpy as np


def extract_yyyymm_from_filename(filename: str) -> int:
    """Extract YYYYMM as int from filename like 'MERRA2_100.instM_2d_asm_Nx.198001.nc4'.

    Returns
    -------
    int
        YYYYMM as integer (e.g., 198001). Raises ValueError if not found.
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

    Expected filename pattern: MERRA2_*.instM_2d_asm_Nx.YYYYMM.nc4
    """
    results: List[Tuple[int, str]] = []
    for yyyymm in month_iter(start_yyyymm, end_yyyymm):
        # Find matching file pattern
        pattern = f"MERRA2_*.instM_2d_asm_Nx.{yyyymm:06d}.nc4"
        for filename in os.listdir(directory):
            if re.match(pattern.replace("*", r"\d+"), filename):
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    results.append((yyyymm, path))
                    break  # stop after first matching file
    results.sort(key=lambda x: x[0])
    return results


def load_merra2_series(
    directory: str,
    start_yyyymm: int = 198001,
    end_yyyymm: int = 202412,
):
    """Load MERRA2 monthly fields and stack by time.

    Parameters
    ----------
    directory : str
        Directory containing files like 'MERRA2_*.instM_2d_asm_Nx.YYYYMM.nc4'.
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
    cwv : np.ndarray
        3D array with shape (time, lat, lon) for column water vapor (TQV).
    sst : np.ndarray
        3D array with shape (time, lat, lon) for sea surface temperature (TS).
    """
    month_files = list_existing_month_files(directory, start_yyyymm, end_yyyymm)
    if not month_files:
        raise FileNotFoundError(
            f"No files found in {directory} for months {start_yyyymm}-{end_yyyymm}"
        )

    time_list: List[int] = []
    cwv_slices: List[np.ndarray] = []
    sst_slices: List[np.ndarray] = []
    latitude = None
    longitude = None

    for yyyymm, filepath in month_files:
        with Dataset(filepath, mode="r") as ds:
            if latitude is None:
                # Try alternate latitude variable names
                lat_vars = ["lat", "latitude", "LAT", "Latitude"]
                lon_vars = ["lon", "longitude", "LON", "Longitude"]
                
                lat_var = None
                lon_var = None
                
                for var in lat_vars:
                    if var in ds.variables:
                        lat_var = var
                        break
                
                for var in lon_vars:
                    if var in ds.variables:
                        lon_var = var
                        break
                
                if lat_var is None or lon_var is None:
                    raise KeyError(
                        f"Missing latitude or longitude variables in file: {filepath}. "
                        f"Available variables: {list(ds.variables.keys())}"
                    )
                
                latitude = ds.variables[lat_var][:]
                longitude = ds.variables[lon_var][:]

            # Check TQV and TS variables
            if "TQV" not in ds.variables or "TS" not in ds.variables:
                raise KeyError(f"Missing 'TQV' or 'TS' in file: {filepath}")

            tqv_data = ds.variables["TQV"][:]  # Column water vapor
            ts_data = ds.variables["TS"][:]     # Sea surface temperature

            # Normalize shapes to 2D (lat, lon) per file, then stack as (time, lat, lon)
            if tqv_data.ndim == 3:
                if tqv_data.shape[0] == 1:
                    tqv_data = tqv_data[0]
                else:
                    raise ValueError(
                        f"Variable 'TQV' has unexpected shape {tqv_data.shape} in {filepath}. "
                        f"Expected 2D (lat, lon) or 3D with leading size 1."
                    )
            elif tqv_data.ndim != 2:
                raise ValueError(
                    f"Variable 'TQV' must be 2D or 3D (with leading size 1), got shape {tqv_data.shape}"
                )

            if ts_data.ndim == 3:
                if ts_data.shape[0] == 1:
                    ts_data = ts_data[0]
                else:
                    raise ValueError(
                        f"Variable 'TS' has unexpected shape {ts_data.shape} in {filepath}. "
                        f"Expected 2D (lat, lon) or 3D with leading size 1."
                    )
            elif ts_data.ndim != 2:
                raise ValueError(
                    f"Variable 'TS' must be 2D or 3D (with leading size 1), got shape {ts_data.shape}"
                )

            # Basic consistency check across months
            if cwv_slices:
                if tqv_data.shape != cwv_slices[0].shape:
                    raise ValueError(
                        "Inconsistent 'TQV' spatial shape across files: "
                        f"{tqv_data.shape} vs {cwv_slices[0].shape} (file: {filepath})"
                    )
                if ts_data.shape != sst_slices[0].shape:
                    raise ValueError(
                        "Inconsistent 'TS' spatial shape across files: "
                        f"{ts_data.shape} vs {sst_slices[0].shape} (file: {filepath})"
                    )

            time_list.append(yyyymm)
            cwv_slices.append(np.asarray(tqv_data))
            sst_slices.append(np.asarray(ts_data))

    time = np.array(time_list, dtype=np.int32)
    cwv = np.stack(cwv_slices, axis=0)
    sst = np.stack(sst_slices, axis=0)

    return time, latitude, longitude, cwv, sst


def save_merra2_to_netcdf(
    output_directory: str,
    start_yyyymm: int,
    end_yyyymm: int,
    time: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    cwv: np.ndarray,
    sst: np.ndarray,
):
    """Save the MERRA2 arrays into a single NetCDF file.

    Parameters
    ----------
    output_directory : str
        Directory to save the output file.
    start_yyyymm : int
        Start month in YYYYMM format.
    end_yyyymm : int
        End month in YYYYMM format.
    time : np.ndarray
        Time array.
    latitude : np.ndarray
        Latitude array.
    longitude : np.ndarray
        Longitude array.
    cwv : np.ndarray
        Column water vapor array (from TQV).
    sst : np.ndarray
        Sea surface temperature array (from TS).
        
    Returns
    -------
    str
        Path to the created NetCDF file.
    """
    os.makedirs(output_directory, exist_ok=True)
    outfile = os.path.join(
        output_directory, f"merra2_cwv_sst_s{start_yyyymm:06d}_e{end_yyyymm:06d}.nc"
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
        grid_shape = cwv.shape[1:]
        if latitude.shape != grid_shape or longitude.shape != grid_shape:
            raise ValueError(
                "When latitude/longitude are 2D, their shapes must match each CWV slice"
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
        v_cwv = ds.createVariable("cwv", np.float32, data_dims)
        v_sst = ds.createVariable("sst", np.float32, data_dims)
        
        # Minimal attributes
        v_time.long_name = "month in YYYYMM"
        v_lat.units = "degrees_north"
        v_lon.units = "degrees_east"
        v_cwv.long_name = "column_water_vapor"
        v_cwv.units = "kg m-2"
        v_sst.long_name = "sea_surface_temperature"
        v_sst.units = "K"
        
        # Data
        v_time[:] = np.asarray(time, dtype=np.int32)
        v_lat[:] = np.asarray(latitude, dtype=np.float32)
        v_lon[:] = np.asarray(longitude, dtype=np.float32)
        v_cwv[:] = np.asarray(cwv, dtype=np.float32)
        v_sst[:] = np.asarray(sst, dtype=np.float32)
    
    return outfile


if __name__ == "__main__":
    merra2_dir = "../reanalysis/merra2"
    start = 198001
    end = 202412
    print(f"Loading MERRA2 monthly files from {merra2_dir} ({start}-{end})...")
    time, latitude, longitude, cwv, sst = load_merra2_series(merra2_dir, start, end)

    print("Loaded months:", time.shape[0])
    print("First/last YYYYMM:", int(time[0]), int(time[-1]))
    print("Latitude shape:", np.shape(latitude))
    print("Longitude shape:", np.shape(longitude))
    print("CWV shape (time, lat, lon):", cwv.shape)
    print("SST shape (time, lat, lon):", sst.shape)

    # Save arrays to a single NetCDF file
    outfile = save_merra2_to_netcdf(
        output_directory=merra2_dir,
        start_yyyymm=start,
        end_yyyymm=end,
        time=time,
        latitude=latitude,
        longitude=longitude,
        cwv=cwv,
        sst=sst,
    )
    print(f"Written file: {outfile}")
