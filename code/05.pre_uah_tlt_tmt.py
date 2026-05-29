import os
import re
import numpy as np
from netCDF4 import Dataset
from typing import List, Tuple


def extract_year_from_filename(filename: str) -> int:
    """Extract year from filename like 'tltmonamg.1978_6.1'."""
    match = re.search(r'\.(\d{4})_', filename)
    if not match:
        raise ValueError(f"Cannot find year in filename: {filename}")
    return int(match.group(1))


def list_existing_year_files(directory: str, start_year: int, end_year: int) -> List[Tuple[int, str]]:
    """List (year, filepath) for files existing in directory in the range."""
    results: List[Tuple[int, str]] = []
    if not os.path.exists(directory):
        return results
    
    for filename in os.listdir(directory):
        if filename.startswith(('tltmonamg.', 'tmtmonamg.', 'tlsmonamg.')) and filename.endswith('.1'):
            try:
                year = extract_year_from_filename(filename)
                if start_year <= year <= end_year:
                    path = os.path.join(directory, filename)
                    if os.path.isfile(path):
                        results.append((year, path))
            except ValueError:
                continue
    results.sort(key=lambda x: x[0])
    return results


def read_uah_monthly_file(filepath: str) -> np.ndarray:
    """Read UAH monthly anomaly file and return 144x72 grid data.
    
    Format: 144x72 grid, -9999 is missing, values in hundredths of degrees.
    Returns data in degrees Celsius (divided by 100).
    """
    data = np.full((12, 72, 144), np.nan, dtype=np.float32)
    
    with open(filepath, 'r') as f:
        for month in range(12):
            # Read header line: ISAT, IYR, IMON, ACHA
            header = f.readline().strip()
            if not header:
                break
            
            # Parse header (3 integers + 12 character string)
            parts = header.split()
            if len(parts) < 3:
                continue
            
            try:
                isat = int(parts[0])
                iyr = int(parts[1])
                imon = int(parts[2])
            except ValueError:
                continue

            print(isat, iyr, imon)
            
            # Read 144x72 grid data: 16 values per line, 648 lines total (144*72/16)
            month_data = np.full((72, 144), np.nan, dtype=np.float32)
            data_values = []
            
            # Read all 648 lines for this month
            for line_idx in range(648):
                line = f.readline().strip()
                if not line:
                    break
                
                # Parse 16 integers per line (I5 format)
                values = line.split()
                if len(values) >= 16:
                    for val_str in values[:16]:
                        try:
                            val = int(val_str)
                            if val != -9999:  # Missing value
                                data_values.append(val / 100.0)  # Convert to degrees
                            else:
                                data_values.append(np.nan)
                        except (ValueError, IndexError):
                            data_values.append(np.nan)
                else:
                    # Fill with NaN if line has insufficient values
                    data_values.extend([np.nan] * 16)
            
            # Reshape data to (72, 144) - lat, lon order
            if len(data_values) >= 72 * 144:
                month_data = np.array(data_values[:72*144]).reshape(72, 144)
            
            data[month] = month_data
    
    return data


def create_uah_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Create UAH grid coordinates (144x72, 2.5 degree grid, 'odd' grid).
    
    Grid center at 1.25 degrees different from integer multiples of 2.5.
    NDATA(1,1) = -178.75, -88.75
    NDATA(2,1) = -176.25, -88.75
    NDATA(1,2) = -178.75, -86.25
    """
    # Longitude: 144 points from -178.75 to 178.75
    lon = np.linspace(-178.75, 178.75, 144)
    
    # Latitude: 72 points from -88.75 to 88.75  
    lat = np.linspace(-88.75, 88.75, 72)
    
    return lat, lon


def load_uah_series(directory: str, start_year: int = 1978, end_year: int = 2023):
    """Load UAH monthly anomaly data and stack by time.
    
    Parameters
    ----------
    directory : str
        Directory containing UAH monthly files (tltmonamg.YYYY_6.1 or tmtmonamg.YYYY_6.1).
    start_year : int
        Start year (inclusive).
    end_year : int
        End year (inclusive).
        
    Returns
    -------
    time : np.ndarray
        1D array of YYYYMM integers for loaded files.
    latitude : np.ndarray
        1D latitude array (72 points).
    longitude : np.ndarray
        1D longitude array (144 points).
    data : np.ndarray
        3D array with shape (time, lat, lon) for temperature anomalies.
    """
    year_files = list_existing_year_files(directory, start_year, end_year)
    if not year_files:
        raise FileNotFoundError(
            f"No UAH files found in {directory} for years {start_year}-{end_year}"
        )
    
    time_list: List[int] = []
    data_slices: List[np.ndarray] = []
    latitude, longitude = create_uah_grid()
    
    for year, filepath in year_files:
        print(f"Reading {filepath}...")
        year_data = read_uah_monthly_file(filepath)
        
        # Stack 12 months for this year
        for month in range(12):
            if not np.all(np.isnan(year_data[month])):
                yyyymm = year * 100 + (month + 1)
                time_list.append(yyyymm)
                data_slices.append(year_data[month])
    
    time = np.array(time_list, dtype=np.int32)
    data = np.stack(data_slices, axis=0)
    
    return time, latitude, longitude, data


def save_uah_to_netcdf(output_directory: str, start_year: int, end_year: int,
                      time: np.ndarray, latitude: np.ndarray, longitude: np.ndarray,
                      tlt: np.ndarray, tmt: np.ndarray, tls: np.ndarray):
    #Save the UAH arrays into a single NetCDF file.
    os.makedirs(output_directory, exist_ok=True)
    outfile = os.path.join(
        output_directory, f"uah_tlt_tmt_tls_s{start_year:04d}01_e{end_year:04d}12.nc"
    )
    
    time_len = int(time.shape[0])
    lat_len = int(latitude.shape[0])
    lon_len = int(longitude.shape[0])
    
    with Dataset(outfile, mode="w") as ds:
        # Dimensions
        ds.createDimension("time", time_len)
        ds.createDimension("latitude", lat_len)
        ds.createDimension("longitude", lon_len)
        
        # Variables
        v_time = ds.createVariable("time", "i4", ("time",))
        v_lat = ds.createVariable("latitude", np.float32, ("latitude",))
        v_lon = ds.createVariable("longitude", np.float32, ("longitude",))
        v_tlt = ds.createVariable("tlt", np.float32, ("time", "latitude", "longitude"))
        v_tmt = ds.createVariable("tmt", np.float32, ("time", "latitude", "longitude"))
        v_tls = ds.createVariable("tls", np.float32, ("time", "latitude", "longitude"))
        # Attributes
        v_time.long_name = "month in YYYYMM"
        v_lat.units = "degrees_north"
        v_lon.units = "degrees_east"
        v_tlt.long_name = "lower_troposphere_temperature_anomaly"
        v_tlt.units = "degrees_Celsius"
        v_tmt.long_name = "middle_troposphere_temperature_anomaly"
        v_tmt.units = "degrees_Celsius"
        v_tls.long_name = "lower_stratosphere_temperature_anomaly"
        v_tls.units = "degrees_Celsius"
        # Data
        v_time[:] = np.asarray(time, dtype=np.int32)
        v_lat[:] = np.asarray(latitude, dtype=np.float32)
        v_lon[:] = np.asarray(longitude, dtype=np.float32)
        v_tlt[:] = np.asarray(tlt, dtype=np.float32)
        v_tmt[:] = np.asarray(tmt, dtype=np.float32)
        v_tls[:] = np.asarray(tls, dtype=np.float32)
    return outfile


def main():
    # Directories
    tlt_dir = "../cdr/uah/tlt"
    tmt_dir = "../cdr/uah/tmt"
    tls_dir = "../cdr/uah/tls"
    output_dir = "../cdr"
    
    start_year = 1978
    end_year = 2024
    
    print(f"Loading UAH TLT data from {tlt_dir} ({start_year}-{end_year})...")
    time_tlt, lat_tlt, lon_tlt, tlt_data = load_uah_series(tlt_dir, start_year, end_year)
    
    print(f"Loading UAH TMT data from {tmt_dir} ({start_year}-{end_year})...")
    time_tmt, lat_tmt, lon_tmt, tmt_data = load_uah_series(tmt_dir, start_year, end_year)
    
    print(f"Loading UAH TLS data from {tls_dir} ({start_year}-{end_year})...")
    time_tls, lat_tls, lon_tls, tls_data = load_uah_series(tls_dir, start_year, end_year)
    
    # Ensure time arrays match
    if not (np.array_equal(time_tlt, time_tmt) and np.array_equal(time_tlt, time_tls)):
        print("Warning: TLT, TMT, and TLS time arrays don't match, using intersection...")
        common_time = np.intersect1d(np.intersect1d(time_tlt, time_tmt), time_tls)
        tlt_indices = np.where(np.isin(time_tlt, common_time))[0]
        tmt_indices = np.where(np.isin(time_tmt, common_time))[0]
        tls_indices = np.where(np.isin(time_tls, common_time))[0]
        time = common_time
        tlt_data = tlt_data[tlt_indices]
        tmt_data = tmt_data[tmt_indices]
        tls_data = tls_data[tls_indices]
    else:
        time = time_tlt
    
    print("Loaded months:", time.shape[0])
    print("First/last YYYYMM:", int(time[0]), int(time[-1]))
    print("Latitude shape:", lat_tlt.shape)
    print("Longitude shape:", lon_tlt.shape)
    print("TLT shape (time, lat, lon):", tlt_data.shape)
    print("TMT shape (time, lat, lon):", tmt_data.shape)
    print("TLS shape (time, lat, lon):", tls_data.shape)
    
    # Save arrays to a single NetCDF file
    outfile = save_uah_to_netcdf(
        output_directory=output_dir,
        start_year=start_year,
        end_year=end_year,
        time=time,
        latitude=lat_tlt,
        longitude=lon_tlt,
        tlt=tlt_data,
        tmt=tmt_data,
        tls=tls_data,
    )
    print(f"Written file: {outfile}")


if __name__ == "__main__":
    main()
