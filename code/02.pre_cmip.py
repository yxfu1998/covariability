import os
import glob
import re
from netCDF4 import Dataset, num2date
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

def write_nc(filename,variname,data,time,lat,lon):
    n1 = len(time)
    n2 = len(lat)
    n3 = len(lon)

    nc_file = Dataset(filename, 'w', format='NETCDF4')

    nc_file.createDimension('time', n1)
    nc_file.createDimension('lat', n2)
    nc_file.createDimension('lon', n3)

    times = nc_file.createVariable('time', np.float32, ('time',))
    lats = nc_file.createVariable('lat', np.float32, ('lat',))
    lons = nc_file.createVariable('lon', np.float32, ('lon',))
    values = nc_file.createVariable(variname, np.float32, ('time', 'lat', 'lon',))

    times[:] = time
    lats[:] = lat
    lons[:] = lon

    values[:, :, :] = data

    nc_file.close()

    return

def cal_tt(vari, plev0, wt_tt):
    # Define interpolation function
    f = interp1d(wt_tt[:, 1], wt_tt[:, 6], kind='linear')
    
    # Compute interpolated values and normalize
    plev = plev0/100.
    y = f(np.array(plev))
    normalized_y = y / np.sum(y)
    
    # Mask vari, keeping only values > 0 and < 1000
    mask = (vari > 0) & (vari < 1000)
    vari_masked = np.where(mask, vari, 0)
    
    # Compute weighted mean
    data = np.tensordot(normalized_y, vari_masked, axes=(0, 0))
    return data

def list_months(start_year, start_month, end_year, end_month):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    months = []
    while start_date <= end_date:
        month_string = start_date.strftime("%Y%m")
        months.append(month_string)
        start_date = start_date + timedelta(days=32)
        start_date = start_date.replace(day=1)

    months = np.int32(np.array(months))
    return months

def calculate_date_from_start(n, start_year):
    # Initialize empty arrays for results
    years = np.empty_like(n)
    months = np.empty_like(n)
    days = np.empty_like(n)

    # Loop over elements to compute corresponding dates
    for i, n_value in enumerate(n):
        # Initial date is the specified start year
        start_date = datetime(start_year, 1, 1)

        # Compute date n days later
        target_date = start_date + timedelta(days=n_value - 1)  # subtract 1 because dates are 1-based

        # Extract year/month/day into corresponding arrays
        years[i] = target_date.year
        months[i] = target_date.month
        days[i] = target_date.day

    return years, months, days

def extract_year_from_date(date_string):
    # Split string on whitespace
    parts = date_string.split()

    # Extract third part (index 2), the date portion
    date_part = parts[2]

    # Split date portion on "-"
    date_parts = date_part.split("-")

    # Extract year portion (index 0)
    year = np.array(date_parts[0])

    return year

def extract_model_ensemble_from_filename(filename):
    """
    Extract model and ensemble names from filename
    Filename format: {variable}_Amon_{model}_{experiment}_{ensemble}_g[nr]_{date_range}.nc
    Example: prw_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc
          prw_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc
    """
    basename = os.path.basename(filename)
    # Grid token position is flexible; allow any non-underscore token: _gXXX_
    pattern = r'([^_]+)_Amon_([^_]+)_([^_]+)_([^_]+)_g[^_]+_'
    match = re.match(pattern, basename)
    if match:
        variable = match.group(1)
        model = match.group(2)
        experiment = match.group(3)
        ensemble = match.group(4)
        return model, ensemble, experiment, variable
    return None, None, None, None

def extract_date_range_from_filename(filename):
    """
    Extract date range from filename
    Filename format: {variable}_Amon_{model}_{experiment}_{ensemble}_g[nr]_{start_date}-{end_date}.nc
    Example: ts_Amon_UKESM1-0-LL_historical_r2i1p1f2_gn_195001-201412.nc
    Returns: (start_year, start_month, end_year, end_month)
    """
    basename = os.path.basename(filename)
    # Extract date-range part in YYYYMM-YYYYMM format
    pattern = r'_(\d{6})-(\d{6})\.nc$'
    match = re.search(pattern, basename)
    if match:
        start_str = match.group(1)
        end_str = match.group(2)
        start_year = int(start_str[:4])
        start_month = int(start_str[4:6])
        end_year = int(end_str[:4])
        end_month = int(end_str[4:6])
        return start_year, start_month, end_year, end_month
    return None, None, None, None


deri = '../cmip6/'
vari_names = ['ts', 'prw', 'ta']

import pandas as pd

# Weight file path (consistent with reference program)
path = '../cmip6/' 

all_data = {}

filenames = ['ch5','ch7','ch9']

for filename in filenames:
    xls = path+filename+'.csv'
    df = pd.read_csv(xls, header=1)  # header=1 means read starting from the second row
    df = df.dropna(how='all')
    all_data[filename] = df

# Example: access a sub-table, e.g. Ch1
df_ch5 = all_data['ch5']
wt_tmt=np.array(df_ch5)
df_ch7 = all_data['ch7']
wt_tut=np.array(df_ch7)
df_ch9 = all_data['ch9']
wt_tls=np.array(df_ch9)
wt_tlt = 1.430*wt_tmt-0.462*wt_tut+0.032*wt_tls
wt_tmt = 1.1*wt_tmt-0.1*wt_tls

# Set time range to 1980-2024
standard_time = list_months(1980, 1, 2024, 12)

# Read model-ensemble list
cmip_name_file = '../cmip6/cmip_name.txt'
cmip6_data_dir = '../cmip6/'
output_dir = '../cmip6'

# Read all model-ensemble combinations
target_combinations = set()
with open(cmip_name_file, 'r') as file:
    # Read remaining lines
    for line in file:
        line = line.strip()
        if line:
            target_combinations.add(line)

print(f"Target combinations: {len(target_combinations)}")

# Traverse all subdirectories to find matching files
exclude_dirs = {'tt_star', 'extended_hist'}
file_map = {}  # {(model, ensemble, variable): [list of files]}

print(f"Scanning directory: {cmip6_data_dir}")
for item in os.listdir(cmip6_data_dir):
    item_path = os.path.join(cmip6_data_dir, item)
    
    # Skip non-directories and excluded directories
    if not os.path.isdir(item_path) or item in exclude_dirs:
        continue
    
    # Find all .nc files
    nc_files = glob.glob(os.path.join(item_path, '*.nc'))
    
    for nc_file in nc_files:
        model, ensemble, experiment, variable = extract_model_ensemble_from_filename(nc_file)
        
        if model and ensemble and experiment and variable:
            combination = f"{model}_{ensemble}"
            # Only process files in target combinations
            if combination in target_combinations and variable in vari_names:
                key = (model, ensemble, variable)
                if key not in file_map:
                    file_map[key] = []
                file_map[key].append(nc_file)

print(f"Found files for {len(file_map)} model-ensemble-variable combinations")

# Process each model-ensemble combination
for cmipname in sorted(target_combinations):
    parts = cmipname.split('_', 1)
    if len(parts) != 2:
        continue
    model = parts[0]
    ensemble = parts[1]
    
    print(f"Processing: {cmipname}")
    
    for vari_name in vari_names:
        key = (model, ensemble, vari_name)
        if key not in file_map:
            print(f"  Warning: No files found for {vari_name}")
            continue
        
        files = file_map[key]
        files.sort()  # Sort by filename
        
        print(f"  Processing {vari_name}, found {len(files)} files")
    
        if vari_name=='ta':
            array1 = [None] * len(standard_time)
            array2 = [None] * len(standard_time) 
        else:
            array = [None] * len(standard_time)
    
        numb_file = len(files)
    
        lat = None
        lon = None
        data_found = False
    
        for i in range(numb_file):
            try:
                nc_file = Dataset(files[i], 'r')

                time_var = nc_file.variables['time']
                time = time_var[:]
                if lat is None:
                    lat = nc_file.variables['lat'][:]
                if lon is None:
                    lon = nc_file.variables['lon'][:]
                vari = nc_file.variables[vari_name][:]
    
                if vari_name=='ta':
                    plev = nc_file.variables['plev'][:]
    
                ntime = len(time)
                
                # Extract date range from filename and build month sequence
                start_year, start_month, end_year, end_month = extract_date_range_from_filename(files[i])
                if start_year is None:
                    print(f"  Warning: Cannot extract date range from filename {files[i]}")
                    nc_file.close()
                    continue
                
                # Build month sequence from date range in filename
                file_months = list_months(start_year, start_month, end_year, end_month)
                
                # Check whether time lengths match
                if ntime != len(file_months):
                    print(f"  Warning: Time length mismatch in {files[i]}: data has {ntime} time steps, filename suggests {len(file_months)} months")
                    # use the shorter length
                    ntime = min(ntime, len(file_months))
    
                for itime in range(ntime):
                    # Use month information from filename
                    yyyymm = file_months[itime]
                    # Map current YYYYMM to standard time-axis index
                    match_idx = np.where(yyyymm == standard_time)[0]
                    if match_idx.size == 0:
                        continue
                    idx = match_idx[0]
                    if vari_name=='ta':
                        tlt = cal_tt(vari[itime,:,:,:],plev,wt_tlt)
                        tmt = cal_tt(vari[itime,:,:,:],plev,wt_tmt)
                        array1[idx] = tlt
                        array2[idx] = tmt
                    else:
                        data = vari[itime,:,:]
                        array[idx] = data
                    data_found = True
                    data_found = True
    
                nc_file.close()
            except Exception as e:
                print(f"  Error processing file {files[i]}: {e}")
                continue
        
        # Check whether data exist
        if not data_found:
            print(f"  Warning: No data found for {vari_name} in time range 1980-2024")
            continue
        
        # Output file
        if vari_name=='ta':
            outfile1 = os.path.join(output_dir, cmipname+'_tlt1980-2024.nc')
            outfile2 = os.path.join(output_dir, cmipname+'_tmt1980-2024.nc')
            
            # Check whether sufficient data exist
            valid_indices = [i for i in range(len(array1)) if array1[i] is not None]
            if not valid_indices:
                print(f"  Warning: No valid data for {vari_name}")
                continue
            
            # Use full standard_time as time axis, consistent with reference program
            n1 = len(standard_time)
            [n2,n3] = array1[valid_indices[0]].shape
            tlt = np.empty((n1,n2,n3))
            tmt = np.empty((n1,n2,n3))
            
            # Initialize to NaN
            tlt[:] = np.nan
            tmt[:] = np.nan
            
            # Fill valid data
            for i in valid_indices:
                tlt[i,:,:] = array1[i]
                tmt[i,:,:] = array2[i]
            
            write_nc(outfile1,'tlt',tlt,standard_time,lat,lon)
            write_nc(outfile2,'tmt',tmt,standard_time,lat,lon)
            print(f"  Written: {outfile1}, {outfile2} (valid months: {len(valid_indices)}/{n1})")
        else:
            outfile = os.path.join(output_dir, cmipname.strip()+'_'+vari_name+'1980-2024.nc')
            
            # Check whether sufficient data exist
            valid_indices = [i for i in range(len(array)) if array[i] is not None]
            if not valid_indices:
                print(f"  Warning: No valid data for {vari_name}")
                continue
            
            # Use full standard_time as time axis, consistent with reference program
            n1 = len(standard_time)
            [n2,n3] = array[valid_indices[0]].shape
            outdata = np.empty((n1,n2,n3))
            
            # Initialize to NaN
            outdata[:] = np.nan
            
            # Fill valid data
            for i in valid_indices:
                outdata[i,:,:] = array[i]
            
            write_nc(outfile,vari_name,outdata,standard_time,lat,lon)
            print(f"  Written: {outfile} (valid months: {len(valid_indices)}/{n1})")

print("Processing completed!")

