import os
import glob
import re

def extract_model_ensemble_from_filename(filename):
    """
    Extract model and ensemble names from filename
    Filename format: {variable}_Amon_{model}_{experiment}_{ensemble}_g[nr]_{date_range}.nc
    Example: prw_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc
          prw_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_185001-201412.nc
    """
    basename = os.path.basename(filename)
    # Use regex to extract model and ensemble; supports gn and gr grid types
    pattern = r'([^_]+)_Amon_([^_]+)_([^_]+)_([^_]+)_g[^_]+_'
    match = re.match(pattern, basename)
    if match:
        variable = match.group(1)
        model = match.group(2)
        experiment = match.group(3)
        ensemble = match.group(4)
        return model, ensemble, experiment, variable
    return None, None, None, None

def find_valid_combinations(base_dir):
    """
    Traverse directories to find model-ensemble combinations with prw, ts, ta and both historical and ssp585 experiments
    Returns: (valid_combinations, missing_info)
    """
    base_path = base_dir
    exclude_dirs = {'tt_star', 'extended_hist'}
    
    valid_combinations = set()
    missing_info = []  # list to store missing-information records
    
    required_vars = {'prw', 'ts', 'ta'}
    required_experiments = {'historical', 'ssp585'}
    
    # Traverse all subdirectories
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        # Skip non-directories and excluded directories
        if not os.path.isdir(item_path) or item in exclude_dirs:
            continue
        
        print(f"Processing directory: {item}")
        
        # Store file information under this directory
        file_info = {}  # {(model, ensemble, experiment): {variable: [files]}}
        
        # Find all .nc files
        nc_files = glob.glob(os.path.join(item_path, '*.nc'))
        
        for nc_file in nc_files:
            model, ensemble, experiment, variable = extract_model_ensemble_from_filename(nc_file)
            
            if model and ensemble and experiment and variable:
                key = (model, ensemble, experiment)
                if key not in file_info:
                    file_info[key] = {}
                if variable not in file_info[key]:
                    file_info[key][variable] = []
                file_info[key][variable].append(nc_file)
        
        # Group by model-ensemble combination
        model_ensemble_combinations = {}
        for (model, ensemble, experiment), variables in file_info.items():
            key = (model, ensemble)
            if key not in model_ensemble_combinations:
                model_ensemble_combinations[key] = {}
            model_ensemble_combinations[key][experiment] = variables
        
        # Check whether each model-ensemble combination meets requirements
        # Requires prw, ts, ta variables and both historical and ssp585 experiments
        for (model, ensemble), experiments in model_ensemble_combinations.items():
            combination = f"{model}_{ensemble}"
            missing_details = []
            
            # Check for historical and ssp585 experiments
            missing_experiments = required_experiments - set(experiments.keys())
            if missing_experiments:
                missing_details.append(f"missing experiment(s): {', '.join(sorted(missing_experiments))}")
            
            # Check variables for each experiment
            for exp in required_experiments:
                if exp in experiments:
                    exp_vars = set(experiments[exp].keys())
                    missing_vars = required_vars - exp_vars
                    if missing_vars:
                        missing_details.append(f"{exp}missing variable(s): {', '.join(sorted(missing_vars))}")
                else:
                    missing_details.append(f"{exp}missing variable(s): {', '.join(sorted(required_vars))}")
            
            # If fully satisfied, add to valid combinations
            if not missing_details:
                valid_combinations.add(combination)
                print(f"  Found valid combination: {combination}")
            else:
                # Record missing information
                missing_info.append({
                    'combination': combination,
                    'details': missing_details
                })
                print(f"  Missing data for {combination}: {'; '.join(missing_details)}")
    
    return sorted(valid_combinations), missing_info

def main():
    base_dir = '../cmip6'
    output_file = '../cmip6/cmip_name.txt'
    missing_file = '../cmip6/cmip_missing.txt'
    
    print(f"Scanning directory: {base_dir}")
    print("Excluding directories: tt_star, extended_hist")
    print("Looking for combinations with: prw, ts, ta variables and historical, ssp585 experiments")
    print("-" * 60)
    
    valid_combinations, missing_info = find_valid_combinations(base_dir)
    
    print("-" * 60)
    print(f"Found {len(valid_combinations)} valid combinations")
    print(f"Found {len(missing_info)} combinations with missing data")
    
    # Write valid combinations file
    with open(output_file, 'w') as f:
        for combination in valid_combinations:
            f.write(combination + '\n')
    
    print(f"Results written to {output_file}")
    
    # Write missing-information file
    with open(missing_file, 'w') as f:
        for item in missing_info:
            f.write(f"{item['combination']}: {'; '.join(item['details'])}\n")
    
    print(f"Missing information written to {missing_file}")

if __name__ == '__main__':
    main()

