"""TMT vs SST, 1988–2024: y-axis all TMT excluding ERA5/MERRA2; x-axis SST excluding ERA5/MERRA2/USTC."""
import os
from supply_plot import (
    find_files_with_variable,
    draw_obs_pair_figure,
    exclude_era5_merra2,
    exclude_cdr_files,
)


def exclude_ustc_sst(paths):
    """Exclude USTC SST (ustc_*.txt; cdr_ via exclude_cdr_files)"""
    out = []
    for p in paths:
        if "ustc_" in os.path.basename(p).lower():
            continue
        out.append(p)
    return sorted(set(out))


def main():
    data_dir = "../data"
    y_files = exclude_era5_merra2(find_files_with_variable(data_dir, "tmt"))
    x_files = exclude_ustc_sst(
        exclude_cdr_files(
            exclude_era5_merra2(find_files_with_variable(data_dir, "sst"))
        )
    )

    print(f"y-axis files (excluding ERA5/MERRA2)（{len(y_files)}）:", [os.path.basename(f) for f in y_files])
    print(f"x-axis files (no ERA5/MERRA2/USTC SST)（{len(x_files)}）:",
          [os.path.basename(f) for f in x_files])

    out_path = os.path.join("../plot", "fig_s10.png")
    draw_obs_pair_figure(
        y_files, x_files, out_path, "tmt", "sst",
        common_start=198101,
        common_end=202412,
        single_y_two_rows=False,
    )


if __name__ == "__main__":
    main()
