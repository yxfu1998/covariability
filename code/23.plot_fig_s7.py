"""TCWV vs SST, 2002–2024: all datasets excluding ERA5/MERRA2 (CWV×SST pairs from find_data_files)."""
import os
from supply_plot import find_data_files, draw_tcwv_sst_figure, exclude_era5_merra2


def main():
    data_dir = "../data"
    cwv_files, sst_files = find_data_files(data_dir)
    cwv_files = exclude_era5_merra2(cwv_files)
    sst_files = exclude_era5_merra2(sst_files)

    print(f"CWV files excluding ERA5/MERRA2 ({len(cwv_files)}):", [os.path.basename(f) for f in cwv_files])
    print(f"SST files excluding ERA5/MERRA2 ({len(sst_files)}):", [os.path.basename(f) for f in sst_files])

    out_path = os.path.join("../plot", "fig_s7.png")
    draw_tcwv_sst_figure(
        cwv_files, sst_files, out_path,
        common_start=200201,
        common_end=202412,
        single_cwv_two_rows=False,
    )


if __name__ == "__main__":
    main()
