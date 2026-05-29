"""TCWV vs SST, 1988–2024: RSS TCWV × SST excluding ERA5/MERRA2 and USTC/cdr_.
Single TCWV (RSS) uses two panel rows to avoid an overly wide single row."""
import os
from supply_plot import find_data_files, draw_tcwv_sst_figure, exclude_era5_merra2


def _basename_lower(path):
    return os.path.basename(path).lower()


def filter_rss_cwv(paths):
    """RSS TCWV (e.g. rss_cwv_*.txt)"""
    out = []
    for p in paths:
        b = _basename_lower(p)
        if "rss_" in b and "cwv" in b:
            out.append(p)
    return sorted(set(out))


def exclude_ustc_sst(paths):
    """Exclude USTC/CDR SST (cdr_*.txt or ustc_*.txt)"""
    out = []
    for p in paths:
        b = _basename_lower(p)
        if "cdr_" in b or "ustc_" in b:
            continue
        out.append(p)
    return sorted(set(out))


def main():
    data_dir = "../data"
    cwv_all, sst_all = find_data_files(data_dir)
    cwv_files = filter_rss_cwv(cwv_all)
    sst_files = exclude_ustc_sst(exclude_era5_merra2(sst_all))

    print(f"RSS CWV files ({len(cwv_files)}):", [os.path.basename(f) for f in cwv_files])
    print(f"SST files (no ERA5/MERRA2, no USTC/CDR SST)（{len(sst_files)}）:",
          [os.path.basename(f) for f in sst_files])

    out_path = os.path.join("../plot", "fig_s8.png")
    draw_tcwv_sst_figure(
        cwv_files, sst_files, out_path,
        common_start=198801,
        common_end=202412,
        single_cwv_two_rows=True,
    )


if __name__ == "__main__":
    main()
