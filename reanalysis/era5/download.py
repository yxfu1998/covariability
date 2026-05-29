import cdsapi

for year in range(1960, 2025):
    for month in range(1,13):
        dataset = "reanalysis-era5-single-levels-monthly-means"
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "sea_surface_temperature",
                "total_column_water_vapour"
            ],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        filename = f"ERA5_2d_{year:04d}{month:02d}.nc"
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(filename)


for year in range(1960, 2025):
    for month in range(1,13):
        dataset = "reanalysis-era5-pressure-levels-monthly-means"
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "temperature"
            ],
            "pressure_level": [
                "1", "2", "3",
                "5", "7", "10",
                "20", "30", "50",
                "70", "100", "125",
                "150", "175", "200",
                "225", "250", "300",
                "350", "400", "450",
                "500", "550", "600",
                "650", "700", "750",
                "775", "800", "825",
                "850", "875", "900",
                "925", "950", "975",
                "1000"
            ],
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        filename = f"ERA5_3d_{year:04d}{month:02d}.nc"
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(filename)

