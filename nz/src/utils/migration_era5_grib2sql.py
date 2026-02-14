import os
import cfgrib
import sqlite3
import pandas as pd
from tqdm import tqdm

path_era5 = "../../data/raw/era5_downloads/"
path_db_era5 = "../../data/raw/era5_downloads/era5_data.db"
conn = sqlite3.connect(path_db_era5)

conn.execute("DROP TABLE IF EXISTS weather_data")
conn.commit()

print("Processing era5...")

files = [f for f in os.listdir(path_era5) if f.endswith('.grib')]
total_lines = 0

for filename in tqdm(files):
    try:

        parts = filename.split("_")
        y = int(parts[-2])
        m = int(parts[-1].split('.')[0])

        datasets = cfgrib.open_datasets(
            os.path.join(path_era5, filename),
            backend_kwargs={"indexpath": ""}
        )

        df_inst = datasets[1].to_dataframe().reset_index()[['time', 'latitude', 'longitude', 'msl', 'tcc', 'u10', 'v10', 't2m', 'd2m']]
        df_accum = datasets[0].to_dataframe().reset_index()[['time', 'latitude', 'longitude', 'tp', 'cp', 'ssrd']]
        df_final = pd.merge(
            df_inst,
            df_accum,
            on=['time', 'latitude', 'longitude'],
            how='inner'
        )

        df_final['time'] = pd.to_datetime(df_final['time'])

        df_final = df_final[
            (df_final['time'].dt.year == y) &
            (df_final['time'].dt.month == m)
            ]

        df_final['time'] = df_final['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df_final.to_sql('weather_data', conn, if_exists='append', index=False)

        print(f"{filename} just been processed : {len(df_final)} total lines !")
        total_lines += len(df_final)

    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("Processing index...")
conn.execute('CREATE INDEX idx_coords ON weather_data (latitude, longitude, time)')
conn.close()
print(" end !")