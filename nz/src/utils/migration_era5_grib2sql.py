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
files.sort()
total_lines = 0

for filename in tqdm(files):
    try:

        datasets = cfgrib.open_datasets(
            os.path.join(path_era5, filename),
            backend_kwargs={"indexpath": ""}
        )

        df_inst = datasets[1].to_dataframe().reset_index()#[['time', 'latitude', 'longitude', 'msl', 'tcc', 'u10', 'v10', 't2m', 'd2m']]
        df_accum = datasets[0].to_dataframe().reset_index()#[['time', 'latitude', 'longitude', 'tp', 'cp', 'ssrd']]
        cols_i = ['time', 'latitude', 'longitude', 'msl', 'tcc', 'u10', 'v10', 't2m', 'd2m']
        cols_a = ['time', 'latitude', 'longitude', 'tp', 'cp', 'ssrd']

        df_inst = df_inst[[c for c in cols_i if c in df_inst.columns]]
        df_accum = df_accum[[c for c in cols_a if c in df_accum.columns]]

        df_final = pd.merge(
            df_inst,
            df_accum,
            on=['time', 'latitude', 'longitude'],
            how='outer'
        )

        df_final['time'] = pd.to_datetime(df_final['time'])
        df_final['time'] = df_final['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_final['latitude'] = df_final['latitude'].round(2)
        df_final['longitude'] = df_final['longitude'].round(2)

        df_final = df_final.drop_duplicates(subset=['time', 'latitude', 'longitude'])

        df_final.to_sql('weather_data', conn, if_exists='append', index=False)

        print(f"{filename} just been processed : {len(df_final)} total lines !")
        total_lines += len(df_final)

    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("Processing index...")
conn.execute('CREATE INDEX idx_coords ON weather_data (latitude, longitude, time)')
conn.close()
print(" end !")