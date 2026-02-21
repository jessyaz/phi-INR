import sqlite3
import pandas as pd
import time
from queue import Empty
from tqdm import tqdm
import json

from multiprocessing import Process, Queue, Event, Manager


from scipy.spatial import cKDTree

import numpy as np


import pytz



def chk_conn(conn):
    try:
        conn.cursor()
        return True
    except Exception as ex:
        return False


def mount_worker(worker_id, tasks, db_path, db_path_era5, chunksize, output_queue, stop_event, progress_dict):
    if isinstance(chunksize, str):
        chunksize = int(chunksize.replace("_", ""))

    with sqlite3.connect(db_path) as conn, sqlite3.connect(db_path_era5) as conn_era5:

        with open('./nz/data/raw/era5_downloads/weather_grid.json', "r", encoding="utf-8") as f:
            weather_grid = pd.DataFrame(json.load(f))
            grid_coords = weather_grid[['longitude', 'latitude']].values
            tree = cKDTree(grid_coords)

        try:
            for region, stations_df in tasks:
                if stop_event.is_set():
                    break

                try:
                    u_siteref_list = stations_df['SITEREF'].unique().tolist()


                    for siteref in u_siteref_list:
                        print(siteref)

                        #placeholders = ','.join(['?'] * len(u_siteref_list))
                        query = f"SELECT * FROM flow WHERE SITEREF = ? ORDER BY DATETIME"

                        chunk_idx = 0

                        #for chunk in pd.read_sql(query, conn, params=[siteref]):
                        if True:
                            chunk = pd.read_sql(query, conn, params=[siteref])

                            if stop_event.is_set():
                                break


                            chunk = chunk.sort_values('DATETIME')

                            chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME'])



                            chunk = chunk[chunk['DATETIME'] >= '2013-01-02']
                            if chunk.empty:
                                continue

                            chunk = chunk.merge(stations_df[['SITEREF', 'LON', 'LAT']], on='SITEREF', how='left')


                            chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME']).dt.floor('h')



                            try:
                                chunk['DATETIME_NZ'] = (
                                    chunk['DATETIME']
                                    .dt.tz_localize('Pacific/Auckland', ambiguous=False, nonexistent='shift_forward')
                                )
                            except Exception as e:
                                print(e)
                                print(chunk['DATETIME'])

                           # print(len(chunk['DATETIME_NZ']))

                            #chunk = chunk.sort_values(['WEIGHT', 'DIRECTION', 'DATETIME'])


                            #chunk.groupby(['DATETIME','WEIGHT','DIRECTION'])
                            chunk['time_in_utc'] = chunk['DATETIME_NZ'].dt.tz_convert('UTC').dt.tz_localize(None).values

                            points_trafic = chunk[['LON', 'LAT']].values
                            _, indices = tree.query(points_trafic)

                            chunk['era5_lon'] = grid_coords[indices, 0].round(2)
                            chunk['era5_lat'] = grid_coords[indices, 1].round(2)

                            u_lons = chunk['era5_lon'].unique().tolist()
                            u_lats = chunk['era5_lat'].unique().tolist()
                            u_times = chunk['time_in_utc'].dt.strftime('%Y-%m-%d %H:%M:%S').unique().tolist()

                            query_weather = f"""
                                SELECT * FROM weather_data 
                                WHERE time IN ({','.join(['?']*len(u_times))})
                                AND longitude IN ({','.join(['?']*len(u_lons))})
                                AND latitude IN ({','.join(['?']*len(u_lats))})
                            """

                            weather_params = u_times + u_lons + u_lats
                            weather_df = pd.read_sql(query_weather, conn_era5, params=weather_params)



                            if not weather_df.empty:
                                weather_df['time'] = pd.to_datetime(weather_df['time'])
                                weather_df['longitude'] = weather_df['longitude'].astype(float).round(2)
                                weather_df['latitude'] = weather_df['latitude'].astype(float).round(2)

                                chunk = chunk.merge(
                                    weather_df,
                                    left_on=['era5_lon', 'era5_lat', 'time_in_utc'],
                                    right_on=['longitude', 'latitude', 'time'],
                                    how='left'
                                )


                                nan_report = chunk.isna().sum()
                                if nan_report.any():
                                    print(f"\n{'='*80}")
                                    print(f"NaNs in {region} (Chunk {chunk_idx}): {nan_report[nan_report > 0].to_dict()}")
                                    with pd.option_context('display.max_columns', None, 'display.width', 1000):
                                        print("\n[A] Lignes avec NaNs (Trafic) :")
                                        print(chunk[chunk.isna().any(axis=1)].head(5))
                                        print("\n[B] Echantillon Météo disponible :")
                                        print(weather_df.head(5))
                                    print(f"{'='*80}\n")

                                drop_cols = [c for c in ['longitude', 'latitude', 'time', 'time_nz_flow'] if c in chunk.columns]
                                chunk = chunk.drop(columns=drop_cols)
                            else:
                                print(f"[Worker {worker_id}] No weather data for chunk {chunk_idx} in {region}")

                            output_queue.put({
                                'worker_id': worker_id,
                                'region': region,
                                'chunk_idx': chunk_idx,
                                'data': chunk,
                            })

                            progress_dict['total_chunks_produced'] += 1
                            progress_dict[f'producer_{worker_id}_chunks'] = progress_dict.get(f'producer_{worker_id}_chunks', 0) + 1
                            chunk_idx += 1

                except Exception as e:
                    print(f"[Producer {worker_id}] Error in region {region}: {e}")

                progress_dict['total_tasks_done'] = progress_dict.get('total_tasks_done', 0) + 1

        except Exception as e:
            print(f"[Producer {worker_id}] Critical Error: {e}")

def mount_consumer(consumer_id, input_queue, stop_event, process_func, progress_dict):

    cols_to_keep = [
        'SITEREF', 'DATETIME', 'FLOW', 'WEIGHT', 'DIRECTION',
        'LON', 'LAT', 'DATETIME_NZ',
        'era5_lon', 'era5_lat',
        'msl', 'tcc', 'u10', 'v10', 't2m', 'd2m', 'tp', 'cp', 'ssrd'
    ]

    processed_count = 0

    with sqlite3.connect("./nz/data/processed/db.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL;")

        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=1)

                if item is None:
                    break

                chunk = item['data']

                chunk = chunk[chunk.columns.intersection(cols_to_keep)]

               # if process_func:
                #    process_func(chunk)

                chunk.to_sql(
                    'data',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=500
                )

                processed_count += 1
                progress_dict['total_chunks_processed'] += 1
                progress_dict[f'consumer_{consumer_id}_chunks'] = processed_count

            except Empty:
                continue
            except Exception as e:
                print(f"[Consumer {consumer_id}] Error: {e}")

def progress_monitor(progress_dict, stop_event, n_producers, n_consumers, total_tasks):

    pbar = tqdm(total=total_tasks, desc="GLOBAL PROGRESS", dynamic_ncols=True)

    while not stop_event.is_set():
        tasks_done = progress_dict.get('total_tasks_done', 0)
        produced = progress_dict.get('total_chunks_produced', 0)
        processed = progress_dict.get('total_chunks_processed', 0)

        pbar.n = tasks_done

        pbar.set_description(f"Tasks: {tasks_done}/{total_tasks} | Stock: {produced-processed} | Total Chunks: {processed}")

        pbar.refresh()
        time.sleep(0.5)

    pbar.close()