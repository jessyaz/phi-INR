import sqlite3
import pandas as pd
import time
from queue import Empty
from tqdm import tqdm
import json

from multiprocessing import Process, Queue, Event, Manager


from scipy.spatial import cKDTree



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

        print('connexion bdd set : traffic, era5 :', chk_conn(conn), chk_conn(conn_era5))

        with open('./nz/data/raw/era5_downloads/weather_grid.json', "r", encoding="utf-8") as f: # TEMP
            data = json.load(f)
            weather_grid = pd.DataFrame(data)

            grid_coords = weather_grid[['longitude', 'latitude']].values
            tree = cKDTree(grid_coords)


        try:

            for year, region in tasks:

                if stop_event.is_set():
                    break
                try:

                    siteref_card = conn.cursor().execute(
                        "SELECT DISTINCT SITEREF, LON, LAT FROM flow_meta WHERE REGION = ?", (region,)
                    ).fetchall()

                    if not siteref_card:
                        print(f"No siteref to follow... ({region} , {year})")
                        continue

                    siteref_card_df = pd.DataFrame(siteref_card, columns=['SITEREF', 'LON', 'LAT'])
                    siteref_list = siteref_card_df['SITEREF'].tolist()

                    placeholders = ','.join(['?'] * len(siteref_list))
                    query = f"SELECT * FROM flow WHERE SITEREF IN ({placeholders}) AND strftime('%Y', DATETIME) = ?"
                    params = siteref_list + [str(year)]

                    chunk_idx = 0
                    for chunk in pd.read_sql(query, conn, params=params, chunksize=chunksize):
                        if stop_event.is_set(): break

                        chunk = chunk.merge(siteref_card_df, on='SITEREF', how='left')

                        chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME'])
                        dt_nz = (
                            chunk['DATETIME']
                            .dt.tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='shift_forward')
                            .dt.floor('h')
                        )
                        chunk['DATETIME_NZ'] = dt_nz
                        chunk['time_in_utc'] = dt_nz.dt.tz_convert('UTC').dt.tz_localize(None)


                        points_trafic = chunk[['LON', 'LAT']].values
                        _, indices = tree.query(points_trafic)

                        chunk['era5_lon'] = grid_coords[indices, 0].round(2)
                        chunk['era5_lat'] = grid_coords[indices, 1].round(2)


                        u_lons = chunk['era5_lon'].unique().tolist()
                        u_lats = chunk['era5_lat'].unique().tolist()
                        #u_times = chunk['time_in_utc'].dt.strftime('%Y-%m-%d %H:%M:%S').unique().tolist()

                        valid_times = chunk['time_in_utc'].dropna()
                        u_times = valid_times.dt.strftime('%Y-%m-%d %H:%M:%S').unique().tolist()

                        ### Get Weather Context

                        query_weather = f"""
                        SELECT * FROM weather_data 
                        WHERE time IN ({','.join(['?']*len(u_times))})
                        AND longitude IN ({','.join(['?']*len(u_lons))})
                        AND latitude IN ({','.join(['?']*len(u_lats))})
                        """

                       # params = u_times + [lon_min, lon_max, lat_min, lat_max]
                        params = list(u_times) + list(u_lons) + list(u_lats)#u_lons + u_lats  +u_times

                        weather_df = pd.read_sql(query_weather, conn_era5, params=params)


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

                            if chunk.isna().sum().sum() > 1000:
                                print("AAAAAAAAAAAA")


                            chunk = chunk.drop(columns=[c for c in ['longitude', 'latitude', 'time', 'time_nz_flow'] if c in chunk.columns])


                           # print(chunk.columns)
                        else:
                            print(f"No weather data... ({region} , {year})")

                        output_queue.put({
                            'worker_id': worker_id,
                            'year': year,
                            'region': region,
                            'chunk_idx': chunk_idx,
                            'data': chunk,
                        })

                        progress_dict['total_chunks_produced'] += 1
                        progress_dict[f'producer_{worker_id}_chunks'] = progress_dict.get(f'producer_{worker_id}_chunks', 0) + 1
                        chunk_idx += 1

                except Exception as e:
                    print(f"[Producer {worker_id}] Error {year}-{region}: {e}")

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