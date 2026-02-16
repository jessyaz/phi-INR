import sqlite3
import pandas as pd
import time
from queue import Empty
from tqdm import tqdm

from multiprocessing import Process, Queue, Event, Manager




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
                    #siteref_card = [row[0] for row in siteref_card]

                    siteref_list = siteref_card_df['SITEREF'].tolist()

                    placeholders = ','.join(['?'] * len(siteref_list))
                    query = f"SELECT * FROM flow WHERE SITEREF IN ({placeholders}) AND strftime('%Y', DATETIME) = ?"
                    params = siteref_list + [str(year)]

                    df_holiday = pd.read_sql("SELECT * FROM holiday WHERE (REGION == 'All' OR REGION LIKE ?)", conn, params=(f'%{region.split(' - ')[-1].strip()}%',))
                    df_holiday['START_DATE'] = pd.to_datetime(df_holiday['START_DATE']).dt.tz_localize('Pacific/Auckland', ambiguous=True).dt.tz_convert('UTC').dt.tz_localize(None)
                    df_holiday['STOP_DATE'] = pd.to_datetime(df_holiday['STOP_DATE']).dt.tz_localize('Pacific/Auckland', ambiguous=True).dt.tz_convert('UTC').dt.tz_localize(None)

                    chunk_idx = 0
                    for chunk in pd.read_sql(query, conn, params=params, chunksize=chunksize):
                        if stop_event.is_set(): break

                        h_cols = [c for c in df_holiday.columns if c not in ['REGION', 'START_DATE', 'STOP_DATE']]

                        chunk = chunk.merge(siteref_card_df, on='SITEREF', how='left')

                        chunk['time_nz_flow'] = (
                            pd.to_datetime(chunk['DATETIME'])
                            .dt.tz_localize('Pacific/Auckland', ambiguous=True, nonexistent='shift_forward') # Gere le passage a lhiver ?
                            .dt.tz_convert('UTC')
                            .dt.tz_localize(None)
                            .dt.floor('h')
                        )

                        for col in h_cols: chunk[col] = None
                        if not df_holiday.empty:
                            for _, row in df_holiday.iterrows():
                                mask = (chunk['time_nz_flow'] >= row['START_DATE']) & (chunk['time_nz_flow'] <= row['STOP_DATE'])
                                if mask.any():
                                    for col in h_cols: chunk.loc[mask, col] = row[col]

                      #  chunk['grid_lon'] = (((chunk['LON'] + 180) % 360 - 180) * 4).round() / 4
                      #  chunk['grid_lat'] = (chunk['LAT'] * 4).round() / 4

                        chunk['grid_lon'] = (((chunk['LON'] + 180) % 360 - 180) * 4).round() / 4
                        chunk['grid_lat'] = (chunk['LAT'] * 4).round() / 4

                        u_lons = chunk['grid_lon'].round(2).unique().tolist()
                        u_lats = chunk['grid_lat'].round(2).unique().tolist()
                        u_times = chunk['time_nz_flow'].dt.strftime('%Y-%m-%d %H:%M:%S').unique().tolist()

                        #test
                        lon_min, lon_max = chunk['grid_lon'].min() - 0.5, chunk['grid_lon'].max() + 0.5
                        lat_min, lat_max = chunk['grid_lat'].min() - 0.5, chunk['grid_lat'].max() + 0.5


                        if (len(u_lons) + len(u_lats)) > 500:
                            print(f"Trop de stations ({len(u_lons) + len(u_lats)}).")


                      #  query_weather = f"""
                      #      SELECT * FROM weather_data
                      #      WHERE longitude IN ({','.join(['?']*len(u_lons))})
                      #      AND longitude BETWEEN ? AND ?
                      #      AND latitude BETWEEN ? AND ? """
                       #     AND latitude IN ({','.join(['?']*len(u_lats))})
                         #   AND time IN ({','.join(['?']*len(u_times))})
                       # """

                        query_weather = f"""
                        SELECT * FROM weather_data 
                        WHERE time IN ({','.join(['?']*len(u_times))})
                        AND longitude BETWEEN ? AND ?
                        AND latitude BETWEEN ? AND ?
                        """

                        params = u_times + [lon_min, lon_max, lat_min, lat_max]
                        #params = u_lons + u_lats + u_times

                        weather_df = pd.read_sql(query_weather, conn_era5, params=params)
                        weather_df['time'] = pd.to_datetime(weather_df['time'])

                        chunk['time_nz_flow'] = pd.to_datetime(chunk['time_nz_flow']).dt.tz_localize(None)
                        weather_df['time'] = pd.to_datetime(weather_df['time']).dt.tz_localize(None)

                        chunk['time_nz_flow'] = chunk['time_nz_flow'].dt.floor('h')
                        weather_df['time'] = weather_df['time'].dt.floor('h')

                        chunk['grid_lon'] = chunk['grid_lon'].astype(float).round(2)
                        chunk['grid_lat'] = chunk['grid_lat'].astype(float).round(2)
                        weather_df['longitude'] = weather_df['longitude'].astype(float).round(2)
                        weather_df['latitude'] = weather_df['latitude'].astype(float).round(2)

                        ###


                        ###

                        if not weather_df.empty:
                            chunk = chunk.merge(
                                weather_df,
                                left_on=['grid_lon', 'grid_lat', 'time_nz_flow'],
                                right_on=['longitude', 'latitude', 'time'],
                                how='left'
                            )#.drop(columns=['longitude', 'latitude', 'time'])

                            mask_nan = chunk.isna().any(axis=1)

                            # On utilise .any() pour dire : "Si au moins UNE ligne est vraie"
                            if mask_nan.any():
                                cols_debug = [c for c in ['grid_lon', 'grid_lat', 'longitude', 'latitude', 'time', 'time_nz_flow'] if c in chunk.columns]
                                print(f"\n⚠️ [DEBUG] {mask_nan.sum()} lignes avec des NaN détectées :")
                                print(chunk.loc[mask_nan, cols_debug].head(10))



                            weather_cols = [c for c in weather_df.columns if c not in ['longitude', 'latitude', 'time']]
                            if weather_cols:

                                pct_nan = chunk[weather_cols[0]].isna().mean() * 100
                                print(f"📊 [DEBUG Merge] Taux de NaN brut : {pct_nan:.2f}%")

                           # if weather_cols:
                           #     chunk[weather_cols] = chunk.groupby('time_nz_flow')[weather_cols].ffill().bfill()

                            chunk = chunk.drop(columns=[c for c in ['longitude', 'latitude', 'time', 'time_nz_flow'] if c in chunk.columns])

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
    processed_count = 0
    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=1)

            if item is None:
                break

            chunk = item['data']

            if process_func:
                process_func(chunk)

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