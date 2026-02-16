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
                    query = f"SELECT * FROM flow WHERE SITEREF IN ({placeholders}) AND strftime('%M', DATETIME) = ?"
                    params = siteref_list + [str(year)]

            #        # Get another data in nz traffic - DO this in preprocessing to get scheme .
            #        df_holiday = pd.read_sql(
            #            "SELECT * FROM extreme_weather WHERE REGION = ? AND strftime('%M', DATETIME) = ?",
            #            conn,
            #            params=params
            #        )
            #        df_extreme_weather = pd.read_sql(
            #            "SELECT * FROM holiday WHERE REGION = ? AND strftime('%Y', DATETIME) = ?",
            #            conn,
            #            params=params
            #        )

                    chunk_idx = 0
                    for chunk in pd.read_sql(query, conn, params=params, chunksize=chunksize):
                        if stop_event.is_set(): break

                        chunk = chunk.merge(siteref_card_df, on='SITEREF', how='left')

                        #((df['lon'] % 360) * 4).round() / 4
                        #(df['lat'] * 4).round() / 4
                   #    lon, lat = ((chunk['LON']% 360) * 4).round() / 4, (chunk['LAT'] * 4).round() / 4
                   #     weather = conn_era5.cursor().execute(
                   #         f"SELECT * FROM weather_data WHERE longitude = ({lon}) AND latitude = ({lat}) ;",
                   #         (lon,lat)
                   #     )

                        chunk['grid_lon'] = ((chunk['LON'] % 360) * 4).round() / 4
                        chunk['grid_lat'] = (chunk['LAT'] * 4).round() / 4

                        unique_long_lat = chunk[['grid_lon', 'grid_lat']].drop_duplicates()
                        weather_chunks = []
                        for _, row in unique_long_lat.iterrows():
                            weather_data = pd.read_sql(
                                "SELECT * FROM weather_data WHERE longitude = ? AND latitude = ?",
                                conn_era5,
                                params=(row['grid_lon'], row['grid_lat'])
                            )
                            weather_chunks.append(weather_data)

                        if weather_chunks: # Pour le test local -- suppr
                            weather_df = pd.concat(weather_chunks).drop_duplicates()
                            chunk = chunk.merge(weather_df,
                                left_on=['grid_lon', 'grid_lat'],
                                right_on=['longitude', 'latitude'],
                                how='left'
                            )
                        else:
                            print(f"No weather data... ({region} , {year})")

                        print(chunk.columns)
                        print(chunk.head())

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