import multiprocessing
import pandas as pd
import time
from typing import List, Tuple, Callable
import nz.src.data_processors.test_db_struct as DBStruct
from pathlib import Path

from nz.src.data_processors.utils_pipeline import mount_worker, mount_consumer, progress_monitor


import os


def parallel_orchestrator(db_path: str, db_path_era5: str, tasks: List[Tuple], chunksize: int, n_producers: int, n_consumers: int, process_func: Callable, queue_maxsize: int, total_tasks : int):

    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()

    progress_dict = manager.dict()
    progress_dict['total_chunks_produced'] = 0
    progress_dict['total_chunks_processed'] = 0

    data_queue = ctx.Queue(maxsize=queue_maxsize)
    stop_event = ctx.Event()
    k, m = divmod(len(tasks), n_producers)
    producer_tasks = list(tasks[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_producers))

    print(f"Starting: {n_producers} producteurs, {n_consumers} consommateurs")

    consumers = []
    for i in range(n_consumers):
        p = ctx.Process(target=mount_consumer, args=(i, data_queue, stop_event, process_func, progress_dict))
        p.start()
        consumers.append(p)

    producers = []
    for i in range(n_producers):
        p = ctx.Process(target=mount_worker, args=(i, producer_tasks[i], db_path, db_path_era5,chunksize, data_queue, stop_event, progress_dict))
        p.start()
        producers.append(p)

    monitor_thread = ctx.Process(target=progress_monitor, args=(progress_dict, stop_event, n_producers, n_consumers, total_tasks))
    monitor_thread.start()

    try:

        for p in producers:
            p.join()
        print(">>> Producteurs terminés.")

        print("Arrêt des consommateurs...")
        for _ in range(n_consumers):
            data_queue.put(None)

        for c in consumers:
            c.join()
        print(">>> Consommateurs terminés.")

    except KeyboardInterrupt:
        print("!!! KeyboardInterrupt !!!")
        stop_event.set()
        for p in producers + consumers:
            p.terminate()
            p.join()

    finally:
        stop_event.set()
        monitor_thread.join()
        manager.shutdown()

def run(db_paths : dict, orchestrator_params : dict) -> None:

    print("Initialisation preprocessing...")

    print("Launch from : ", os.listdir())

    #db_file = Path(db_paths['nzdb'])
    #db_era5_folder = Path(db_paths['era5db'])

    test_result = DBStruct.test_dbs_struct()
    if test_result:
        print("Error while checking database structure : Structure does'nt math or db does'nt exist at path.")
        sys.exit(-1)


    db_instance = DBStruct.NZStruct(db_paths['nzdb'])
    metadata = db_instance.getMetadata()

    unique_regions = metadata['flow_region']['region'].unique().tolist()

    tasks = []
    for region in unique_regions:
        stations_df = db_instance.getSiterefFromRegion(region)
        if not stations_df.empty:
            tasks.append((region, stations_df))

    print(f"Total tasks: {len(tasks)}")

    parallel_orchestrator(
        db_path=Path(db_paths['nzdb']),
        db_path_era5=Path(db_paths['era5db']),
        tasks=tasks,
        chunksize=int(str(orchestrator_params['chunk_size']).replace("_", "")), # Sécurité si str
        n_producers=int(orchestrator_params['n_producers']),
        n_consumers=int(orchestrator_params['n_consumers']),
        process_func=orchestrator_params['process_func'],
        queue_maxsize=int(orchestrator_params['queue_maxsize']), # Conversion ici
        total_tasks=len(tasks)
    )

if __name__ == "__main__":
    run()
