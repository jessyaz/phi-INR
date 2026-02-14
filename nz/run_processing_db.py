import multiprocessing
import pandas as pd
import time
from typing import List, Tuple, Callable
import nz.src.data_processors.test_db_struct as db_struct
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

def run(base_path='./nz/data/raw/'):
    print("Initialisation...")

    print(os.listdir())

    path_obj = Path(base_path)
    db_file = path_obj / 'nz_downloads/NZDB.db'
    db_era5_file = path_obj / 'era5_downloads/era5_data.db'

    test_result = db_struct.test()

    db_instance = db_struct.NzStruct(str(path_obj))
    metadata = db_instance.getMetadata()

    flow_datetime = pd.to_datetime(metadata['flow_datetime']['datetime'])
    flow_region = metadata['flow_region']['region']

    tasks = [
        (year, region)
        for year in flow_datetime.dt.year.unique()
        for region in flow_region.unique()
    ]

    print(f"Total Tâches: {len(tasks)}")

    parallel_orchestrator(
        db_path=str(db_file),
        db_path_era5=str(db_era5_file),
        tasks=tasks,
        chunksize="100_000",
        n_producers=8,
        n_consumers=8,
        process_func=None,
        queue_maxsize=10,
        total_tasks=len(tasks)
    )

if __name__ == "__main__":
    run()
