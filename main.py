import sys
from typing import Any

#import runpy

import nz.run_processing_db as run_processing

def run():
    print("Starting ...")

    run_name="__main__"

    db_paths = {
        'raw_folder' : './nz/data/raw',
        'processed_folder' : './nz/data/processed',
        'era5' : '/era5_downloads/',
        'nztraffic' : '/nz_downloads/',
        'era5db':'./nz/data/raw/era5_downloads/era5_data.db',
        'nzdb':'./nz/data/raw/nz_downloads/NZDB.db',
    }

    def maybe(x:Any) -> Any:
        print("p call test")
        return x#

    orchestrator_params = {
        'chunk_size': 1000,
        'n_producers':4,
        'n_consumers':2,
        'process_func': None, # Callable
        'queue_maxsize': 10
    }
    run_processing.run(db_paths, orchestrator_params)

if __name__ == "__main__":

    print("===================================")

    run()

    print("===================================")



