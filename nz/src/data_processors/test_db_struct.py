import os
import json
import sys
# import a revoir

from tqdm import tqdm
from typing import Tuple, List, Any, Callable
from multiprocessing import Process, Queue, Event, Manager
from queue import Empty
import sqlite3

import pandas as pd

from pathlib import Path
import os

def test():
    print("Test structure ...")
    return False # Temporary


class NzStruct():

    def __init__(self, path: str = '../data/raw') -> None:
        self.path = Path(path)
        json_file = self.path / 'nz_downloads/metadata.json'

        if os.path.exists(json_file):
            print("Lecture du cache..")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = {k: pd.DataFrame(v) for k, v in data.items()}

        else:

            flow_date, flow_region = self.__fetchDataBase()

            self.metadata: dict = {
                'flow_datetime': pd.DataFrame(flow_date, columns=['datetime']),
                'flow_region': pd.DataFrame(flow_region, columns=['region'])
            }

            self.path.mkdir(parents=True, exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self.metadata.items()}, f, indent=4)


    def __fetchDataBase(self) -> Tuple[list, list]:
        print("Fetching metadata in database ...")

        db_path = self.path / 'nz_downloads/NZDB.db'
        self.conn = sqlite3.connect(str(db_path))

        try:
            flow_date = self.conn.cursor().execute(
                "SELECT DISTINCT DATETIME FROM flow;"
            ).fetchall()
            flow_region = self.conn.cursor().execute(
                "SELECT DISTINCT REGION FROM flow_meta"
            ).fetchall()

            flow_date = [row[0] for row in flow_date]
            flow_region = [row[0] for row in flow_region]

        except Exception as e:
            print(f"Error while fetching db (no data?): {e}")
            flow_date, flow_region = [], []

        return flow_date, flow_region

    def getMetadata(self) -> dict:
        return self.metadata

class ERA5Struct():
    def __init__(self, path: str = './nz/data/raw') -> None:
        self.path = Path(path)
        json_file = self.path / 'weather_grid.json'
        self.temp = json_file # A refacto
        self.weather_grid = None

        if os.path.exists(json_file):
            print("Lecture du cache..")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.weather_grid = pd.DataFrame(data)
        else:
            self.weather_grid = self.__fetchDataBase()

            self.path.mkdir(parents=True, exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                self.weather_grid.to_json(json_file, orient='records', indent=4, date_format='iso')

    def __fetchDataBase(self) -> Tuple[list, list]:
        print("Fetching weather grid in database ...")

        path = self.path / 'era5_data.db'

        try:

            try:
                weather_grid = pd.read_sql("""
                    SELECT DISTINCT latitude, longitude FROM weather_data;
                """, sqlite3.connect(path) )

            except Exception as e:
                print(f"Error while fetching db (no data?): {e}")
                sys.exit()

        except Exception as e:
            print(f"Error while fetching db (no data?): {e}")
            flow_date, flow_region = [], []

        return weather_grid