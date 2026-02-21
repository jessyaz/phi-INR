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

def test_dbs_struct():
    print("Test structure ...")
    return False # Temporary


class NZStruct():

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.base_dir = self.db_path.parent
        json_file = self.base_dir / 'metadata.json'

        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = {k: pd.DataFrame(v) for k, v in data.items()}
        else:
            flow_date, flow_region = self.__fetch_from_db()
            self.metadata = {
                'flow_datetime': pd.DataFrame(flow_date, columns=['datetime']),
                'flow_region': pd.DataFrame(flow_region, columns=['region'])
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self.metadata.items()}, f, indent=4)

    def __fetch_from_db(self) -> Tuple[list, list]:
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                flow_date = conn.execute("SELECT DISTINCT DATETIME FROM flow;").fetchall()
                flow_region = conn.execute("SELECT DISTINCT REGION FROM flow_meta;").fetchall()
                return [r[0] for r in flow_date], [r[0] for r in flow_region]
            except Exception as e:
                print(f"Error fetching metadata: {e}")
                return [], []

    def getMetadata(self) -> dict:
        return self.metadata

    def getSiterefFromRegion(self, region: str) -> pd.DataFrame:

        with sqlite3.connect(str(self.db_path)) as conn:

            try:
                query = "SELECT DISTINCT SITEREF, LON, LAT FROM flow_meta WHERE REGION = ?"
                df = pd.read_sql(query, conn, params=(region,))
                return df
            except Exception as e:
                print(f"Error fetching siterefs: {e}")
                return pd.DataFrame(columns=['SITEREF', 'LON', 'LAT'])



class ERA5Struct():
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.base_dir = self.db_path.parent
        json_file = self.base_dir / 'weather_grid.json'
        self.weather_grid = None

        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.weather_grid = pd.DataFrame(data)
        else:
            self.weather_grid = self.__fetch_from_db()
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.weather_grid.to_json(json_file, orient='records', indent=4)

    def __fetch_from_db(self) -> pd.DataFrame:
        print("Fetching weather grid from database...")
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                df = pd.read_sql("SELECT DISTINCT latitude, longitude FROM weather_data;", conn)
                return df
        except Exception as e:
            print(f"Error fetching weather grid: {e}")
            return pd.DataFrame(columns=['latitude', 'longitude'])

    def get_grid(self) -> pd.DataFrame:
        return self.weather_grid