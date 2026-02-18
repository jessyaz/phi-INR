import sqlite3
conn = sqlite3.connect("../../data/raw/era5_downloads/era5_data.db")

test_time = '2013-02-17 23:00:00'
res = conn.execute("SELECT count(*) FROM weather_data WHERE time = ?", (test_time,)).fetchone()


import pandas as pd

df = pd.read_sql("SELECT * FROM weather_data WHERE time = ?",conn,params=(test_time) )

print(df.head())
print(f"Nombre de lignes pour {test_time} : {res[0]}")
