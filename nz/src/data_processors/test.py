import sqlite3
conn = sqlite3.connect("../../data/raw/era5_downloads/era5_data.db")

# Test sur une heure précise qui a échoué dans ton log
test_time = '2013-02-17 23:00:00'
res = conn.execute("SELECT count(*) FROM weather_data WHERE time = ?", (test_time,)).fetchone()

print(f"Nombre de lignes pour {test_time} : {res[0]}")

# Si c'est 0, alors le problème vient de ton IMPORT GRIB -> SQL
# Si c'est > 0, alors le problème vient de ton code de MATCH (Type de donnée)