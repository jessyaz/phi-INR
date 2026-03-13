import duckdb
from tqdm import tqdm

def extract_and_split_data(db_path, num_regions=4):
    con = duckdb.connect()

    with tqdm(total=4, desc="Initialisation", unit="step") as pbar:


        con.execute(f"ATTACH '{db_path}' AS src (TYPE sqlite, READ_ONLY)")

        north_regions = (
            'Waikato', 'Hawkes Bay', 'Northland', 'Auckland', 'Wellington',
            'Bay of Plenty', 'Manawatu-Wanganui', 'Taranaki', 'Gisborne'
        )

        north_regions__ = " OR ".join([f"REGION LIKE '%{r}'" for r in north_regions])
        # A refacto, ambiguité site_info lors de la crea


        query = f"""
            SELECT DISTINCT SITEREF 
            FROM src.site_info 
            WHERE {north_regions__}
        """

        north_sites = con.execute(query).fetchdf()['siteref'].tolist()


        pbar.update(1)

        sites_list = ', '.join(f"'{s}'" for s in north_sites)

        con.execute(f"""
            CREATE TEMP VIEW filtered AS
            SELECT *,
                year(DATETIME) AS yr
            FROM src.data
            WHERE SITEREF IN ({sites_list})
              AND REGION IN (SELECT DISTINCT REGION FROM src.data)
              AND DATETIME >= '2014-01-01'
              AND DATETIME  < '2019-01-01'
        """)

        pbar.update(1)

        con.execute("COPY (SELECT * FROM filtered WHERE yr IN (2014,2015,2016)) TO 'train_data.parquet' (FORMAT PARQUET)")
        con.execute("COPY (SELECT * FROM filtered WHERE yr = 2017)              TO 'val_data.parquet'   (FORMAT PARQUET)")
        con.execute("COPY (SELECT * FROM filtered WHERE yr = 2018)              TO 'test_data.parquet'  (FORMAT PARQUET)")

        con.execute("COPY (SELECT * FROM src.catalog)                           TO 'metadata.parquet'  (FORMAT PARQUET)")


        pbar.update(1)

        for split, f in [('Train', 'train_data.parquet'), ('Val', 'val_data.parquet'), ('Test', 'test_data.parquet')]:
            n = con.execute(f"SELECT COUNT(*) FROM '{f}'").fetchone()[0]
            print(f"{split} : {n:,} lignes")

        pbar.update(1)

        con.close()

if __name__ == "__main__":
    extract_and_split_data("/home/jazizi/dev/data-nz-era5-processing/nz/data/processed/db.db")