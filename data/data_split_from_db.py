import duckdb
from tqdm import tqdm

def extract_and_split_data(db_path, num_regions=4):
    con = duckdb.connect()

    # Régions Nord (île du Nord)
    north_regions = (
        'Waikato', 'Hawkes Bay', 'Northland', 'Auckland', 'Wellington',
        'Bay of Plenty', 'Manawatu-Wanganui', 'Taranaki', 'Gisborne'
    )

    # Régions Sud (île du Sud) — pour le test OOD
    south_regions = (
        'Canterbury', 'Otago', 'Southland', 'Marlborough',
        'Nelson', 'Tasman', 'West Coast'
    )

    OOD_START = '2018-01-01'
    OOD_END   = '2019-01-01'

    with tqdm(total=4, desc="Initialisation", unit="step") as pbar:

        con.execute(f"ATTACH '{db_path}' AS src (TYPE sqlite, READ_ONLY)")

        # --- Récupération des sites Nord ---
        north_filter = " OR ".join([f"REGION LIKE '%{r}'" for r in north_regions])
        north_sites  = con.execute(f"""
            SELECT DISTINCT SITEREF
            FROM src.site_info
            WHERE {north_filter}
        """).fetchdf()['siteref'].tolist()

        # --- Récupération des sites Sud ---
        south_filter = " OR ".join([f"REGION LIKE '%{r}'" for r in south_regions])
        south_sites  = con.execute(f"""
            SELECT DISTINCT SITEREF
            FROM src.site_info
            WHERE {south_filter}
        """).fetchdf()['siteref'].tolist()

        pbar.update(1)

        # --- Vue filtrée Nord : 2014–2018 ---
        north_list = ', '.join(f"'{s}'" for s in north_sites)
        con.execute(f"""
            CREATE TEMP VIEW filtered_north AS
            SELECT *,
                year(DATETIME) AS yr
            FROM src.data
            WHERE SITEREF IN ({north_list})
              AND DATETIME >= '2014-01-01'
              AND DATETIME  < '2019-01-01'
        """)

        # --- Vue filtrée Sud : 6 mois 2018 pour l'OOD ---
        south_list = ', '.join(f"'{s}'" for s in south_sites)
        con.execute(f"""
            CREATE TEMP VIEW filtered_south AS
            SELECT *,
                year(DATETIME) AS yr
            FROM src.data
            WHERE SITEREF IN ({south_list})
              AND DATETIME >= '{OOD_START}'
              AND DATETIME  < '{OOD_END}'
        """)

        pbar.update(1)

        # --- Exports ---
        # Train  : Nord, années 2014–2016
        con.execute("COPY (SELECT * FROM filtered_north WHERE yr IN (2014, 2015, 2016)) TO 'train_data.parquet'    (FORMAT PARQUET)")
        # Val    : Nord, année 2017
        con.execute("COPY (SELECT * FROM filtered_north WHERE yr = 2017)               TO 'val_data.parquet'      (FORMAT PARQUET)")
        # Test   : Nord, année 2018  — distribution in-domain
        con.execute("COPY (SELECT * FROM filtered_north WHERE yr = 2018)               TO 'test_north_data.parquet' (FORMAT PARQUET)")
        # Test OOD : Sud, 6 mois 2018 — distribution out-of-domain
        con.execute("COPY (SELECT * FROM filtered_south)                               TO 'test_south_ood_data.parquet' (FORMAT PARQUET)")

        # Métadonnées
        con.execute("COPY (SELECT * FROM src.catalog)                                  TO 'metadata.parquet'      (FORMAT PARQUET)")

        pbar.update(1)

        # --- Résumé ---
        splits = [
            ('Train',         'train_data.parquet',         'Nord  | 2014–2016'),
            ('Val',           'val_data.parquet',           'Nord  | 2017'),
            ('Test (in-dist)','test_north_data.parquet',    'Nord  | 2018'),
            ('Test OOD',      'test_south_ood_data.parquet',f'Sud   | {OOD_START} → {OOD_END}'),
        ]
        print()
        for label, f, desc in splits:
            n = con.execute(f"SELECT COUNT(*) FROM '{f}'").fetchone()[0]
            print(f"  {label:<18} ({desc}) : {n:>10,} lignes")

        pbar.update(1)

    con.close()


if __name__ == "__main__":
    extract_and_split_data("/home/jazizi/dev/data-nz-era5-processing/nz/data/processed/db.db")