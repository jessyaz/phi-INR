import cdsapi
import os

"""
SCRIPT A REVOIR -> COMPATIBLE SUR APT21 MAIS PAS EN LOCAL
/!\ CDSAPI pas compatible sur python 313 -> nz-env2 : ### Refaire un environnement stable pour le projet
"""


c = cdsapi.Client()

output_folder = "../../data/raw/era5_downloads"

if not os.path.isdir(output_folder):
    print(f"Le dossier {output_folder} n'existe pas")
    sys.exit(1)

print("ok")

sys.exit(1)


# Zone
AREA = [-34, 166, -48, 179]

# Variables
VARIABLES = [
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '2m_dewpoint_temperature', '2m_temperature',
    'mean_sea_level_pressure', 'total_precipitation',
    'surface_solar_radiation_downwards', 'total_cloud_cover',
    'convective_precipitation',
]

# Boucle Année
for year in range(2013, 2023):
    # Boucle Mois
    for month in range(1, 13):

        # Nom du fichier : ex: era5_2013_01.grib
        filename = f"era5_{year}_{month:02d}.grib"
        filepath = os.path.join(output_folder, filename)

        # Reprise sur erreur : si le fichier existe, on passe au suivant
        if os.path.exists(filepath):
            print(f"✅ {filename} existe déjà. Ignoré.")
            continue

        print(f"⏳ Téléchargement : {year}-{month:02d} ...")

        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': VARIABLES,
                    'year': str(year),
                    'month': f"{month:02d}", # Convertit 1 en "01", etc.
                    'day': [
                        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
                    ],
                    'area': AREA,
                },
                filepath)

        except Exception as e:
            print(f"Erreur sur {filename} : {e}")

print("Téléchargement terminé !")