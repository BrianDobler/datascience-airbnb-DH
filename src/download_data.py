import os
import gdown


RAW_PATH = "data/raw"
os.makedirs(RAW_PATH, exist_ok=True)


FOLDER_URL = "https://drive.google.com/drive/folders/1xKTfaKZDYLadXDW8H-HKG8FZ6vXaLdul"

print("Descargando datasets desde Google Drive...")

gdown.download_folder(
    url=FOLDER_URL,
    output=RAW_PATH,
    quiet=False,
    use_cookies=False
)

print("\nDescarga finalizada.")
print(f"Los archivos quedaron en: {RAW_PATH}")
