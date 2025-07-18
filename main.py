import os
import zipfile


EMOJIS_ZIP = "emojis.zip"
EMOJIS_EXTRACTED = EMOJIS_ZIP.rstrip(".zip")
EMOJIS_SIZE_PIXELS = 32

class ImageToEmojis:
    def __init__(self):
        if not os.path.exists(EMOJIS_EXTRACTED):
            self._extract_emojis()
    
    def _extract_emojis(self):
        with zipfile.ZipFile(EMOJIS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EMOJIS_EXTRACTED)
        print(f"Emojis extracted to {EMOJIS_EXTRACTED}")
        

if __name__ == "__main__":
    app = ImageToEmojis()