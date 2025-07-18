import os
import zipfile
import shutil
from PIL import Image


EMOJIS_ZIP = "emojis.zip"
EMOJIS_EXTRACTED = EMOJIS_ZIP.rstrip(".zip")
EMOJIS_SCALED_PATH = 'emojis_scaled'
EMOJIS_SIZE_PIXELS = 32

class ImageToEmojis:
    def __init__(self):
        if not os.path.exists(EMOJIS_EXTRACTED):
            self.extract_emojis()
            
        self.scale_emojis()
        
        self.emojis = self.load_emojis()
    
    def extract_emojis(self):
        if os.path.exists(EMOJIS_EXTRACTED) and os.listdir(EMOJIS_EXTRACTED):
            return
        with zipfile.ZipFile(EMOJIS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"Emojis extracted to {EMOJIS_EXTRACTED}")
        
    def scale_emojis(self):
        if not os.path.exists(EMOJIS_SCALED_PATH):
            os.makedirs(EMOJIS_SCALED_PATH, exist_ok=True)

        for emoji_file in os.listdir(EMOJIS_EXTRACTED):
            if emoji_file.endswith('.png'):
                scaled_path = os.path.join(EMOJIS_SCALED_PATH, emoji_file)
                if os.path.exists(scaled_path):
                    print(f"{scaled_path} already exists. Skipping.")
                    continue
                emoji_path = os.path.join(EMOJIS_EXTRACTED, emoji_file)
                with Image.open(emoji_path) as img:
                    img = img.resize((EMOJIS_SIZE_PIXELS, EMOJIS_SIZE_PIXELS), Image.Resampling.LANCZOS)
                    img.save(scaled_path)
                    print(f"Scaled {emoji_file} to {scaled_path}")
                    
    def load_emojis(self):
        if not os.path.exists(EMOJIS_SCALED_PATH):
            print(f"Scaled emojis path {EMOJIS_SCALED_PATH} does not exist.")
            return
        
        emojis = []
        for emoji_file in os.listdir(EMOJIS_SCALED_PATH):
            emoji_path = os.path.join(EMOJIS_SCALED_PATH, emoji_file)
            if os.path.isfile(emoji_path):
                emoji = Image.open(emoji_path)
                emojis.append(emoji)
                print(f"Loaded emoji from {emoji_path}")
        

if __name__ == "__main__":
    app = ImageToEmojis()