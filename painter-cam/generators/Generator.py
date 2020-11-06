from PIL import Image

class Generator:
    def __init__(self, image_path):
        self.img = Image.open(image_path)