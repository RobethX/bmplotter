import os
from PIL import Image

class Generator:
    def __init__(self, img=None, input_path=None):
        if img.size:
            self.setImage(img)
        elif input_path:
            self.loadImage(input_path)

    def setImage(self, img):
        self.img = Image.fromarray(img, mode="L")
        # TODO: verify image

    def loadImage(self, input_path):
        assert os.path.exists(input_path) # make sure the file exists
        self.input_path = input_path
        self.img = Image.open(input_path)
        #self.generate()

    # TODO: make a function to get image from array instead of file

    def generate(self):
        raise NotImplementedError # placeholder