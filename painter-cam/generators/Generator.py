import os
from PIL import Image

class Generator:
    def __init__(self, input_path, output_path):
        #self.img = Image.open(input_path)
        #self.input_path = input_path
        self.setImage(input_path)
        self.output_path = output_path

    def setImage(self, input_path):
        assert os.path.exists(input_path) # make sure the file exists
        self.input_path = input_path
        self.img = Image.open(input_path)
        #self.generate()

    # TODO: make a function to get image from array instead of file

    def generate(self):
        raise NotImplementedError # placeholder