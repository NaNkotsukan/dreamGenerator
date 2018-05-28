import numpy as np
import cupy as cp
import cv2


class img:
    def __init__(self, path):
        self.image = cv2.imread(path)
    
    def noise(self):
        pass

class data:
    def __init__(self):
        pass

    def load(self):
        pass
