import numpy as np
import cupy as cp
import cv2
import time
from concurrent.futures import ThreadPoolExecutor as Pool
from copy import deepcopy

class RandomFigure:
    def __init__(self, r=700, l=500):
        self.img = np.zeros((512,512), np.uint8)
        # self.img = self.random()
        for i in range(np.random.randint(4, 7)):
            self.img = self.random(img=self.img)
        self.rows = np.random.randint(2, r)
        self.cols = np.random.randint(2, l)
        self.img = cv2.resize(self.img , (self.cols, self.rows))
        
    def random(self, img):
        a = np.random.rand()
        if (a < 0.5):
            x = np.random.randint(10, 110)
            return cv2.ellipse(img,(np.random.randint(50,462),np.random.randint(50,462)),(x,170-x+np.random.randint(30)),np.random.randint(360),0,360,255,-1)
            # return cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
        elif(a < 0.8):
            x = np.random.randint(10, 110)
            img = cv2.rectangle(img,(np.random.randint(50,462),np.random.randint(50,462)),(x,170-x+np.random.randint(30)),255, -1)
            M = cv2.getRotationMatrix2D((256, 256),np.random.randint(360),1)
            return cv2.warpAffine(img,M,(512,512))
            # return cv2.rectangle(img,(384,0),(510,128),255,3)
        else:
            font = [cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC][np.random.randint(9)]
            txt = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[np.random.randint(52)]
            img = cv2.putText(img, txt, (np.random.randint(50,462),np.random.randint(50,462)), font, 3 + np.random.randint(15), 255, 2 + np.random.randint(40), cv2.LINE_AA, True)
            M = cv2.getRotationMatrix2D((256, 256),np.random.randint(360),1)
            return cv2.warpAffine(img,M,(512,512))

    def save(self, filename = "hoge.png"):
        cv2.imwrite(filename, self.img)

class Mosaic(RandomFigure):
    def __init__(self, image):
        self.imageRows, self.imageCols = image.shape[:2]
        super().__init__(r=self.imageRows, l=self.imageCols)
        self.image = image
        self.x = np.random.randint(self.imageRows - self.rows)
        self.y = np.random.randint(self.imageCols - self.cols)
    
    def saveImage(self, filename="hoge"):
        cv2.imwrite(filename, self.image)

class BlockMosaic(Mosaic):
    def __init__(self, image):
        super().__init__(image)

    def make(self):
        self.makeMosaic(np.random.randint(2, 40))

    def makeMosaic(self, k):
        a = cv2.resize(self.image[self.x:self.x+self.rows, self.y:self.y+self.cols], (max(self.cols//k, 1), max(self.rows//k, 1)), interpolation=np.random.choice([3, 5, 10, 2, 4, 1, 0]))
        self.b = (max(self.cols//k, 1)*k, max(self.rows//k, 1)*k)
        self.mosaic = cv2.resize(a, self.b, interpolation=cv2.INTER_NEAREST)

    def setMosaic(self):
        a = self.img[:self.b[1],:self.b[0]]>128
        # print(">----------")
        # print(self.b)
        # print(a.shape)
        # print(self.mosaic.shape)
        # print(self.image[self.x:self.x+self.b[1], self.y:self.y+self.b[0]].shape)
        # print("----------<")
        x = min(self.b[1], a.shape[0])
        y = min(self.b[0], a.shape[1])
        self.image[self.x:self.x+x, self.y:self.y+y][a[:x,:y]] = self.mosaic[:x,:y][a[:x,:y]]
        
        # self.image[self.x:self.x+self.b[1], self.y:self.y+self.b[0]] = self.mosaic

class AverageMosaic(Mosaic):
    def __init__(self, image):
        super().__init__(image)

    def make(self):
        self.makeMosaic(np.random.randint(2, 40))

    def makeMosaic(self, k):
        self.mosaic = cv2.blur(self.image[self.x:self.x+self.rows, self.y:self.y+self.cols], (k, k))

    def setMosaic(self):
        a = self.img > 127
        self.image[self.x:self.x+self.rows, self.y:self.y+self.cols][a] = self.mosaic[a]

class WatermarkMosaic(Mosaic):
    def __init__(self, image):
        super().__init__(image)
    
    def make(self):
        self.makeMosaic(np.random.randint(-120, 120, 3))

    def makeMosaic(self, b):
        self.mosaic = np.clip(self.image[self.x:self.x+self.rows, self.y:self.y+self.cols].astype(np.uint16)+np.array(b), 0, 255).astype(np.uint8)
    
    def setMosaic(self):
        self.image[self.x:self.x+self.rows, self.y:self.y+self.cols][self.img==255] = self.mosaic[self.img==255]


def main():
    t = time.time()
    img = cv2.imread("_.jpg")
    for i in range(1, 10):
        x = WatermarkMosaic(img)
        x.make()
        x.setMosaic()
        x.saveImage(f"{i}.png")
    print(time.time()-t)

if __name__ == '__main__':
    main()