import numpy as np
import cupy as cp
import cv2
import time

class RandomFigure:
    def __init__(self):
        self.img = np.zeros((512,512), np.uint8)
        # self.img = self.random()
        for i in range(np.random.randint(4, 7)):
            self.img = self.random(img=self.img)
        self.rows = np.random.randint(700)
        self.cols = 500 - self.rows + np.random.randint(400)
        self.img = cv2.resize(self.img , (self.rows, self.cols))
        
    def random(self, img):
        a = np.random.rand()
        if (a < 0.5):
            x = np.random.randint(10, 110)
            return cv2.ellipse(img,(np.random.randint(100,412),np.random.randint(100,412)),(x,170-x+np.random.randint(30)),np.random.randint(360),0,360,255,-1)
            # return cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
        elif(a < 0.8):
            x = np.random.randint(10, 110)
            img = cv2.rectangle(img,(np.random.randint(100,412),np.random.randint(100,412)),(x,170-x+np.random.randint(30)),255, -1)
            M = cv2.getRotationMatrix2D((256, 256),np.random.randint(360),1)
            return cv2.warpAffine(img,M,(512,512))
            # return cv2.rectangle(img,(384,0),(510,128),255,3)
        else:
            font = [cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC][np.random.randint(9)]
            txt = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[np.random.randint(52)]
            img = cv2.putText(img, txt, (np.random.randint(100,412),np.random.randint(100,412)), font, 3 + np.random.randint(15), 255, 2 + np.random.randint(40), cv2.LINE_AA, True)
            M = cv2.getRotationMatrix2D((256, 256),np.random.randint(360),1)
            return cv2.warpAffine(img,M,(512,512))

    def save(self, filename = "hoge.png"):
        cv2.imwrite(filename, self.img)

class Mosaic(RandomFigure):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.imageRows, self.imageCols = image.shape[:2]
        self.x = np.random.randint(self.imageRows - self.rows)
        self.y = np.random.randint(self.imageCols - self.cols)

class BlockMosaic(Mosaic):
    def __init__(self, image):
        super().__init__(image)

    def makeMosaic(self, k):
        a = cv2.resize(self.image[self.x:self.x+self.rows, self.y:self.y+self.cols], (self.imageRows//k, self.imageCols//k))
        self.b = (self.imageRows//k*k, self.imageCols//k*k)
        self.mosaic = cv2.resize(a, self.b)

    def setMosaic(self):
        self.image[self.x:self.x+self.b[0], self.y:self.y+self.b[1]][self.img==255] = self.mosaic[self.img[:self.b[0],:self.b[1]]==255]

class AverageMosaic(Mosaic):
    def __init__(self, image):
        super().__init__()

    def makeMosaic(self, k):
        self.mosaic = cv2.blur(self.image[self.x:self.x+self.rows, self.y:self.y+self.cols], (k, k))

    def setMosaic(self):
        self.image[self.x:self.x+self.rows, self.y:self.y+self.cols][self.img==255] = self.mosaic[self.img==255]

class WatermarkMosaic(Mosaic):
    def __init__(self, image):
        super().__init__()

    def makeMosaic(self, b):
        self.mosaic = np.clip(self.image[self.x:self.x+self.rows, self.y:self.y+self.cols].astype(np.uint16)+np.array(b), 0, 255).astype(np.uint8)
    
    def setMosaic(self):
        self.image[self.x:self.x+self.rows, self.y:self.y+self.cols][self.img==255] = self.mosaic[self.img==255]


def main():
    t = time.time()
    for i in range(10):
        x = RandomFigure()
        x.save(f"{i}.png")
    print(time.time()-t)

if __name__ == '__main__':
    main()