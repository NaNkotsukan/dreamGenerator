import numpy as np
import cupy as cp
import cv2
from mosaic import BlockMosaic, AverageMosaic, WatermarkMosaic
from concurrent.futures.thread import ThreadPoolExecutor as Pool
import os, pickle
from copy import deepcopy

class Img:
    def __init__(self, path, pool):
        self.y = cv2.imread(path)
        self.x = deepcopy(self.y)
        self.pool = pool
    
    def noise(self, n):
        noiseList = [[BlockMosaic, AverageMosaic, WatermarkMosaic][np.random.randint(3)](self.x) for i in range(n)]
        [x.result() for x in [self.pool.submit(x.make) for x in noiseList]]
        [x.setMosaic() for x in noiseList]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['pool']
        return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self.pool = self.
    

class Data:
    def __init__(self, path):
        self.path = path
        self.pool = Pool(max_workers=8)
        # self.imgList = [Img(x, self.pool) for x in os.listdir(path)]
        self.imgList = os.listdir(path)
        self.getImg = self.genImg()
    
    def genImg(self):
        while True:
            for i in np.random.permutation(len(self.imgList)):
                yield self.path+self.imgList[i]
            print("-------------------------------------------------")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['getImg']
        del state['pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.getImg = self.genImg()
    
    def load(self, n):
        img = [Img(next(self.getImg), self.pool) for _ in range(n)]
        x = []
        y = []
        for a in img:
            a.noise(int(np.abs(np.random.normal(0, 2)))+1)
            x.append(a.x.reshape((1,)+a.x.shape))
            y.append(a.y.reshape((1,)+a.x.shape))
        return cp.transpose(cp.vstack(x), (0, 3, 1, 2)).astype(cp.float32)/255, cp.transpose(cp.vstack(y), (0, 3, 1, 2)).astype(cp.float32)/255


def main():
    path = "D:/data/dataset/dream/pix"
    hoge = Data(path)
    with open('data.pickle', 'wb') as f:
        pickle.dump(hoge, f)


if __name__ == '__main__':
    main()