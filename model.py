import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers
from chainer import Variable as V
import cupy as xp

def pixelShuffler(x, r):
    batchsize = x.shape[0]
    in_channels = x.shape[1]
    out_channels = in_channels // (r ** 2)
    in_height = x.shape[2]
    in_width = x.shape[3]
    out_height = in_height * r
    out_width = in_width * r
    h = F.reshape(x, (batchsize, r, r, out_channels, in_height, in_width))
    h = F.transpose(h, (0, 3, 4, 1, 5, 2))
    h = F.reshape(h, (batchsize, out_channels, out_height, out_width))
    return h

class Down(Chain):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=3, pad=1)
    
    def __call__(self, x):
        h = F.max_pooling_2d(x, 2)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        return h

class Up(Chain):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=3, pad=1)
    
    def __call__(self, x0, x1):
        h = F.concat((x0, pixelShuffler(x1, 2)[:,:,:x0.shape[2],:x0.shape[3]]))
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        return h

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            n = 32
            self.down0 = Down(n, n*2)
            self.down1 = Down(n*2, n*4)
            self.down2 = Down(n*4, n*8)
            self.down3 = Down(n*8, n*16)
            self.down4 = Down(n*16, n*24)
            self.down5 = Down(n*24, n*32)
            self.down6 = Down(n*32, n*32)
            self.down7 = Down(n*32, n*32)

            self.up0 = Up(n*40, n*32)
            self.up1 = Up(n*40, n*32)
            self.up2 = Up(n*32, n*24)
            self.up3 = Up(n*22, n*16)
            self.up4 = Up(n*12, n*8)
            self.up5 = Up(n*6, n*4)
            self.up6 = Up(n*3, n*2)
            self.up7 = Up(n//2+n, 32)

            self.conv0 = L.Convolution2D(3, 32, ksize=3, pad=1)
            self.conv1 = L.Convolution2D(32, 32, ksize=3, pad=1)
            self.conv = L.Convolution2D(32, 3, ksize=1)
    
    def __call__(self, x):
        h = self.conv0(x)
        h = F.relu(h)
        h = self.conv1(h)
        h0 = F.relu(h)
        h1 = self.down0(h0)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)
        h6 = self.down5(h5)
        h7 = self.down6(h6)
        h8 = self.down7(h7)

        h = self.up0(h7, h8)
        h = self.up1(h6, h)
        h = self.up2(h5, h)
        h = self.up3(h4, h)
        h = self.up4(h3, h)
        h = self.up5(h2, h)
        h = self.up6(h1, h)
        h = self.up7(h0, h)
        h = self.conv(h)
        return h

