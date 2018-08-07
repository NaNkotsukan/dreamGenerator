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
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, pad=(x.shape[2]%2, x.shape[3]%2))
        return h

class Up(Chain):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=3, pad=1)
    
    def __call__(self, x0, x1):
        h = F.concat(x0, pixelShuffler(x1, 2)[:,:,x0.shape[1]%2:,x0.shape[1]%2:])
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        return h

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.down0 = Down(3, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.down3 = Down(128, 256)
            self.down4 = Down(256, 512)
            self.down5 = Down(512, 768)
            self.down6 = Down(768, 1024)
            self.down7 = Down(1024, 1024)

            self.up0 = Up(1280, 1024)
            self.up1 = Up(1024, 1024)
            self.up2 = Up(768, 512)
            self.up3 = Up(384, 256)
            self.up4 = Up(192, 128)
            self.up5 = Up(96, 64)
            self.up6 = Up(48, 32)

            self.conv = L.Convolution2D(32, 3, ksize=1)
    
    def __call__(self, x):
        h0 = self.down0(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        h5 = self.down5(h4)
        h6 = self.down6(h5)
        h7 = self.down7(h6)

        h = self.up0(h6, h7)
        h = self.up1(h5, h)
        h = self.up2(h4, h)
        h = self.up3(h3, h)
        h = self.up4(h2, h)
        h = self.up5(h1, h)
        h = self.up6(h0, h)
        h = self.conv(h)
        return h

