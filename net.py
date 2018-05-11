import chainer
import chainer.links as L
import chainer.functions as F


class Discriminator(chainer.Chain):
    """
    Discriminator

    Parametars
    ------------------
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # register layer with variable
        with self.init_scope():
            # initializers
            w = chainer.initializers.Normal(scale=0.02)
            self.c0 = L.Convolution2D(
                in_channels=None,
                out_channels=64,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w)
            self.c1 = L.Linear(in_size=None, out_size=256, initialW=w)
            self.c2 = L.Linear(in_size=None, out_size=1, initialW=w)

    def __call__(self, x):
        """
        Computes forward

        Parametors
        ----------------
        x: Variable
           input image data. this shape is (N, C, H, W)

        """
        h = F.leaky_relu(self.c0(x))
        # h = F.flatten(h)
        # print("flatten shape:", h.shape)
        h = F.dropout(F.leaky_relu(self.c1(h)), ratio=0.5)
        logits = self.c2(h)

        return logits


class Generator(chainer.Chain):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    Attributes
    ---------------------
    """

    def __init__(self, n_hidden=100, bottom_width=7, ch=128, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)  # initializers
            # w = chainer.initializers.HeNormal()  # He initialize value
            self.l0 = L.Linear(
                in_size=None,
                out_size=1024
                initialW=w,
                nobias=True)
            self.l1 = L.Linear(
                in_size=None,
                out_size=self.ch*self.bottom_width*self.bottom_width,
                initialW=w,
                nobias=True)
            self.dc2 = L.Deconvolution2D(
                in_channels=None,
                out_channels=64,
                ksize=6,
                stride=2,
                pad=2,
                initialW=w,
                nobias=True)  # (, 64, 14, 14)
            self.dc3 = L.Deconvolution2D(
                in_channels=None,
                out_channels=1,
                ksize=6,
                stride=2,
                pad=2,
                initialW=w,
                nobias=True)  # (, 1, 28, 28)

            self.bn0 = L.BatchNormalization(
                1024)
            self.bn1 = L.BatchNormalization(
                self.ch*self.bottom_width*self.bottom_width)
            self.bn2 = L.BatchNormalization(64)

    def make_hidden(self, batchsize):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        Parameters
        ----------------
        batchsize: int
           batchsize indicate len(z)

        Return
        ---------------
        noize: np.ndarray(shape=(batchsize, n_hidden), type=np.float32)

        """
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden))\
                        .astype(np.float32)

    def __call__(self, z):
        """
        Function that computs foward

        Parametors
        ----------------
        z: Variable
           random vector drown from a uniform distribution,
           this shape is (N, n_hidden)

        """
        h = F.leaky_relu(self.bn0(self.l0(z)))
        h = F.leaky_relu(self.bn1(self.l1(h)))
        h = F.reshape(h, (len(z), self.ch, self.bottom_width,
                          self.bottom_width))  # dataformat is NCHW
        h = F.leaky_relu(self.bn2(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.uniform(-1, 1, (1, 1, 28, 28)).astype("f")
    dis = Discriminator()
    logits = dis(Variable(z))

    # print(img)
    g = c.build_computational_graph(logits)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
