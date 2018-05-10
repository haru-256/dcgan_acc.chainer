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