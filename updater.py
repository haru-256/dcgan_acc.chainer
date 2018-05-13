import chainer
from chainer import Variable
import chainer.functions as F


class DCGANUpdater(chainer.training.StandardUpdater):
    """
    costomized updater for DCGAN

    Parameters
    ------------------
    models: tuple of Links
        elements of this tuple are Generator and Discriminator

    iterator: chainer.iterators
       iterator of train dataset whose range is -1 ~ 1

    optimizer: dict of Optimizer
       Optimizer to update parameters. It can also be a dictionary  
        that maps strings to optimizers. "gen", "dis" indicate Generator and Discriminator respectively
    """

    def __init__(self, models, optimizer, iterator, device, *args, **kwargs):
        self.gen, self.dis = models
        super(DCGANUpdater, self).__init__(iterator, optimizer, device=device)

    def update_core(self):
        """
        This method implements the things to do in one iteration
        """
        gen_optimizer = self.get_optimizer("gen")
        dis_optimizer = self.get_optimizer("dis")

        # obtain batc data
        batch = self.get_iterator("main").next()

        # inference of real
        x_real = Variable(self.converter(batch, self.device))
        y_real = self.dis(x_real)

        # inference of fake
        xp = chainer.backends.cuda.get_array_module(
            x_real.data)  # adapt to GPU
        z = Variable(xp.asarray(self.gen.make_hidden(len(batch))))
        x_fake = self.gen(z)
        y_fake = self.dis(x_fake)

        # update Generatar and Discriminator
        gen_optimizer.update(self.loss_gen, gen, y_fake)
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)

    def loss_dis(self, dis, y_fake, y_real):
        """
        calucurate Discriminator loss

        Parametrs
        ----------------
        dis: chainer.Link
            Discriminator

        y_fake: Variable
            Estimation result of real image

        y_real: Variable
            Estimation result of fake image

        Returns
        ----------------
        loss: Variable
            Discriminator Loss
        """
        L1 = F.sum(F.softplus(-y_real)) / len(y_real)
        L2 = F.sum(F.softplus(y_fake)) / len(y_fake)
        loss = L1 + L2
        # https://docs.chainer.org/en/stable/reference/util/generated/chainer.Reporter.html?highlight=Observer
        # https://github.com/chainer/chainer/blob/v4.0.0/chainer/training/trainer.py#L144-L149
        chainer.report({'loss': loss}, dis)  # observer は Trainerクラスが作ってくれる

        return loss

    def loss_gen(self, gen, y_fake):
        """
        calucurate Generator loss

        Parametrs
        ----------------
        gen: chainer.Link
            Generator

        y_real: Variable
            Estimation result of fake image

        Returns
        ----------------
        loss: Variable
            Generator Loss
        """

        loss = F.sum(F.softplus(-y_fake)) / len(y_fake)
        chainer.report({'loss': loss}, gen)

        return loss
