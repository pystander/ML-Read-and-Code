import torch.nn as nn

class GAN(nn.Module):
    """
    A min-max game between a generative model (team of counterfeiters) and a discriminative model (police).

    Summary:
    •   Deep generative models: Difficult to approximatie probabilistic computations in MLE, and leverage benefits of PiLU
    •   Eliminate Markov chains in GSNs
    •   Different from adversarial examples (not for training a model)
    •   Simultaneously train D, and train G to minimize log(1 - D(G(z))) until Nash equilibrium D(x) = 1/2 (two-sample test)
        min_G max_D V(D, G) = E_x~p_data(x)[log(D(x))] + E_z~p_z(z)[log(1 - D(G(z)))]
        y -> a log(y) + b log(1 - y) achieves maximum in [0, 1] at a / (a + b)
    •   In general, GANs are unstable to train and converge

    Theorems and Propositions:
    1.  The global minimum of the virtual training criterion C(G) is achieved if and only if p_g = p_data. At that point, C(G) achieves the value - log 4
        KL(p ∥ g) = E_x~p[log(p(x) / g(x))]
    2.  If G and D have enough capacity, and at each step of Algorithm 1,
        the discriminator is allowed to reach its optimum given G, and pg is updated so as to improve the criterion

    Interesting Stories:
    •   Jürgen Schmidhuber believes GAN is a variant of predictability minimization (PM) - "Reverse PM"

    Referneces:
    https://arxiv.org/pdf/1406.2661.pdf
    https://papers.nips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    https://github.com/goodfeli/adversarial
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
    """

    class Generator(nn.Module):
        """
        A MLP that generates samples from a distribution.
        """
        def __init__(self):
            super(GAN.Generator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(100, 128), # 100 -> 128
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(128, 256), # 128 -> 256
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(256, 512), # 256 -> 512
                nn.BatchNorm1d(512, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(512, 1024), # 512 -> 1024
                nn.BatchNorm1d(1024, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(1024, 784), # 1024 -> 784 (1 x 28 x 28)
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), 1, 28, 28) # (64 x 1 x 28 x 28)
            return img

    class Discriminator(nn.Module):
        """
        A MLP that classifies samples as real or fake.
        """
        def __init__(self):
            super(GAN.Discriminator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(784, 512), # 784 -> 512
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(512, 256), # 512 -> 256
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(256, 1), # 256 -> 1
                nn.Sigmoid()
            )

        def forward(self, img):
            img = img.view(img.size(0), -1) # (64 x 784)
            validity = self.model(img)
            return validity
