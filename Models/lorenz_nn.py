from NODE.NODE import *


class LorenzTrain(ODEF):
    """
    neural network for learning the chaotic lorenz system
    """
    def __init__(self):
        super(LorenzTrain, self).__init__()
        self.lin = nn.Linear(3, 256)
        self.lin3 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin(x))
        x = self.lin3(x)
        return x


class LorenzSindyKNODE(ODEF):
    """
    KNODE combining incorrectly SINDy-identified lorenz system and a neural network
    """
    def __init__(self):
        super(LorenzSindyKNODE, self).__init__()
        self.lin_im = nn.Linear(6, 3, bias=False)
        # xy and xz are excluded from the library of functions
        self.W = Tensor([[-9.913, 9.913, 0, 0, 0, 0],
                         [-7.175, 20.507, 0, -0.613, 0, 0],
                         [0, 0, -3.05, 0, 0.504, 0.479]])

        self.lin_im.weight = nn.Parameter(self.W)

        self.lin1 = nn.Linear(3, 32)
        self.lin2 = nn.Linear(32, 512)
        self.lin3 = nn.Linear(512, 32)
        self.lin4 = nn.Linear(32, 3)

        self.Mout = nn.Linear(6, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 3)
        bs, _, _ = x.size()
        y = torch.zeros([bs, 1, 6])
        y[:, :, 0] = x[:, :, 0]
        y[:, :, 1] = x[:, :, 1]
        y[:, :, 2] = x[:, :, 2]
        y[:, :, 3] = x[:, :, 1] * x[:, :, 2]
        y[:, :, 4] = x[:, :, 0] ** 2
        y[:, :, 5] = x[:, :, 1] ** 2

        y = self.lin_im(y)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.lin4(x)

        x = self.Mout(torch.cat([x, y], 2))
        return x


class LorenzModifiedKNODE(ODEF):
    """
    KNODE combining limit cycle lorenz system and a neural network
    """
    def __init__(self):
        super(LorenzModifiedKNODE, self).__init__()
        # imperfect model
        self.lin_im = nn.Linear(5, 3, bias=False)
        # Using inaccurate coefficients
        self.W = Tensor([[-10, 10, 0, 0, 0],
                         [-4.8, 7.2, 0, -1, 0],
                         [0, 0, -8 / 3, 0, 1]])
        self.lin_im.weight = nn.Parameter(self.W)

        # neural network
        self.lin1 = nn.Linear(3, 32)
        self.lin2 = nn.Linear(32, 512)
        self.lin3 = nn.Linear(512, 32)
        self.lin4 = nn.Linear(32, 3)
        self.Mout = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 3)
        bs, _, _ = x.size()
        y = torch.zeros([bs, 1, 5])
        y[:, :, 0] = x[:, :, 0]
        y[:, :, 1] = x[:, :, 1]
        y[:, :, 2] = x[:, :, 2]
        y[:, :, 3] = x[:, :, 0] * x[:, :, 2]
        y[:, :, 4] = x[:, :, 0] * x[:, :, 1]
        y = self.lin_im(y)

        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.lin4(x)
        x = self.Mout(torch.cat([x, y], 2))
        return x
