from NODE.NODE import *


class Lorenz(ODEF):
    """
    chaotic lorenz system
    """
    def __init__(self):
        super(Lorenz, self).__init__()
        self.lin = nn.Linear(5, 3, bias=False)
        W = Tensor([[-10, 10, 0, 0, 0],
                    [28, -1, 0, -1, 0],
                    [0, 0, -8 / 3, 0, 1]])
        self.lin.weight = nn.Parameter(W)

    def forward(self, x):
        bs, _, dim = x.shape
        y = y = torch.ones([bs, 5])
        y[:, 0] = x[:, :, 0]
        y[:, 1] = x[:, :, 1]
        y[:, 2] = x[:, :, 2]
        y[:, 3] = x[:, :, 0] * x[:, :, 2]
        y[:, 4] = x[:, :, 0] * x[:, :, 1]
        x_dot = self.lin(y)
        return x_dot.view(bs, -1, dim)


class LorenzLimitCycle(ODEF):
    """
    modified lorenz system which forms a limit cycle
    """
    def __init__(self):
        super(LorenzLimitCycle, self).__init__()
        self.lin = nn.Linear(5, 3, bias=False)
        W = Tensor([[-10, 10, 0, 0, 0],
                    [-4.8, 7.2, 0, -1, 0],
                    [0, 0, -8 / 3, 0, 1]])
        self.lin.weight = nn.Parameter(W)

    def forward(self, x):
        y = y = torch.ones([1, 5])
        y[0][0] = x[0][0]
        y[0][1] = x[0][1]
        y[0][2] = x[0][2]
        y[0][3] = x[0][0] * x[0][2]
        y[0][4] = x[0][0] * x[0][1]
        return self.lin(y)


class LorenzSindy(ODEF):
    """
    incorrectly identified lorenz system using SINDy
    """
    def __init__(self):
        super(LorenzSindy, self).__init__()
        self.lin = nn.Linear(5, 3, bias=False)
        # system identified by SINDy using the correct nonlinearities
        W = Tensor([[-9.913, 9.913, 0, 0, 0],
                    [27.212, -0.848, 0, -0.978, 0],
                    [0, 0, -2.636, 0, 0.988]])
        self.lin.weight = nn.Parameter(W)

    def forward(self, x):
        y = y = torch.ones([1, 5])
        y[0][0] = x[0][0]
        y[0][1] = x[0][1]
        y[0][2] = x[0][2]
        y[0][3] = x[0][0] * x[0][2]
        y[0][4] = x[0][0] * x[0][1]
        return self.lin(y)

