from NODE.NODE import *


class HopfNormal(ODEF):
    """
    Hopf normal form
    """
    def __init__(self):
        super(HopfNormal, self).__init__()
        self.lin = nn.Linear(9, 3, bias=False)
        W = Tensor([[0, 1, 0, 1, 0, -1, -1, 0, 0],
                    [-1, 0, 0, 0, 1, 0, 0, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.lin.weight = nn.Parameter(W)

    def forward(self, x):
        y = torch.ones(1, 9)
        y[0][0] = x[0][0]
        y[0][1] = x[0][1]
        y[0][2] = x[0][2]
        y[0][3] = x[0][0] * x[0][2]
        y[0][4] = x[0][1] * x[0][2]
        y[0][5] = x[0][0] ** 3
        y[0][6] = x[0][0] * x[0][1] ** 2
        y[0][7] = x[0][1] ** 3
        y[0][8] = x[0][1] * x[0][0] ** 2
        y = self.lin(y)
        return y
