from NODE.NODE import *

class VanderPol(ODEF):
    """
    The Van der Pol oscillator
    """
    def __init__(self):
        super(VanderPol, self).__init__()
        self.lin = nn.Linear(4, 3, bias=False)
        W = Tensor([[0, 1, 0, 0],
                    [-1, 0, 1, -1],
                    [0, 0, 0, 0]])
        self.lin.weight = nn.Parameter(W)

    def forward(self, t, x):
        try:
            y = torch.ones([1, 4])
            y[0][0] = x[0][0]
            y[0][1] = x[0][1]
            y[0][2] = x[0][1] * x[0][2]
            y[0][3] = x[0][1] * x[0][0] ** 2 * x[0][2]
            y = self.lin(y)
        except:
            y = np.zeros(4)
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[1] * x[2]
            y[3] = x[1] * x[0] ** 2 * x[2]
            y = self.lin(Tensor(y))
            y = y.view(1, -1)
        return y