from NODE.NODE import *


class HopfNormalTrain(ODEF):
    """
    neural network for learning the hopf normal form
    """
    def __init__(self):
        super(HopfNormalTrain, self).__init__()
        self.lin = nn.Linear(3, 256)
        self.lin1 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin(x))
        x = self.lin1(x)
        x = x.view(-1,3)
        x[:, 2] = 0
        return x.unsqueeze(1)
