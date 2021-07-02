class VanderPolTrain(ODEF):
    """
    neural network for learning the stiff van der pol oscillator
    """
    def __init__(self):
        super(VanderPolTrain, self).__init__()
        self.lin1 = nn.Linear(3, 512, bias=False)
        self.lin2 = nn.Linear(512, 32, bias=False)
        self.lin3 = nn.Linear(32, 2, bias=False)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus(beta=1.8)

    def forward(self, t, x):
        if isinstance(x, np.ndarray): x = FloatTensor(x)
        x = x.float()
        x = self.tanh(self.lin1(x))
        x = self.softplus(self.lin2(x))
        x = self.lin3(x)

        x = x.view(1, -1)
        x = torch.cat([x, Tensor([[0]])], 1)

        return x