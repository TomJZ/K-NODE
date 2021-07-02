from NODE.NODE import *


class KS_conv64(ODEF):
    """
    neural network for learning the KS equation
    """
    def __init__(self):
        super(KS_conv64, self).__init__()
        # Encoder
        bias = False
        padding_mode = 'replicate'
        self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv2 = nn.Conv1d(32, 256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv5 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)
        self.enc_conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=bias)

        self.lin1 = nn.Linear(2048, 64, bias=bias)

        self.relu = nn.LeakyReLU(0.05)
        self.tanh = nn.Tanh()

    def forward(self, x, t):
        x = x.view(1, 1, -1)
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = self.tanh(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = self.relu(self.enc_conv5(x))
        x = self.enc_conv6(x)

        x = x.view(-1)
        x = self.lin1(x)
        x = x.view(1, -1)
        return x