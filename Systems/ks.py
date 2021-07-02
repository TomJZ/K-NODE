import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from Utils.Plotters import *
"""
Simulating the Kumaramoto-Sivashinsky Equation
"""
class KSSimulation(object):
    def __init__(self):
        ############## simulation parameters ###############
        self.n_steps = 5000 # total number of steps to simulate
        self.transient = 1000 # number of steps to discard as transient state
        self.d = 60  # periodicity length
        self.tau = 0.25  # time step
        self.N = 64  # number of grid points
        self.const = 0  # error value
        self.E, self.E2, self.Q, self.f1, self.f2, self.f3, self.g = self.precompute_KS()

    def KS_pred(self, u, param):
        v = fft(u)
        vv = np.zeros(self.N, self.n_step)
        vv[:, 0] = v
        for i in range(self.n_step):
            Nv = self.g * fft(np.real(ifft(v)) ** 2)
            a = self.E2 * v + self.Q * Nv
            Na = self.g * fft(np.real(ifft(a)) ** 2)
            b = self.E2 * v + self.Q * Na
            Nb = self.g * fft(np.real(ifft(b)) ** 2)
            c = self.E2 * a + self.Q * (2 * Nb - Nv)
            Nc = self.g * fft(np.real(ifft(c)) ** 2)
            v = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
            vv[:, i] = v
        uu = np.real(vv)
        return uu

    def KS_forecast(self, u):
        v = fft(u)
        Nv = self.g * fft(np.real(ifft(v)) ** 2)
        a = self.E2 * v + self.Q * Nv
        Na = self.g * fft(np.real(ifft(a)) ** 2)
        b = self.E2 * v + self.Q * Na
        Nb = self.g * fft(np.real(ifft(b)) ** 2)
        c = self.E2 * a + self.Q * (2 * Nb-Nv)
        Nc = self.g * fft(np.real(ifft(c)) ** 2)
        v = self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2 + Nc * self.f3
        u = np.real(ifft(v))
        return u

    def precompute_KS(self):
        k = np.concatenate((np.arange(0, self.N/2),
                            np.array([0]),
                            np.arange(-self.N / 2 + 1, 0))).T * 2 * np.pi / self.d  # wave number
        L = (1 + self.const) * k ** 2 - k ** 4  # fourier multiplier
        E = np.exp(self.tau * L)
        E2 = np.exp(self.tau * L / 2)
        M = 16  # number of points for complex means
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  #roots of unity
        LR = self.tau * np.tile(L, (M, 1)).T + np.tile(r, (self.N, 1))
        Q = self.tau * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        f1 = self.tau*np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        f2 = self.tau*np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        f3 = self.tau*np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
        g = -0.5j * k
        E = E.reshape(1, -1)
        E2 = E2.reshape(1, -1)
        Q = Q.reshape(1, -1)
        f1 = f1.reshape(1, -1)
        f2 = f2.reshape(1, -1)
        f3 = f3.reshape(1, -1)
        g = g.reshape(1, -1)
        return E, E2, Q, f1, f2, f3, g

    def generate_KS_data(self):
        np.random.seed(3) # seed 3 for data
        x = 10 * (-1 + 2 * np.random.rand(1, self.N))
        data = []
        for i in range(self.n_steps):
            x = self.KS_forecast(x)
            data.append(x)

        # discarding transient data
        truncated_data = np.array(data)[self.transient:].reshape([self.n_steps - self.transient, -1, 1])
        return truncated_data

KSsim = KSSimulation()
KS_64 = KSsim.generate_KS_data()
make_color_map(KS_64, figure_size=(15, 3), title="Kuramoto-Sivashinsky Equation 64 Grids")
plt.show()