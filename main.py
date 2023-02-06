import torch
from training_loop import sample_and_grow, sample_data
from Models.lorenz_nn import *
from Utils.Solvers import RK
from Systems.lorenz import Lorenz
from NODE.NODE import NeuralODE

################### Initialize Models ##################
simulation_solver = RK
simulation_step_size = 0.01
training_solver = RK
training_step_size = 0.01
torch.manual_seed(0)
ode_train = NeuralODE(LorenzModifiedKNODE(), training_solver, training_step_size)
ode_true = NeuralODE(Lorenz(), simulation_solver, simulation_step_size)
hybrid = False
loss_arr = []
save_path = None

################### Generating Training Data ###################
sampling_rate = simulation_step_size * 1  # seconds per instance i.e. 1/Hz, assumed to be lower than simulation rate
SimICs = torch.tensor([[[-8., 7., 27.]]])  # initial condition for simulation
t0 = 0  # start point (index of time)
N_POINTS = 800  # Number of times the solver steps. total_time_span = N_POINTS * simulation_step_size
NOISE_VAR = 0  # 0.316227766  # Variance of gaussian noise added to the observation. Assumed to be 0-mean
times, obs_noiseless, t_v, x_v = sample_data(ode_true, t0, N_POINTS, SimICs, simulation_step_size, sampling_rate)
torch.manual_seed(6)
obs = obs_noiseless + torch.randn_like(obs_noiseless) * NOISE_VAR
obs = obs.detach()  # [len, 1, dim]
times = times.detach()

################## Training ###################
# Training Parameters
EPOCHs = 2000  # No. of epochs to train
LOOKAHEAD = 2  # lookahead
name = "lookahead_" + str(LOOKAHEAD - 1)
LR = 0.01  # learning rate
sample_and_grow(ode_train, obs, times, EPOCHs, LR, hybrid, LOOKAHEAD, loss_arr, plot_freq=20)
