from torch.nn import functional as F
from Systems.lorenz import *
from Utils.Plotters import *
from Utils.Chaos01Test import *
import matplotlib.pyplot as plt


def sample_data(func, t_0, n_points, ics, sim_h, sampling_r):
    """
    Samples simulated trajectories to generate training data

    :param func: the function whose dynamics is to be simulated
    :param t_0: time stamp of the initial condition
    :param n_points: the total number of steps to simulate
    :param ics: initial conditions
    :param sim_h: simulation step size
    :param sampling_r: sampling rate
    :return: sampled time stamps (times) and trajectory (obs)
    :return: simulated time stamps (t) and trajectory (x)
    """
    t = torch.from_numpy(np.arange(t_0, t_0 + n_points, 1)).to(ics)
    x = func(ics, t, return_whole_sequence=True)
    ratio = int(sampling_r / sim_h)
    obs = x[0::ratio]
    times = t[0::ratio]
    return times, obs.squeeze(1), t, x.squeeze(1)


def sample_and_grow(ode_train, true_sampled_traj, true_sampled_times, epochs,
                    lr, hybrid, lookahead, loss_arr, plot_freq=50, save_path=None):
    """
    The main training loop

    :param ode_train: the ode to be trained
    :param true_sampled_traj: sampled observations (training data)
    :param true_sampled_times: sampled time stamps
    :param epochs: the total number of epochs to train
    :param lookahead: lookahead
    :param loss_arr: array where the training losses are stored
    :param plot_freq: frequency of which the trajectories are plotted
    :return: None
    """
    plot_title = "Epoch: {0} Loss: {1:.3e} Sim Step: {2} \n No. of Points: {3} Lookahead: {4} LR: {5}"
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ode_train.parameters()), lr=lr)
    n_segments = len(true_sampled_traj)
    fig = plt.figure()

    for i in range(epochs):
        # Train Neural ODE
        true_segments_list = []
        all_init = []
        for j in range(0, n_segments - lookahead + 1, 1):
            true_sampled_segment = true_sampled_traj[j:j + lookahead]
            true_segments_list.append(true_sampled_segment)

        all_init = true_sampled_traj[:n_segments-lookahead+1]  # the initial condition for each segment
        true_sampled_time_segment = torch.tensor(np.arange(lookahead))  # the times step to predict

        # predicting
        z_ = ode_train(all_init, true_sampled_time_segment, return_whole_sequence=True)
        z_ = z_.view(-1, 3)
        obs_ = torch.cat(true_segments_list, 1)
        obs_ = obs_.view(-1, 3)

        # computing loss
        loss = F.mse_loss(z_, obs_)
        loss_arr.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if hybrid:
            ode_train.func.lin_im.weight.grad *= 0

        optimizer.step()

        if i % plot_freq == 0:
            # saving model
            if save_path is not None:
                CHECKPOINT_PATH = save_path + "Lorenz_" + name + ".pth"
                torch.save({'ode_train': ode_train, 'ode_true': ode_true, 'loss_arr': loss_arr},
                           CHECKPOINT_PATH)

            # computing trajectory using the current model
            z_p = ode_train(true_sampled_traj[0], true_sampled_times, return_whole_sequence=True)
            # plotting
            plot_trajectories(fig, obs=[true_sampled_traj], noiseless_traj=[true_sampled_traj],
                              times=[true_sampled_times], trajs=[z_p[:int(true_sampled_times[-1])]],
                              save=None, title=plot_title.format(i, loss.item(), ode_train.STEP_SIZE, n_segments,
                                                                 lookahead - 1, lr))
            print(plot_title.format(i, loss.item(), ode_train.STEP_SIZE, n_segments,
                                    lookahead - 1, lr))
