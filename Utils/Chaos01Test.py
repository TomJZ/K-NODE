from scipy.optimize import curve_fit
import numpy as np

def plot_pq(p, q, save=None, figsize=(6, 6), title=''):
    """
    plotting the p-q system driven by a trajectory
    :param p: p
    :param q: q
    :param save: path where the image gets saved
    :param figsize: figure size
    :param title: title of the plot
    """
    mpl.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_title(title)
    ax.set_xlabel("$p_c$", labelpad=2)
    ax.set_ylabel("$q_c$", labelpad=2)

    for i in range(len(p)):
        ax.plot(p[i:i + 10], q[i:i + 10], color=plt.cm.jet(i / len(p) / 1.6), linewidth=2)

    if save is not None:
        plt.savefig(save + '.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0)
        pass
    plt.show()

def compute_pq(traj):
    """
    Computing p and q for a 1D trajectory
    :param traj: 1D trajectory
    :return: numpy arrays of p and q
    """
    c = 0.4
    p_list = []
    q_list = []
    p = traj[0]*np.cos(c)
    q = traj[0]*np.sin(c)
    p_list.append(p)
    q_list.append(q)
    for n in range(len(traj)-1):
        p = p + traj[n] * np.cos((n+1)*c)
        q = q + traj[n] * np.sin((n+1)*c)
        p_list.append(p)
        q_list.append(q)
    return np.array(p_list), np.array(q_list)

def compute_M(p, q):
    """
    Compute the mean square displacement of the 2D p-q system
    :param p: numpy array of p
    :param q: numpy array of q
    :return: numpy array of the mean square displacement
    """
    M_list = []
    n_cut = int(len(p)/10)
    N = len(p)-n_cut
    for n in range(n_cut):
      M = np.mean([(p[j+n] - p[j])**2 + (q[j+n] - q[j])**2 for j in range(N)])
      M_list.append(M)
    print("Size of M:\n", len(M_list))
    return np.array(M_list)

def compute_Kc(pred):
    """
    Computing Kc of the 0-1 test using the first dimension of the state
    :param pred: predicted trajectory of type torch tensor and size [len, bs, dim]
    :return: Kc
    """
    dim = pred.size()[-1]
    data_for_test = pred[:][::5]
    traj = data_for_test.detach().numpy().reshape([-1,dim])[:,0] #only taking the first dimension
    p_traj, q_traj = compute_pq(traj)
    M = compute_M(p_traj, q_traj)
    def test(x, m, c):
      return m*x+c
    ns = np.log(np.arange(1,len(M)+1))
    log_M = np.log(M+1)
    param, param_cov = curve_fit(test, ns, log_M)
    print("Kc is", param[0])
    return param[0]