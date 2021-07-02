import matplotlib.pyplot as plt
import numpy as np
import time

def plot_trajectories(fig, obs=None, noiseless_traj=None,times=None, trajs=None, save=None, title=''):
    plt.ion()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    if title is not None:
      ax.set_title('True Trajectory and Predicted Trajectory\n'+title)

    if noiseless_traj is not None:
      z = np.array([o.detach().numpy() for o in noiseless_traj])
      z = np.reshape(z, [-1,3])
      for i in range(len(z)):
        ax.plot(z[i:i+10, 0], z[i:i+10, 1], z[i:i+10, 2], color=plt.cm.jet(i/len(z)/1.6))

    if obs is not None:
      z = np.array([o.detach().numpy() for o in obs])
      z = np.reshape(z, [-1,3])
      ax.scatter(z[:,0], z[:,1], z[:,2], marker='.', color='k', alpha=0.5, linewidths=0, s=45)

    if trajs is not None:
      z = np.array([o.detach().numpy() for o in trajs])
      z = np.reshape(z, [-1,3])
      for i in range(len(z)):
        ax.plot(z[i:i+10, 0], z[i:i+10, 1], z[i:i+10, 2], color='r', alpha=0.3)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)
    plt.show()
    if save is not None:
        plt.savefig(save+'.png', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
        pass


############## VISUALIZING KS ################
def make_color_map(DATA_TO_PLOT, title="Untitled", slice=False, save=None, figure_size=(10,3), save_eps=False):
  nsteps, ngrids,*a = DATA_TO_PLOT.shape
  offset = 0 # number of grids (starting from 0th) not to plot

  # creating mesh
  x = np.linspace(0,nsteps-1, num=nsteps)*0.25*0.089
  y = np.linspace(0,ngrids-offset-1, num=ngrids-offset)
  X, Y = np.meshgrid(x,y)

  # plotting color grid
  fig = plt.figure(figsize=figure_size)
  ax = plt.axes()
  c = ax.pcolormesh(X, Y, DATA_TO_PLOT.T.reshape(ngrids,nsteps)[offset:ngrids,:], cmap="jet")
  ax.set_title(title)
  ax.set_xlabel('$\Lambda_{max} t$') # adding axes labels changes the appearance of the color map
  ax.set_ylabel('space')
  fig.tight_layout()
  fig.colorbar(c)
  if save is not None:
    plt.savefig(save+'.png', format='png', dpi=600, bbox_inches ='tight', pad_inches = 0)
    if save_eps:
      plt.savefig(save+'.eps', format='eps', dpi=600, bbox_inches ='tight', pad_inches = 0)
    pass

  if slice:
    # plotting one single trajectory
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(1, 1, 1)
    #for i in range(12):
    #i = i*10
    ax.plot(DATA_TO_PLOT[0,:])
    ax.plot(DATA_TO_PLOT[1,:])
    ax.set_title(title+'\nSingle Trajectory')
    ax.set_xlabel('$\Lambda_{max} t$')
    ax.set_ylabel('state')
    fig.tight_layout()