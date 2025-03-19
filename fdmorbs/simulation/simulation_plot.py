import numpy as np
import matplotlib.pyplot as plt
from .simulation import Simulation

class SimulationPlot():   
        def plot(self, simulation : Simulation):
            time_list = np.linspace(0, 2, simulation.n_res_t)
            freqs_plot = ((np.repeat(simulation.freq_list, simulation.n_res_t)/ simulation.pot.core_freq)).reshape((simulation.n_res_freq, simulation.n_res_t))

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(time_list, freqs_plot, simulation.frac_change(-1).T, cmap='RdBu')
            hm = fig.colorbar(im, ax=ax, pad = 0.05, shrink=0.7)
            im.set_clim(np.mean(simulation.frac_change(-1)[::-1]) - np.ptp(simulation.frac_change(-1)[::-1])/2, np.mean(simulation.frac_change(-1)[::-1]) + np.ptp(simulation.frac_change(-1)[::-1])/2)
            hm.set_label('$(\Delta E / E_{1:1}) \\times 100$',  fontsize=16)
            ax.set_xlabel('$t_{start}$ [Gyr]', fontsize=16)
            ax.set_ylabel('$\\Omega_{orbit}/\\Omega_{core}$',  fontsize=16)
            ax.set_title('Lightly Perturbed Soliton', fontsize=18)
            ax.text(0.035, 0.05, f'Simulation Time : {np.round(simulation.steps*simulation.timestep, 2)} Gyr', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            plt.show() 

def plot_from_values(
          value, 
          n_res_t, 
          freq_list, 
          core_freq, 
          n_res_freq, 
          times,
          t_index=-1, 
          cbar_rel_to_t_ind=-1,
          xlabel='$t_{start}$ [Gyr]',
          xlabel_size=16,
          ylabel='$\\Omega_{orbit}/\\Omega_{core}$',
          ylabel_size=16,
          title = '',
          title_size=18
          ):
    '''
    value : np.array
        array of values to plot. Should be shape (n_res_freq, n_res_t, timesteps)
    cbar_rel_to_t_ind : int
        index of time at which you want energy of system to set colorbar scale
        '''
    time_list = np.linspace(0, 2, n_res_t)
    freqs_plot = ((np.repeat(freq_list, n_res_t)/ core_freq)).reshape((n_res_freq, n_res_t))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(time_list, freqs_plot, value[:,:,t_index].T, cmap='RdBu')
    hm = fig.colorbar(im, ax=ax, pad = 0.05, shrink=0.7)
    im.set_clim(np.mean(value[:,:,cbar_rel_to_t_ind][::-1]) - np.ptp(value[:,:,cbar_rel_to_t_ind][::-1])/2, np.mean(value[:,:,cbar_rel_to_t_ind][::-1]) + np.ptp(value[:,:,cbar_rel_to_t_ind][::-1])/2)
    hm.set_label('$(\Delta E / E_{1:1}) \\times 100$',  fontsize=16)
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.set_title(title, fontsize=title_size)
    ax.text(0.035, 0.05, f'Simulation Time : {np.round(times[t_index], 2)} Gyr', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.show() 