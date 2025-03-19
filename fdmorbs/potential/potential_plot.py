from .fdm import FDM
import matplotlib.pyplot as plt

class PlotPotential():
        '''
        Plotting pipeline for the potential.

        ===========================================================

        Params:

        type (str) : the type of plot you'd like to make.

                    'dens_vs_t' --> time evolution of the density at a given radius, 
                                    r_eval (automatically set to 0.001 kpc)
                    'phi_vs_t'  --> time evolution of the evolving potential (pot_ev) 
                                    at a given radius, r_eval (automatically set to 0.001 kpc)
        
        r_eval (float) : radius at which to evaluate given values for plotting
        '''
        
        def plot(self, potential : FDM, type, r_eval = 0.001, save=False):
            if type == 'dens_vs_t': 
                if potential.has_dens == True:
                    __, ax = plt.subplots(figsize=(15,5))
                    if r_eval == 0.001:
                        rho_r = potential.d
                    else:
                        rho_r = potential.dens_at_r(r_eval)
                    ax.plot(potential.timesteps, rho_r, 'k', label=potential.name, alpha=0.7, linewidth=3)
                    ax.set_xlabel('t [Gyr]', fontsize=16)
                    ax.set_ylabel(f'$\\rho(r = {potential.r_eval} kpc)$ [M$_\odot$/kpc$^3$]', fontsize=16)
                    ax.set_title(f'{potential.name}', fontsize=20)
                    if save == True:
                        plt.savefig(f'{potential.name}_dens_vs_t.png')
                    plt.show()
                else:
                    raise(AttributeError('No Density Function Defined'))
            elif type == 'phi_vs_t':
                __, ax = plt.subplots(figsize=(15,5))
                ax.plot(potential.timesteps, potential.pot_at_r(r_eval), 'k', label=potential.name, alpha=0.7, linewidth=3)
                ax.set_xlabel('t [Gyr]', fontsize=16)
                ax.set_ylabel(f'$\\Phi(r = {potential.r_eval} kpc)$', fontsize=16)
                ax.set_title(f'{potential.name}', fontsize=20)
                if save == True:
                    plt.savefig(f'{potential.name}_phi_vs_t.png')
                plt.show()
            elif type == 'fft':
                plt.plot(potential.xf, 2.0/potential.N * np.abs(potential.yf[0:potential.N//2]))
                #plt.vlines(potential.xf[potential.local_maxima_indices[potential.i_max]], 0, 8e6, color='k', alpha=0.5, linestyle='--')
                plt.grid()
                plt.show()