from .base import AbstractBasePotential
from tqdm import tqdm
import agama as ag
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema
import os
import matplotlib.pyplot as plt

ag.setUnits(mass=1, length=1, velocity=1)

class FDM(AbstractBasePotential):
    '''
    Class for FDM Potential, subclass of Potential
    '''
    def __init__(self, axion_mass_name, halo_mass_name, n_snapshots=1000, tmax=40., r_eval=0.001, has_dens = True, freq_calc_value = 'density', pathfile="sph_harm_pot"):
        '''
        FDM Potential for a given axion mass and halo mass

        Parameters:
        -----------
        axion_mass_name : str
            axion mass associated with simulation file
        halo_mass_name : str
            halo mass associated with simulation file
        n_snapshots : int
            number of snapshots in the simulation
        tmax : float
            maximum time in the simulation
        r_eval : float
            radius at which to evaluate the potential or density for plotting, etc
        has_dens : bool
            whether or not the simulation has density files
        freq_calc_value: str
            value (potential or density) to use for fourier transform to calculate core oscillation frequency
        pathfile : str
            path to the simulation files. Default is sph_harm_pot for density and pot for potential.

        '''
        self.r_setup = np.logspace(-2, 2, 100)
        self.points = np.zeros((100,3))
        self.points[:,0] = self.r_setup

        self.axion_mass_name = axion_mass_name
        self.halo_mass_name = halo_mass_name
        self.name = axion_mass_name + '/' + halo_mass_name

        self.has_dens = has_dens
        self.n_snapshots = n_snapshots
        self.r_eval = r_eval
        self.pathfile = pathfile

        # Get the directory containing the current script file
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.has_dens == True:
            file_path = os.path.join(self.current_dir, "..", "..", "simulations", self.name, "sph_harm_base.ini")
            file_path = os.path.abspath(file_path)

            self.dens_eq = ag.Density(file_path)
            self.pot_eq = ag.Potential(type="Multipole", density=self.dens_eq, mmax=0)
        else:
            self.dens_eq = None
            file_path = os.path.join(self.current_dir, "..", "..", "simulations", self.name, "pot_base.ini")
            file_path = os.path.abspath(file_path)
            self.pot_eq =  ag.Potential(file_path)

        file_path = os.path.join(self.current_dir, "..", "..", "simulations", self.name, "pot_evolving.ini")
        file_path = os.path.abspath(file_path)
        self.pot_ev = ag.Potential(file_path)
        super().__init__(n_snapshots, tmax, r_eval)
        
        # get core oscillation frequency
        if freq_calc_value == 'density':
            if has_dens == False :
                raise(AttributeError("No density defined yet frequency calculation value (freq_calc_value) is chosen to be density. Consider using the potential instead."))
            else:
                self.d = self.dens_at_r(r_eval)
                self.N = len(self.d)
                self.T = self.timesteps[1] - self.timesteps[0]
                self.yf = fft(self.d)
                self.xf = fftfreq(self.N, self.T)[:self.N//2]

                self.local_maxima_indices = argrelextrema(np.abs(self.yf), np.greater)[0]
                self.i_max = np.argmax(np.abs(self.yf[self.local_maxima_indices]))
                self.core_freq = 2 * np.pi * self.xf[self.local_maxima_indices[self.i_max]]
        elif freq_calc_value == 'potential':
            self.p = self.pot_at_r(r_eval)
            self.N = len(self.p)
            self.T = self.timesteps[1] - self.timesteps[0]
            self.yf = fft(self.p)
            self.xf = fftfreq(self.N, self.T)[:self.N//2]

            self.local_maxima_indices = argrelextrema(np.abs(self.yf), np.greater)[0]
            self.i_max = np.argmax(np.abs(self.yf[self.local_maxima_indices]))
            self.core_freq = 2 * np.pi * self.xf[self.local_maxima_indices[self.i_max]]
            #print('Warning: Density is exluded. This will limit available features. Orbit integration will still be possible, but it may be difficult to calculate the frequency of the core oscillation.')
        else:
            raise(AttributeError("Please chose either density or potential as frequency calculation value"))

    def pot_at_r(self, r):
        '''
        Gives potential evolution at a given radius, r
        
        ==========================================================

        Params:

        r (float)      :   radius at which to evaluate

        t_ls (array)   :   

        '''

        phi_list = np.zeros(self.n_snapshots)
        for ti in tqdm(range(self.n_snapshots)):
            if self.pathfile == "pot_000":
                if ti <= 999:
                    pathfile = self.pathfile
                elif ti > 999 and ti <= 9999:
                    pathfile = "pot_00"
            else:
                pathfile = self.pathfile

            file_path = os.path.join(self.current_dir, "..", "..", "simulations", self.name, f"{pathfile}{str(ti).zfill(3)}.ini")
            file_path = os.path.abspath(file_path)
            phi_t = ag.Potential(file_path)
            phi_list[ti] = phi_t.potential([r,0,0])
        return phi_list


    def dens_at_r(self, r):
        '''
        Gives density evolution at a given radius, r
        
        ==========================================================

        Params:

        r (float)      :   radius at which to evaluate

        t_ls (array)   :   

        '''
        if self.has_dens != True:
            raise(AttributeError('No Density Defined!'))
        dens_list = np.zeros(self.n_snapshots)
        for ti in tqdm(range(self.n_snapshots)):
            file_path = os.path.join(self.current_dir, "..", "..", "simulations", self.name, f"sph_harm{str(ti).zfill(3)}.ini")
            file_path = os.path.abspath(file_path)
            d_t = ag.Density(file_path)
            dens_list[ti] = d_t.density([r,0,0])
        return dens_list
        
    def plot(self, type, r_eval = 0.001, save=False):
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

        if type == 'dens_vs_t': 
            if self.has_dens == True:
                __, ax = plt.subplots(figsize=(15,5))
                if r_eval == 0.001:
                    rho_r = self.d
                else:
                    rho_r = self.dens_at_r(r_eval)
                ax.plot(self.timesteps, rho_r, 'k', label=self.name, alpha=0.7, linewidth=3)
                ax.set_xlabel('t [Gyr]', fontsize=16)
                ax.set_ylabel(f'$\\rho(r = {self.r_eval} kpc)$ [M$_\odot$/kpc$^3$]', fontsize=16)
                ax.set_title(f'{self.name}', fontsize=20)
                if save == True:
                    plt.savefig(f'{self.name}_dens_vs_t.png')
                plt.show()
            else:
                raise(AttributeError('No Density Function Defined'))
        elif type == 'phi_vs_t':
            __, ax = plt.subplots(figsize=(15,5))
            ax.plot(self.timesteps, self.pot_at_r(r_eval), 'k', label=self.name, alpha=0.7, linewidth=3)
            ax.set_xlabel('t [Gyr]', fontsize=16)
            ax.set_ylabel(f'$\\Phi(r = {self.r_eval} kpc)$', fontsize=16)
            ax.set_title(f'{self.name}', fontsize=20)
            if save == True:
                plt.savefig(f'{self.name}_phi_vs_t.png')
            plt.show()
        elif type == 'fft':
            plt.plot(self.xf, 2.0/self.N * np.abs(self.yf[0:self.N//2]))
            #plt.vlines(self.xf[self.local_maxima_indices[self.i_max]], 0, 8e6, color='k', alpha=0.5, linestyle='--')
            plt.grid()
            plt.show()