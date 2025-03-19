import matplotlib.pyplot as plt
from tqdm import tqdm
import agama as ag
import numpy as np
from ..potential.fdm import FDM
from ..utils.adjust_files import loop_potential, potential_final_time, potential_n_snapshots, pot_sim_timestep
from scipy.interpolate import CubicSpline
import h5py

ag.setUnits(mass=1, length=1, velocity=1)

class Simulation:
    def __init__(self, 
        axion_mass_name, 
        halo_mass_name, 
        pot=None, 
        n_res_freq=120, 
        n_res_t=120, 
        ic_params={'type':'automatic', 'center_around':None, 'res_bounds':[None, None], 'r_bounds':[None, None]}, 
        int_params = {'step':1000, 'timestep':0.01/0.978, 'steps':4001, 'st_t_until':10},
        write_to_file=False):
        '''
        ic_params : dict
            dictionary holding parameters for generating initial conditions. 
            type : str
                method to generate initial conditions. Options are 'automatic', 'manual', and 'around_resonance'. Currently only 'automatic' and 'manaul' is implemented!
            center_around : float
                resonance to center the simulation around. Default is None. (include which frequency is numerator and which is denominator)
            r_bounds : list or array
                bounds for the radius of the simulation. Default is None.
            res_bounds : list or array
                bounds for the resonances of the simulation. Default is None.

        int_params : dict
            dictionary holding parameters for the simulation.
            step : int
                number of radius steps in grid for making generating initial conditions

        '''
        if ic_params['type'] == 'around_resonance':
            raise NotImplementedError('around_resonance initial conditions not implemented yet!')
        

        self.axion_mass_name = axion_mass_name
        self.halo_mass_name = halo_mass_name
        self.step = int_params['step'] # number of radius steps in grid for making generating initial conditions
        self.steps = int_params['steps']
        self.timestep = int_params['timestep']
        self.st_t_until = int_params['st_t_until']
        not_enough_time = False
        if pot != None:
            # check if there is enough time in the simulation
            if potential_final_time(axion_mass_name, halo_mass_name) >= self.timestep*self.steps:
                self.pot = pot
            else:
                not_enough_time = True
        if pot is None or not_enough_time == True:
            # Loop potential if necessary
            #print(f'Looping potential for {axion_mass_name} and {halo_mass_name}')
            #original_tmax = potential_final_time(axion_mass_name, halo_mass_name)
            self.n_snap = potential_n_snapshots(axion_mass_name, halo_mass_name)
            self.potential_sim_timestep = pot_sim_timestep(axion_mass_name, halo_mass_name)# timestep of FDM simulation
            #print(n_snap*potential_sim_timestep)
            loop_potential(axion_mass_name, halo_mass_name, self.timestep*self.steps)
            self.pot = FDM(axion_mass_name, halo_mass_name, has_dens=False, tmax=self.n_snap*self.potential_sim_timestep, n_snapshots=self.n_snap, freq_calc_value='potential', pathfile="pot_000")
        self.ic_method = ic_params['type']
        self.center_around = ic_params['center_around']

        if self.ic_method == 'automatic': # radii of potential ADD THESE VALUES IN POTENTIAL OBJECT
            self.r_min = 0.001 * 5 #* self.potential_sim_timestep # ISSUE these variables won't be defined if potential is given # make minimum radius 5 steps away from smallest radius in potential
            self.r_max = 20
        else: # user defined bounds
            self.r_min = ic_params['r_bounds'][0] 
            self.r_max = ic_params['r_bounds'][-1]
        self.min_res = ic_params['res_bounds'][0]
        self.max_res = ic_params['res_bounds'][1]
        self.n_res_freq = n_res_freq
        self.n_res_t = n_res_t
        
        self.start_time_list = np.linspace(0, self.st_t_until, self.n_res_t)
        self.time_list = np.linspace(0, self.steps*self.timestep, self.steps)
        self.r_init, self.freq_list, self.r_center, self.orbit_energies = self.generate_IC()
        self.E0 = self.pot.pot_eq.potential([[0,self.r_center,0]])
        self.pos, self.vel, self.H = self.run()
        self.H0 = self.H[0].T.copy()
        self.all_frac_change = ((self.H.T - np.repeat(self.H0[:,:,np.newaxis], self.steps, axis=-1))/self.E0)
        if write_to_file:
            self.write_to_file()
        
    
    def generate_IC(self):
        af = ag.ActionFinder(self.pot.pot_eq, interp=True)
        r_list = np.linspace(self.r_min, self.r_max, self.step)
        def get_freq(r):
            return af([0,r,0,0,0,0], angles=True)[-1][0]
        def get_r(freq):
            spline_light = CubicSpline(r_list, [af([0,r,0,0,0,0], angles=True)[-1][0] - freq for r in r_list])
            r0 = spline_light.roots() # radius closest to given frequency 
            for r in r0:
                if r >= self.r_max/self.step  and r <= self.r_max:
                    return r
            
        if self.ic_method == 'manual':
            # spline_light = CubicSpline(r_list, [af([0,r,0,0,0,0], angles=True)[-1][0] - self.center_around * self.pot.core_freq for r in r_list]) 
            # r0 = spline_light.roots() # radii associated with center res
            # #print(f'r0 at resonance: {r0}')
            # for r in r0:
            #     if r > 0  and r < 10:
            #         r_center = r
            r_center = get_r(self.center_around * self.pot.core_freq)
            freq_scaling = np.linspace(self.min_res, self.max_res, self.n_res_freq)
            freqs = freq_scaling * self.pot.core_freq
            radii = [] # include the first and last ones too though
        if self.ic_method == 'automatic':
            min_freq = af([0,self.r_max,0,0,0,0], angles=True)[-1][0]
            max_freq = af([0,self.r_min,0,0,0,0], angles=True)[-1][0]
            freqs = np.linspace(min_freq, max_freq, self.n_res_freq)
            center_freq = np.mean([min_freq, max_freq])
            r_center = get_r(center_freq)
            radii = [self.r_max]
        for f in tqdm(freqs[1:-1]):
            #spline_light = CubicSpline(r_list, [af([0,r,0,0,0,0], angles=True)[-1][0] - f for r in r_list]) 
            # r0 = spline_light.roots() # radius closest to given frequency 
            # #print(f'min: {self.r_max/self.step}, max: {self.r_max}')
            # for r in r0:
            #     #print(f'freq:{f}, r={r}')
            #     if r > self.r_max/self.step  and r < self.r_max:
            #         radii = np.append(radii, r)
            r0 = get_r(f)
            radii = np.append(radii, r0)
        if self.ic_method == 'automatic':
            radii = np.append(radii, self.r_min)
        orbit_energies = self.pot.pot_eq.potential([[0,r,0] for r in radii])
        return radii, freqs, r_center, orbit_energies
    
    def run(self, plot=False):
        #print(self.r_init.shape, self.n_res_freq)
        initial_r_all = np.repeat(self.r_init, self.n_res_t).reshape((self.n_res_freq, self.n_res_t)).T.flatten()
        #print(initial_r_all)
        full_ICs = np.zeros((self.n_res_t*self.n_res_freq,6))
        full_ICs[:,1] = initial_r_all
        st_time = np.repeat(self.start_time_list, self.n_res_freq)

        orbits_explore = ag.orbit(potential=self.pot.pot_ev, ic=full_ICs, time=self.steps*self.timestep, trajsize=self.steps, timestart=st_time)
        pos = np.zeros((self.n_res_t*self.n_res_freq, self.steps))
        vel = np.zeros((self.n_res_t*self.n_res_freq, self.steps))
        for i,(times, orbit) in enumerate(tqdm(orbits_explore)):
            time = times
            pos[i] = orbit[:,1]
            vel[i] = orbit[:,4]
        
        ys = np.zeros((self.n_res_t*self.n_res_freq*self.steps, 3))
        for i,y in enumerate(tqdm(pos.flatten())):
            ys[i,1] = y

        pot_en = self.pot.pot_eq.potential(ys)
        kin_en = 0.5 * vel.flatten()**2
        H_unshaped = pot_en + kin_en
        H = H_unshaped.reshape(self.n_res_t, self.n_res_freq, len(time)).T
        H0 = H[0].T.copy() # energy for all particles in the grid at time zero

        return pos, vel, H
    
    def frac_change(self, i):
        '''i is time index'''
        return (self.H[i].T - self.H0)/self.E0
    
    def write_to_file(self):
        sim_output_name = f"sim_outputs/sim_{self.axion_mass_name[:-11]}_{self.halo_mass_name}.h5"

        # Simulated self object (with arrays)
        sim_params = vars(self)
        pot_params = vars(self.pot)

        # Write to HDF5 file
        with h5py.File(sim_output_name, "w") as h5file:
            # Save arrays as datasets
            h5file.create_dataset("frac_change", data=np.array(self.all_frac_change))
            h5file.create_dataset("start_time", data=np.array(self.start_time_list))
            h5file.create_dataset("frequencies", data=np.array(self.freq_list))
            h5file.create_dataset("times", data=np.array(self.time_list))

            # Save parameters as attributes or datasets
            for key, value in sim_params.items():
                if isinstance(value, (list, np.ndarray)):
                    # Convert lists to NumPy arrays and save as datasets
                    h5file.create_dataset(f"sim_params/{key}", data=np.array(value))
                elif isinstance(value, (int, float)):
                    # Save simple types as attributes
                    h5file.attrs[key] = value
                elif isinstance(value, str):
                    # Convert strings to byte strings
                    h5file.attrs[key] = np.string_(value)
                else:
                    # Convert unsupported types to strings
                    h5file.attrs[key] = np.string_(str(value))

            for key, value in pot_params.items():
                if isinstance(value, (list, np.ndarray)):
                    # Convert lists to NumPy arrays and save as datasets
                    h5file.create_dataset(f"pot_params/{key}", data=np.array(value))
                elif isinstance(value, (int, float)):
                    # Save simple types as attributes
                    h5file.attrs[key] = value
                elif isinstance(value, str):
                    # Convert strings to byte strings
                    h5file.attrs[key] = np.string_(value)
                else:
                    # Convert unsupported types to strings
                    h5file.attrs[key] = np.string_(str(value))

        print("Data saved successfully.")



    def plot(self, t_index=-1, cbar_rel_to_t_ind=-1):
            '''
            cbar_rel_to_t_ind : int
                index of time at which you want energy of system to set colorbar scale
            '''
            freqs_plot = ((np.repeat(self.freq_list, self.n_res_t)/ self.pot.core_freq)).reshape((self.n_res_freq, self.n_res_t))

            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_subplot(111)
            im = ax.pcolormesh(self.start_time_list, freqs_plot, self.frac_change(t_index).T, cmap='RdBu')
            hm = fig.colorbar(im, ax=ax, pad = 0.05, shrink=0.7)
            im.set_clim(np.mean(self.frac_change(cbar_rel_to_t_ind)[::-1]) - np.ptp(self.frac_change(cbar_rel_to_t_ind)[::-1])/2, np.mean(self.frac_change(cbar_rel_to_t_ind)[::-1]) + np.ptp(self.frac_change(cbar_rel_to_t_ind)[::-1])/2)
            hm.set_label('$(\Delta E / E_{1:1}) \\times 100$',  fontsize=16)
            ax.set_xlabel('$t_{start}$ [Gyr]', fontsize=16)
            ax.set_ylabel('$\\Omega_{orbit}/\\Omega_{core}$',  fontsize=16)
            ax.set_title('Lightly Perturbed Soliton', fontsize=18)
            ax.text(0.035, 0.05, f'Simulation Time : {np.round(self.steps*self.timestep, 2)} Gyr', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            plt.show()  