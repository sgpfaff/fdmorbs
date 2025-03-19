import sys
import os
import numpy as np
from ..potential.fdm import FDM
import re
def pot_sim_timestep(axion_mass, halo_mass, root='..'):
    '''
    Parameters
    ----------
    axion_mass : str
        axion mass associated with simulation file
    halo_mass : str
        halo mass associated with simulation file
    root : str
        path to root directory from cwd
    
    Returns
    -------
    float
        total time of the simulation in Gyr
    '''
    lines = get_pot_evolving_file_lines(axion_mass, halo_mass, root)
    #print(lines[np.arange(4, len(lines))[0]])

    times = np.array([float(lines[i][:-15]) for i in np.arange(4, len(lines))])
    #print(f'times : {times}')

    return times[1]

def potential_final_time(axion_mass, halo_mass, root=".."):
    '''
    Checks the length of the simulation associated with FDM halo simulation with 
    axion mass of axion_mass and a halo mass of halo_mass. 
    
    Parameters
    ----------
    axion_mass : str
        axion mass associated with simulation file
    halo_mass : str
        halo mass associated with simulation file
    root : str
        path to root directory from cwd
    
    Returns
    -------
    float
        total time of the simulation in Gyr
    '''
    lines = get_pot_evolving_file_lines(axion_mass, halo_mass, root)
    #print(lines[np.arange(4, len(lines))[0]])

    times = np.array([float(lines[i][:-15]) for i in np.arange(4, len(lines))])
    #print(f'times : {times}')

    return times[-1]

def potential_n_snapshots(axion_mass, halo_mass, root=".."):
    lines = get_pot_evolving_file_lines(axion_mass, halo_mass, "..")
    n_snap = int(np.max(np.array([int(lines[p].split()[-1][-8:-4]) for p in range(4, len(lines))])))
    return n_snap
    

def get_pot_evolving_file_lines(axion_mass, halo_mass, root):
    '''
    Reads the pot_evolving.ini file associated with FDM halo simulation with
    axion mass of axion_mass and a halo mass of halo_mass.
    
    Parameters
    ----------
    axion_mass : str
        axion mass associated with simulation file
    halo_mass : str
        halo mass associated with simulation file
    root : str
        path to root directory from cwd
    
    Returns
    -------
    list
        list of lines in the pot_evolving.ini file
    '''
    filepath = f'{root}/simulations/{axion_mass}/{halo_mass}/pot_evolving.ini'
    # Read the original file
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return lines



def loop_potential(axion_mass, halo_mass, target_time, root=".."):
    '''
    Edits pot_evolving.ini file associated with FDM halo simulation with 
    axion mass of axion_mass and a halo mass of halo_mass. Loops over 
    potential files such that 

    Parameters
    ----------
    axion_mass : str
        axion mass associated with simulation file
    halo_mass : str
        halo mass associated with simulation file
    target_time : float
        desired total time for the simulation in Gyr
    period : float
        period of the core oscillation in Gyr
    root : str 
        path to root directory from cwd
    
    Returns
    -------
    None
        if the simulation already has enough timesteps
    '''

    def get_period(lines):
        '''
        Calculate the period of the core oscillation

        Parameters
        ----------
        lines : list
            list of lines in the pot_evolving.ini file

        Returns
        -------
        float
            period of the core oscillation in Gyr
        '''
        n_snap = int(lines[-1].split()[-1][-8:-4])
        tmax = float(lines[-1].split()[0])
        potential = FDM(axion_mass, halo_mass, n_snapshots=n_snap, tmax=tmax, has_dens=False, freq_calc_value='potential', pathfile="pot_000")
        return 2 * np.pi / potential.core_freq

    lines = get_pot_evolving_file_lines(axion_mass, halo_mass, root)
    #print(lines[np.arange(4, len(lines))[0]])

    times = np.array([float(lines[i][:-15]) for i in np.arange(4, len(lines))])
    #print(f'times : {times}')

    final_time = times[-1]
    if final_time >= target_time:
        print(f'Evolving potential for FDM halo with axion mass \033[1m{re.match(r"^\d+(\.\d+)?([eE][+-]?\d+)?", axion_mass).group(0)}\033[0m eV '
              f'and halo mass \033[1m{re.match(r"^\d+(\.\d+)?([eE][+-]?\d+)?", halo_mass).group(0)}\033[0m solar masses \033[1malready exceeds {target_time} Gyr.\033[0m'
              )
        return None
        #raise ValueError('Already enough timesteps')
    
    period = get_period(lines)

    left_off = final_time % period # where in the period of oscillation the potential leaves off at
    #print(f'left off {left_off}')

    start_index = np.argmin(np.abs(times - left_off)) + 1 # index of the closest time to where it leaves off in the potential
    n_passes = final_time/period # number of loops already done
    int_n_passes = int(n_passes) if n_passes != int(n_passes) else int(n_passes)
    #print(start_index, int_n_passes)

    index_wrap_around = np.argmin(np.abs(times - period)) - 1 # nearest index before full period. We will wrap around this index
    #print(f'one period at index {index_wrap_around}, which cooresponds to time {times[index_wrap_around]}')
    
    
    passes_needed = int(target_time/period - n_passes) + 2 # how many more passes we need
    #print(f'{passes_needed} more passes needed, giving a total number of passes of {int_n_passes + passes_needed} and a total time of {(int_n_passes + passes_needed) * period}')

    #print('times to append:')
    current_index = start_index.copy()
    multiply_factor = int_n_passes
    new_timestamps_list = []
    
    while multiply_factor <= passes_needed + n_passes:
        if current_index >= index_wrap_around:
            current_index = 0
            multiply_factor += 1
        new_timestep = str(times[current_index] + period * multiply_factor) + " " + lines[current_index+4][-15:]
        #print(new_timestep)
        new_timestamps_list.append(new_timestep)
        current_index += 1
        
    # Write back the updated file
    filepath = f'{root}/simulations/{axion_mass}/{halo_mass}/pot_evolving.ini'
    with open(filepath, 'a') as file:
        file.writelines(new_timestamps_list)
    print("File updated successfully.")



def loop_potentials(axion_mass_list, halo_mass_list, target_time, root='..'):
    '''
    UPDATE: This method is no longer all that necessary since the potentials are now checked
    and looped when necessary in the simulation run method.

    Edits pot_evolving.ini file for potentials associated with 
    axion_mass_list and halo_mass_list to have a total time of target_time. 

    
    Parameters
    ----------
    axion_mass_list : list
        list of axion masses associated with simulation files
    halo_mass_list : list
        list of halo masses asociated with simulation files
    target_time : float 
        desired total time for the simulation in Gyr
    root : str 
        path to root directory from cwd

    Returns
    -------

    '''
    for axion_mass in axion_mass_list:
        for halo_mass in halo_mass_list:
            lines = get_pot_evolving_file_lines(axion_mass, halo_mass, root)
            times = np.array([float(lines[i][:-15]) for i in np.arange(4, len(lines))])
            if times[-1] >= target_time:
                print(f'Evolving potential for FDM halo with axion mass \033[1m{re.match(r"^\d+(\.\d+)?([eE][+-]?\d+)?", axion_mass).group(0)} eV\033[0m'
                      f'and halo mass \033[1m{re.match(r"^\d+(\.\d+)?([eE][+-]?\d+)?", halo_mass).group(0)}\033[0m solar masses \033[1malready exceeds {target_time} Gyr.\033[0m'
                      )
                return None
            else:
                loop_potential(axion_mass, halo_mass, target_time, root)



