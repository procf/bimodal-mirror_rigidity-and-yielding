## Extract temperature and pressure (negative stress) from all frames
## of a shear-simulation GSD file

########################
""" MODULE LIBRARY """
########################
import numpy as np
import pandas as pd
import gsd.hoomd
import math
import fortranmod as module
import os
import re
import glob
import sys

##########################
""" INPUT PARAMETERS """
##########################

elif from_path == False:
  # manual simulation parameters
  shear_filepath = '../sim-scripts/bi-colloid_50-50/potential-morse-bysize/12kT/kappa10/shear-const-0.5-allinit-bi-DPD/Shear-const-SR0.5-1strains-vinit0.5-bi-DPD.gsd'
  gel_filepath = '../sim-scripts/bi-colloid_50-50/potential-morse-bysize/12kT/kappa60/Gelation.gsd'
  data_outpath = '../sim-scripts/bi-colloid_50-50/potential-morse-bysize/12kT/kappa10/analysis-bi-colloids/data/shear_0.5-allinit-bi-DPD/'
  shear_style = 'constant' 
  shear_rate_const = 0.5
  Amp = 0.0
  omega = 0.0
  kT = 0.1
  eta0 = 0.3
  frames_per_strain = 1000 
  dt_Integration = 1e-4 
  theta = 1.0
  if shear_style == 'const':
    shear_rate = shear_rate_const
    #vinf = shear_rate * L_Y
    t_ramp = theta/shear_rate
  else:
    omega_timesteps = omega*dt_Integration # frequency (1/N_timesteps)
    #vinf = Amp * omega * L_Y
    shear_rate = Amp * omega
    t_ramp = Amp/shear_rate
  Iter_cycle = int(t_ramp / dt_Integration)
  period= int(Iter_cycle/frames_per_strain)
  MDtime = period*dt_Integration
  colloid1_typeid = 1
  colloid2_typeid = 2


##########################
""" ANALYSIS """
##########################

## Get nstrains from the GSD file
traj = gsd.hoomd.open(shear_filepath, 'r')
nframes = len(traj)
nstrains = (nframes-1) / frames_per_strain

## For the noaffinity calculation, be ready to compare to pre-shear gel
affine_ref_filepath = gel_filepath 

# record simulation parameters
print("\nSIMULATION TIME PARAMETERS")
print(" * shear rate:",shear_rate)
print(" * Iter_cycle (timesteps per strain):", Iter_cycle) 
print(" * period (timesteps per frame):", period)
print(" * MDtime per frame (MDtime):",MDtime) 
print(" * total number of frames:",nframes)
print(" * frames per strain:",frames_per_strain)
print(" * total number of simulated strains:",nstrains,"\n")


# create "data" subfolder if it doesn't exit
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)


## extract thermodynamic properties from GSD for all frames
"""
# get temperature and pressure (AKA negative of stress), including all virial components
"""
# create a file to save the thermodynamic properties
f=open(data_outpath+'/gsd-properties.txt', 'w')
f.write('simframe Virial-Pressure Vr_CONS Vr_DISS Vr_RAND Vr_SQUE Vr_CONT PE kT tps\n')

# for each frame
for i in range(0, nframes-1):
  simframe = i
  # extract the total "force virial contribution" of the pressure tensor
  Pr=float(traj[i].log['md/compute/ThermodynamicQuantities/pressure_tensor'][1])
  # extract the decomposed virial compoenents of the pressure tensor
  Vr=np.asarray(traj[i].log['md/compute/ThermodynamicQuantities/virial_ind_tensor'],dtype=float)
  # extract the potential energy and scale to kilo-units (1/1000)
  Pe=float(traj[i].log['md/compute/ThermodynamicQuantities/potential_energy'][0])*0.001
  # extract the kinetic temperature (kT)
  kT=float(traj[i].log['md/compute/ThermodynamicQuantities/kinetic_temperature'][0])
  # extract the transactios per secont (tps) for assessing speed/efficiency of the simulation
  tps=int(traj[i].log['Simulation/tps'][0])

  # write these values to the output file:
  # raw values
  f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(simframe, Pr, 
    Vr[0], Vr[1], Vr[2], Vr[3], Vr[4], Pe, kT, tps))

  # rounded values
  #f.write('%f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %d %0.2f %d\n'%((i+1)*t1, Pr, 
  #  Vr[0], Vr[1], Vr[2], Vr[3], Vr[4], Pe, kT, tps))
print("Data extracted from GSD file")

