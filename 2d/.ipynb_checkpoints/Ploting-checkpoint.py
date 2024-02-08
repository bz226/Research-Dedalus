import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from os import listdir
import re

Lx, Lz = 20,1
Nx, Nz = 640, 32
Ra_M = 4.5e5
D_0 = 0
D_H = 1/3
M_0 = 0
M_H = -1
N_s2=4/3
f=0.05
stop_sim_time = 1000
Prandtl = 0.7
dealias = 3/2

nu = (Ra_M / (Prandtl*(M_0-M_H)*Lz**3))**(-1/2)

load_dir_1='/scratch/zb2113/New/Research-Dedalus/2d/4.5e5 -0.33 f0/snapshots'
load_dir_2='/scratch/zb2113/New/Research-Dedalus/2d/RMRBC/4.5e5 -0.33 f0.05/snapshots'
load_dir_3='/scratch/zb2113/New/Research-Dedalus/2d/RMRBC/4.5e5 -0.33 f0.1/snapshots'

load_dir=[load_dir_1,load_dir_2,load_dir_3]
label_1='4.5e5 -0.33 f0'
label_2='4.5e5 -0.33 f0.05'
label_3='4.5e5 -0.33 f0.1'
labels=[label_1,label_2,label_3]

save_dir_1='/home/zb2113/Dedalus-Postanalysis/2D/'

plt.xlabel('Normalized time scale')
plt.ylabel('total KE')
xt=np.linspace(0,stop_sim_time,10)
xt= nu*xt/Lz**2
plt.xticks(xt)

for i,dir in load_dir:
    totalKE=[]
    totalst=[]
    for file in dir:
        with h5py.File(file, mode='r') as file:
            KE = file['tasks']['total KE'][:,0,0,0]
            totalKE.append(KE)
            st = file['scales/sim_time'][:]
            totalst.append(st)

    # Flatten and concatenate the lists if they are nested lists
    totalKE = np.concatenate(totalKE)
    totalst = np.concatenate(totalst)

    # Scale 'sim_time'
    totalst = nu * totalst / Lz**2
    plt.plot(totalst, totalKE, label=labels[i])
    plt.legend()

plt.savefig(save_dir_1+'totalKE.png')