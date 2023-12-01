# %%
import numpy as np
import logging
logger = logging.getLogger(__name__)
import copy
import h5py
import numpy as np
import matplotlib
import re

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

import os
from os import listdir

# %%
# Parameters
Lx, Lz = 4,1
Nx, Nz = 128, 32
Ra_M = 4.5e5
D_0 = 0
D_H = 1/3
M_0 = 0
M_H = -1
N_s2=4/3
f=0.05

Prandtl = 0.7
dealias = 3/2

# %%
folder_dir = "snapshots"
save_dir= "/home/zb2113/Dedalus-Postanalysis/2D/4.5e5 -0.33 f0.05"

file_paths = [os.path.join(folder_dir, file) for file in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith('.h5')]
#sort by the number in the file name
file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_paths)

# %%
#read coordinates
with h5py.File('snapshots/snapshots_s1.h5', mode='r') as file:
    print(list(file.keys()))
    scalekeys=list(file['scales'].keys())
    taskkeys=list(file['tasks'].keys())
    print(scalekeys)
    print(taskkeys)
    #automatic read x y zhash: testing feature
    xhash=scalekeys[-2]
    zhash=scalekeys[-1]
    x=file['scales'][xhash]
    z=file['scales'][zhash]
    x=np.array(x)
    z=np.array(z)

# %%
#Implement Isentropic Analysis
folder_dir = "snapshots"

file_paths = [os.path.join(folder_dir, file) for file in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith('.h5')]
#sort by the number in the file name
file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_paths)
file_paths=file_paths[3600:]

if not os.path.exists(save_dir+'/isentropic'):
    os.mkdir(save_dir+'/isentropic')

#Preparation of bins and lists
Msize=(M_0-M_H)/100
bin=np.arange(M_H,M_0,Msize)
Mlist=bin+Msize/2
print(Mlist)
timelist=[]
for file in file_paths:
    with h5py.File(file, mode='r') as file:
        st = file['scales/sim_time']
        simtime = np.array(st)
        timelist.append(simtime)
timelist=np.array(timelist)

#Grid mesh
M_grid, z_grid=np.meshgrid(Mlist, z)

#Isentropic functions define
iP=np.zeros((Nz,len(Mlist),timelist.size))
iM=np.zeros((Nz,len(Mlist),timelist.size))
iMass=np.zeros((Nz,len(Mlist),timelist.size))

n=-1
#Calcultion
for file in file_paths:
    n=n+1
    with h5py.File(file, mode='r') as file:
        M = file['tasks']['M']
        uz = file['tasks']['uz']
        st = file['scales/sim_time']
        simtime=np.array(st)
        for t in range(0, len(simtime)):
            for z1 in range(0,Nz):
                for m1 in range(0,len(Mlist)):
                    for x1 in range(0,Nx):
                        if Mlist[m1]-Msize/2<=M[t,x1,z1]<=Mlist[m1]+Msize/2:
                            iP[z1,m1,t+n*len(simtime)]+=1/Msize
                            iM[z1,m1,t+n*len(simtime)]+=M[t,x1,z1]/Msize
                            iMass[z1,m1,t+n*len(simtime)]+=uz[t,x1,z1]/Msize
#time-average
iPav=np.average(iP,axis=2)
iMav=np.average(iM,axis=2)

#Plotting (Incomplete)
logiPav=np.log(iPav)
plt.contourf(M_grid, z_grid, logiPav,cmap='RdBu_r', levels=50)
plt.colorbar(label='Isendist')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
plt.savefig(save_dir+'/isentropic/Isendist.png', dpi=200, bbox_inches='tight')
plt.close()

#Isentropic Stream Function
Psi_M=np.zeros((Nz,len(Mlist),timelist.size))
Psi_Mass=np.zeros((Nz,len(Mlist),timelist.size))

for t in range(0,len(timelist)):
    for z1 in range(0,Nz):
        for m1 in range(0,len(Mlist)):
            for madd in range(0,m1):
                Psi_M[z1,m1,t]+=iM[z1,madd,t]
                Psi_Mass[z1,m1,t]+=iMass[z1,madd,t]

#time-average
Psi_Mav=np.average(Psi_M,axis=2)
Psi_Massav=np.average(Psi_Mass,axis=2)
                
#Plotting (Incomplete)
plt.contourf(M_grid, z_grid, Psi_Mav,cmap='RdBu_r', levels=50)
plt.colorbar(label='Psi_M')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
plt.savefig(save_dir+'/isentropic/Psi_M.png', dpi=200, bbox_inches='tight')
plt.close()

#Plotting
plt.contourf(M_grid, z_grid, Psi_Massav,cmap='RdBu_r', levels=100)
plt.colorbar(label='Psi_Mass')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
plt.savefig(save_dir+'/isentropic/Psi_Mass.png', dpi=200, bbox_inches='tight')
plt.close()

