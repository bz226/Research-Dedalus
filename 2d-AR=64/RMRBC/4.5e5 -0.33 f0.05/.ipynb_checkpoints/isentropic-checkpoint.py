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
file_paths=file_paths[3:]

if not os.path.exists('isentropic'):
    os.mkdir('isentropic')

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

#Isentropic functions define
iP=np.zeros((Nz,len(Mlist),timelist.size))
iM=np.zeros((Nz,len(Mlist),timelist.size))

n=-1
#Calcultion
for file in file_paths:
    n=n+1
    print(n)
    with h5py.File(file, mode='r') as file:
        M = file['tasks']['M']
        st = file['scales/sim_time']
        simtime=np.array(st)
        for t in range(0, len(simtime)):
            for z1 in range(0,Nz):
                for m1 in range(0,len(Mlist)):
                    for x1 in range(0,Nx):
                        if Mlist[m1]-Msize/2<=M[t,x1,z1]<=Mlist[m1]+Msize/2:
                            iP[z1,m1,t+n*len(simtime)]+=1/Msize
                            iM[z1,m1,t+n*len(simtime)]+=M[t,x1,z1]/Msize
#time-average
iPav=np.average(iP,axis=2)
iMav=np.average(iM,axis=2)

#Plotting (Incomplete)
logiPav=np.log(iPav)
plt.contourf(logiPav,cmap='RdBu_r')
plt.colorbar(label='Isendist')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
plt.savefig(f'isentropic/Isendist.png', dpi=200, bbox_inches='tight')
plt.close()

#Isentropic Stream Function
Psi_M=np.zeros((Nz,len(Mlist),timelist.size))
for t in range(0,len(timelist)):
    for z1 in range(0,Nz):
        for m1 in range(0,len(Mlist)):
            for madd in range(0,m1):
                Psi_M[z1,m1,t]+=iM[z1,madd,t]

#time-average
Psi_Mav=np.average(Psi_M,axis=2)
                
#Plotting (Incomplete)
plt.contourf(Psi_Mav,cmap='RdBu_r')
plt.colorbar(label='Psi_M')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
plt.savefig(f'isentropic/Psi_M.png', dpi=200, bbox_inches='tight')
plt.close()

