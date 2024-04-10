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
Lx, Lz = 20,1
Nx, Nz = 640, 32
Ra_M = 4e5
D_0 = 0
D_H = 1/3
M_0 = 0
M_H = -1
N_s2=4/3
f=0.1

Prandtl = 0.7
dealias = 3/2

# %%
folder_dir = "/scratch/zb2113/DedalusData/4e50.33Q=0.0028/snapshots"
save_dir= "/home/zb2113/Dedalus-Postanalysis/2D/4e50.33Q=0.0028"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

file_paths = [os.path.join(folder_dir, file) for file in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith('.h5')]
#sort by the number in the file name
file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_paths)

# %%
#read coordinates
with h5py.File(folder_dir+'/snapshots_s1.h5', mode='r') as file:
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

file_paths = [os.path.join(folder_dir, file) for file in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith('.h5')]
#sort by the number in the file name
file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_paths)


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

iCl=np.zeros((Nz,len(Mlist),timelist.size))

n=-1
# Precompute the bin edges for M
bin_edges = np.linspace(M_H, M_0, num=len(Mlist)+1)

# Calculation
for file_path in file_paths:
    n=n+1
    with h5py.File(file_path, mode='r') as file:
        M = file['tasks']['M'][:] 
        uz = file['tasks']['uz'][:]
        T = file['tasks']['T'][:]
        C = file['tasks']['C'][:]
        simtime = np.array(file['scales/sim_time'])
        
    for t in range(simtime.shape[0]):
        M_t = M[t, :, :]  # Extract the data for this time step
        uz_t = uz[t, :, :]
        T_t= T[t, :, :]
        C_t = C[t, :, :]

        # Digitize M values into bins
        M_indices = np.digitize(M_t, bin_edges) - 1  # -1 to adjust indices

        for z1 in range(Nz):
            for m1 in range(len(Mlist)):
                mask = M_indices[:, z1] == m1
                iP[z1, m1, t+n*len(simtime)] += np.sum(mask)/Msize
                iM[z1, m1, t+n*len(simtime)] += np.sum(M_t[:, z1] * mask)/Msize
                iMass[z1, m1, t+n*len(simtime)] += np.sum(uz_t[:, z1] * mask)/Msize
                iCl[z1, m1, t+n*len(simtime)] += np.sum(C_t[:, z1] * mask)/Msize


#time-average
tiny = 1e-10
iP=np.average(iP,axis=2)/Nx
iM=np.average(iM,axis=2)/Nx
iMass=np.average(iMass,axis=2)/Nx
iCl=np.average(iCl,axis=2)/Nx
iClcond=iCl/(iP + tiny)

#Plotting (Incomplete)
logiClcond=np.log(iClcond)
plt.contour(M_grid, z_grid, iClcond, colors='k')
plt.contourf(M_grid, z_grid, iClcond, cmap='RdBu_r')
plt.colorbar(label='Conditional Distribution of Clock Tracer')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    
plt.title('Conditional neab of clock tracer')
plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Cond Clock.png', dpi=200, bbox_inches='tight')
plt.close()

#Plotting (Incomplete)
#logiMav=np.log(iMass)
plt.contour(M_grid, z_grid, iMav, colors='k')
plt.contourf(M_grid, z_grid, iMav, cmap='RdBu_r')
plt.colorbar(label='Isendist')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    
plt.title('isentropic mass flux')
plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Isendist_Mass.png', dpi=200, bbox_inches='tight')
plt.close()

# #Plotting (Incomplete)
# logiTav=np.log(iTrac)
# plt.contour(M_grid, z_grid, logiTav, colors='k')
# plt.contourf(M_grid, z_grid, logiTav, cmap='RdBu_r')
# plt.colorbar(label='Isendist_T')
# plt.xlabel('M/(M_0-M_H)')
# plt.ylabel('z')
# x_start = np.min(M_grid)  
# x_end = np.max(M_grid)   
# y_start = np.max(z_grid)  
# y_end = np.min(z_grid)    

# plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
# plt.savefig(save_dir+'/isentropic/Isendist_Trac.png', dpi=200, bbox_inches='tight')
# plt.close()

#Plotting (Incomplete)
logiCav=np.log(iCl)
plt.contour(M_grid, z_grid, logiCav, colors='k')
plt.contourf(M_grid, z_grid, logiCav, cmap='RdBu_r')
plt.colorbar(label='Isendist_C')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    

plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Isendist_Clock.png', dpi=200, bbox_inches='tight')
plt.close()

#Isentropic Stream Function
Psi_M=np.zeros((Nz,len(Mlist)))
Psi_Mass=np.zeros((Nz,len(Mlist)))
Psi_C=np.zeros((Nz,len(Mlist)))
Psi_Ccond=np.zeros((Nz,len(Mlist)))

for z1 in range(0,Nz):
    for m1 in range(0,len(Mlist)):
        Psi_M[z1,m1+1] = Psi_M[z1,m1] + iM[z1,m1]
        Psi_Mass[z1,m1+1] = Psi_Mass[z1,m1] + iMass[z1,m1]

#        for madd in range(0,m1):
#            Psi_M[z1,m1]+=iM[z1,madd]
#            Psi_Mass[z1,m1]+=iMass[z1,madd]
#            Psi_C[z1,m1]+=iCl[z1,madd]
#            Psi_Ccond[z1,m1]+=iClcond[z1,madd]

                
#Plotting (Incomplete)
plt.contour(M_grid, z_grid, Psi_M,colors='k')
plt.contourf(M_grid, z_grid, Psi_M,cmap='RdBu_r')
plt.colorbar(label='Psi_M')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    

plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Psi_M.png', dpi=200, bbox_inches='tight')
plt.close()

#Plotting
plt.contour(M_grid, z_grid, Psi_Mass,colors='k')
plt.contourf(M_grid, z_grid, Psi_Mass, cmap='RdBu_r')
plt.colorbar(label='Psi_Mass')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    

plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Psi_Mass.png', dpi=200, bbox_inches='tight')
plt.close()


#Plotting
plt.contour(M_grid, z_grid, Psi_Ccond,colors='k')
plt.contourf(M_grid, z_grid, Psi_Ccond, cmap='RdBu_r')
plt.colorbar(label='Psi_Ccond')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    

plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Psi_Ccond.png', dpi=200, bbox_inches='tight')
plt.close()

# #Plotting
# plt.contour(M_grid, z_grid, Psi_T,colors='k')
# plt.contourf(M_grid, z_grid, Psi_T, cmap='RdBu_r')
# plt.colorbar(label='Psi_T')
# plt.xlabel('M/(M_0-M_H)')
# plt.ylabel('z')
# x_start = np.min(M_grid)  
# x_end = np.max(M_grid)   
# y_start = np.max(z_grid)  
# y_end = np.min(z_grid)    

# plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
# plt.savefig(save_dir+'/isentropic/Psi_T.png', dpi=200, bbox_inches='tight')
# plt.close()


#Plotting
plt.contour(M_grid, z_grid, Psi_C,colors='k')
plt.contourf(M_grid, z_grid, Psi_C, cmap='RdBu_r')
plt.colorbar(label='Psi_C')
plt.xlabel('M/(M_0-M_H)')
plt.ylabel('z')
x_start = np.min(M_grid)  
x_end = np.max(M_grid)   
y_start = np.max(z_grid)  
y_end = np.min(z_grid)    

plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
plt.savefig(save_dir+'/isentropic/Psi_C.png', dpi=200, bbox_inches='tight')
plt.close()