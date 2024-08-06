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
import matplotlib.animation as animation
from PIL import Image

import os
from os import listdir

# %%
# Parameters
Lx, Lz = 64,1
Nx, Nz = 8192, 128
Ra_M = 1e6
D_0 = 0
D_H = 1/3
M_0 = 0
M_H = -1
N_s2=4/3
Qrad=0.0028

Prandtl = 1
dealias = 3/2

if ( os.path.isfile('./Isentropic_param.py')):
    from MRBC2D_param import *

# %%
expname = 'MRBC_2D_RaM_'+"{:.1e}".format(Ra_M)  +'_Pr_'+"{:.1e}".format(Prandtl) \
    +'_QR_'+"{:.1e}".format(Qrad) \
            +'_DH_'+"{:.1e}".format(D_H) \
            +  '_Lx_'+"{:.1e}".format(Lx) \
            +   '_Nz_'+"{:d}".format(Nz)

folder_dir = "/scratch/zb2113/DedalusData/2D/"+expname+'/analysis'
save_dir= "/home/zb2113/Dedalus-Postanalysis/2D/"+expname

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

file_paths = [os.path.join(folder_dir, file) for file in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith('.h5')]
#sort by the number in the file name
file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
print(file_paths)

# %%
#read coordinates
with h5py.File(folder_dir+'/analysis_s1.h5', mode='r') as file:
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


# Define batch size
batch_size = 10  # Adjust this value as needed

# Calculate number of batches
num_batches = len(file_paths) // batch_size

# Initialize arrays for batch processing
iP_batches = np.zeros((num_batches, Nz, len(Mlist)))
iM_batches = np.zeros((num_batches, Nz, len(Mlist)))
iMass_batches = np.zeros((num_batches, Nz, len(Mlist)))
iCl_batches = np.zeros((num_batches, Nz, len(Mlist)))

# Precompute the bin edges for M
bin_edges = np.linspace(M_H, M_0, num=len(Mlist)+1)

# Process data in batches
for batch in range(num_batches):
    start_idx = batch * batch_size
    end_idx = (batch + 1) * batch_size
    
    iP = np.zeros((Nz, len(Mlist), batch_size))
    iM = np.zeros((Nz, len(Mlist), batch_size))
    iMass = np.zeros((Nz, len(Mlist), batch_size))
    iCl = np.zeros((Nz, len(Mlist), batch_size))
    
    for file_idx, file_path in enumerate(file_paths[start_idx:end_idx]):
        with h5py.File(file_path, mode='r') as file:
            M = file['tasks']['M'][:] 
            uz = file['tasks']['uz'][:]
            C = file['tasks']['C'][:]
            simtime = np.array(file['scales/sim_time'])
        
        for t in range(simtime.shape[0]):
            M_t = M[t, :, :]
            uz_t = uz[t, :, :]
            C_t = C[t, :, :]

            M_indices = np.digitize(M_t, bin_edges) - 1

            for z1 in range(Nz):
                for m1 in range(len(Mlist)):
                    mask = M_indices[:, z1] == m1
                    iP[z1, m1, file_idx] += np.sum(mask)/Msize
                    iM[z1, m1, file_idx] += np.sum(M_t[:, z1] * mask)/Msize
                    iMass[z1, m1, file_idx] += np.sum(uz_t[:, z1] * mask)/Msize
                    iCl[z1, m1, file_idx] += np.sum(C_t[:, z1] * mask)/Msize

    # Average over the batch
    iP_batches[batch] = np.average(iP, axis=2)/Nx
    iM_batches[batch] = np.average(iM, axis=2)/Nx
    iMass_batches[batch] = np.average(iMass, axis=2)/Nx
    iCl_batches[batch] = np.average(iCl, axis=2)/Nx

    
# Plotting time evolution
for batch in range(num_batches):
    tiny = 1e-10
    iClcond = iCl_batches[batch] / (iP_batches[batch] + tiny)
    
    # Calculate isentropic streamfunctions for this batch
    Psi_Mass = np.zeros((Nz, len(Mlist)))
    Psi_M = np.zeros((Nz, len(Mlist)))
    Psi_C = np.zeros((Nz, len(Mlist)))
    Psi_Ccond = np.zeros((Nz, len(Mlist)))
    
    for z1 in range(Nz):
        Psi_Mass[z1, 0] = iMass_batches[batch, z1, 0]
        Psi_M[z1, 0] = iM_batches[batch, z1, 0]
        Psi_C[z1, 0] = iCl_batches[batch, z1, 0]
        Psi_Ccond[z1, 0] = iClcond[z1, 0]
        for m1 in range(1, len(Mlist)):
            Psi_Mass[z1, m1] = Psi_Mass[z1, m1-1] + iMass_batches[batch, z1, m1-1]
            Psi_M[z1, m1] = Psi_M[z1, m1-1] + iM_batches[batch, z1, m1-1]
            Psi_C[z1, m1] = Psi_C[z1, m1-1] + iCl_batches[batch, z1, m1-1]
            Psi_Ccond[z1, m1] = Psi_Ccond[z1, m1-1] + iClcond[z1, m1-1]
        
    def plot_and_save(data, title, filename, log=False):
        os.makedirs(f'{save_dir}/isentropic/{filename}', exist_ok=True)
        plt.figure(figsize=(10, 8))
        if log:
            data = np.log(data)
        plt.contour(M_grid, z_grid, data, colors='k')
        plt.contourf(M_grid, z_grid, data, cmap='RdBu_r')
        plt.colorbar(label=title)
        plt.xlabel('M/(M_0-M_H)')
        plt.ylabel('z')
        plt.title(f'{title} - Batch {batch+1}')
        x_start, x_end = np.min(M_grid), np.max(M_grid)
        y_start, y_end = np.max(z_grid), np.min(z_grid)
        plt.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
        plt.savefig(f'{save_dir}/isentropic/{filename}/{filename}_batch_{batch+1}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    plot_and_save(iClcond, 'Conditional Distribution of Clock Tracer', 'Cond_Clock')
    plot_and_save(iMass_batches[batch], 'Isentropic Mass Flux', 'Isendist_Mass')
    plot_and_save(iCl_batches[batch], 'Isendist_C', 'Isendist_Clock', log=True)
    plot_and_save(Psi_M, 'Psi_M', 'Psi_M')
    plot_and_save(Psi_Mass, 'Psi_Mass', 'Psi_Mass')
    plot_and_save(Psi_Ccond, 'Psi_Ccond', 'Psi_Ccond')
    plot_and_save(Psi_C, 'Psi_C', 'Psi_C')



    
plot_types = [
    ('iClcond', 'Conditional Distribution of Clock Tracer', 'Cond_Clock', False),
    ('iMass_batches[batch]', 'Isentropic Mass Flux', 'Isendist_Mass', False),
    ('iCl_batches[batch]', 'Isendist_C', 'Isendist_Clock', True),
    ('Psi_M', 'Psi_M', 'Psi_M', False),
    ('Psi_Mass', 'Psi_Mass', 'Psi_Mass', False),
    ('Psi_Ccond', 'Psi_Ccond', 'Psi_Ccond', False),
    ('Psi_C', 'Psi_C', 'Psi_C', False)
]
def create_animation_from_plots(filename, task_folder):
    # Get all png files in the task folder
    png_files = [f for f in os.listdir(f'{save_dir}/isentropic/{task_folder}') if f.endswith('.png')]
    
    # Sort the files based on the batch number
    png_files.sort(key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))
    
    # Read all images
    images = []
    for png_file in png_files:
        img = Image.open(f'{save_dir}/isentropic/{task_folder}/{png_file}')
        images.append(img)
    
    # Save as GIF
    images[0].save(f'{save_dir}/isentropic/{task_folder}/{filename}.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=500,  # Duration for each frame in milliseconds
                   loop=0)  # 0 means loop indefinitely

    print(f"Animation for {filename} created successfully.")


for plot_type, title, filename, log in plot_types:
    # Create task-specific folder
    task_folder = filename
    os.makedirs(f'{save_dir}/isentropic/{task_folder}', exist_ok=True)
    create_animation_from_plots(filename, task_folder)


# def create_animation(plot_type, title, filename, log=False):
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     tiny = 1e-10
#     iClcond = iCl_avg / (iP_avg + tiny)
    
#     # Calculate isentropic streamfunctions
#     Psi_Mass = np.zeros((Nz, len(Mlist)))
#     Psi_M = np.zeros((Nz, len(Mlist)))
#     Psi_C = np.zeros((Nz, len(Mlist)))
#     Psi_Ccond = np.zeros((Nz, len(Mlist)))
    
#     for z1 in range(Nz):
#         Psi_Mass[z1, 0] = iMass_avg[z1, 0]
#         Psi_M[z1, 0] = iM_avg[z1, 0]
#         Psi_C[z1, 0] = iCl_avg[z1, 0]
#         Psi_Ccond[z1, 0] = iClcond[z1, 0]
#         for m1 in range(1, len(Mlist)):
#             Psi_Mass[z1, m1] = Psi_Mass[z1, m1-1] + iMass_avg[z1, m1-1]
#             Psi_M[z1, m1] = Psi_M[z1, m1-1] + iM_avg[z1, m1-1]
#             Psi_C[z1, m1] = Psi_C[z1, m1-1] + iCl_avg[z1, m1-1]
#             Psi_Ccond[z1, m1] = Psi_Ccond[z1, m1-1] + iClcond[z1, m1-1]
    
#     data = eval(plot_type)
#     if log:
#         data = np.log(data)
    
#     # Create contour plot and colorbar only once
#     cont = ax.contourf(M_grid, z_grid, data, cmap='RdBu_r')
#     fig.colorbar(cont, ax=ax, label=title)
    
#     def animate(frame):
#         ax.clear()
        
#         # Recreate the plot without making a new colorbar
#         ax.contour(M_grid, z_grid, data, colors='k')
#         ax.contourf(M_grid, z_grid, data, cmap='RdBu_r')
#         ax.set_xlabel('M/(M_0-M_H)')
#         ax.set_ylabel('z')
#         ax.set_title(f'{title} - Frame {frame+1}')
#         x_start, x_end = np.min(M_grid), np.max(M_grid)
#         y_start, y_end = np.max(z_grid), np.min(z_grid)
#         ax.plot([x_start, x_end], [y_start, y_end], color='white', linestyle='--', linewidth=2)
    
#     anim = animation.FuncAnimation(fig, animate, frames=20, repeat=True)  # 20 frames for animation
    
#     # Save as GIF
#     anim.save(f'{save_dir}/isentropic/{filename}.gif', writer='pillow', fps=2)
#     plt.close(fig)

# for plot_type, title, filename, log in plot_types:
#     try:
#         plot_and_save()
#         create_animation(plot_type, title, filename, log)
#         print(f"Animation for {filename} created successfully.")
#     except Exception as e:
#         print(f"Error creating animation for {filename}: {str(e)}")

# print("Animation process completed.")