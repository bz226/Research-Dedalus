# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)


def mountain_shape(x):
    return 0.1 * np.exp(-0.5*((x)/6)**2) 

height = mountain_shape(x)


plt.figure()
plt.fill_between(x, height, color='black')
plt.xlim(-10, 10)
plt.ylim(0.0, 1)
plt.axis('off')  
#remove white space at the edge of the figure
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


plt.savefig('mountain.jpg', format='jpg', dpi=300)  


plt.show()


# %%
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import scipy.interpolate as sci

im = image.imread('mountain.jpg')
im=np.array(im)

for i in range(1440):
    for j in range(1920):
        if im[i][j][0] == 0:
            im[i][j]=[255, 255, 255]
        else:
            im[i][j]=[0, 0, 0]
            
plt.imshow(im)

plt.savefig('mountain.jpg', format='jpg', dpi=300)  # 指定分辨率为300 DPI

# %%
cell = np.asarray(im)[:,:,2]

y = 1-np.arange(0,1,1/cell.shape[0])
print(cell.shape)
print(y)
x = np.arange(0,20,20/cell.shape[1])
print(x)

# %%
fig, ax = plt.subplots(figsize=[20,3])
mesh = ax.pcolormesh(x, y, cell, shading='auto', vmin=0, vmax=1)
ax.set_aspect(1)
plt.colorbar(mesh)

# %%
interp = sci.interp2d(x,y,cell)
#interp = sci.RegularGridInterpolator((x,y), cell.T, method='nearest')

# %%
# Coarsen the shape, deliberately
# This lets us use a smooth interpolation to go back to large size but with smooth rather than sharp boundaries for the IBM
# note: this does not need to be a multiple of the original pixel size of the image
print(x.shape, y.shape)
x2 = np.linspace(np.min(x), np.max(x), num=64)
y2 = np.linspace(np.min(y), np.max(y), num=64)
X, Y = np.meshgrid(x2, y2)
Z = interp(x2, y2)

fig, ax = plt.subplots(figsize=[12,12])
ax.pcolormesh(X, Y, Z, shading='auto')
ax.set_aspect(1)

# %%
# interpolate the shape to smooth it; this makes a smooth transition at the edges of the IBM.
# note, this doesn't need to be a multiple of the coarsened grid, but I made it so here for convenience.
interp_smooth = sci.interp2d(x2,y2,Z)
x3 = np.linspace(np.min(x), np.max(x), num=512)
y3 = np.linspace(np.min(y), np.max(y), num=512)
z3 = interp_smooth(x3,y3)
fig, ax = plt.subplots(figsize=[12,12])
ax.pcolormesh(x3, y3, z3, shading='auto',vmax=1, vmin=0)
ax.set_aspect(1)

# %%
import numpy as np
import dedalus.public as d3
import dedalus.extras.plot_tools as dpt
from scipy.interpolate import RegularGridInterpolator
import logging
logger = logging.getLogger(__name__)
import copy
import h5py
import matplotlib
import re

# Parameters
Lx, Lz = 4,1
Nx, Nz = 1024, 256
Ra_M = 1e5
D_0 = 0
D_H = 1/3
M_0 = 0
M_H = -1
N_s2=4/3
Qrad=0.0028

Prandtl = 1
dealias = 3/2
stop_sim_time = 600
timestepper = d3.RK222

dtype = np.float64

# %%
# Bases
coords = d3.CartesianCoordinates('x','z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
D = dist.Field(name='D', bases=(xbasis,zbasis))
M = dist.Field(name='M', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
Z = dist.Field(name='Z', bases=zbasis)
T = dist.Field(name='T', bases=(xbasis,zbasis))
C = dist.Field(name='C', bases=(xbasis,zbasis))

tau_p = dist.Field(name='tau_p')
tau_B1 = dist.Field(name='tau_B1', bases=xbasis)
tau_B2 = dist.Field(name='tau_B2', bases=xbasis)
tau_D1 = dist.Field(name='tau_D1', bases=xbasis)
tau_D2 = dist.Field(name='tau_D2', bases=xbasis)
tau_M1 = dist.Field(name='tau_M1', bases=xbasis)
tau_M2 = dist.Field(name='tau_M2', bases=xbasis)
tau_u1 = dist.VectorField(coords,name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords,name='tau_u2', bases=xbasis)
tau_u3 = dist.Field(name='tau_u3', bases=xbasis)
tau_u4 = dist.Field(name='tau_u4', bases=xbasis)
tau_T1 = dist.Field(name='tau_t1', bases=xbasis)
tau_T2 = dist.Field(name='tau_t2', bases=xbasis)
tau_C1 = dist.Field(name='tau_c1', bases=xbasis)
tau_C2 = dist.Field(name='tau_c2', bases=xbasis)


mask_grid = dist.Field(bases=(xbasis,zbasis))

mask_grid.change_scales(dealias)
xg = dist.local_grid(xbasis,scale=dealias)
print(xg.shape)
zg = dist.local_grid(zbasis,scale=dealias)
print(zg.shape)
# print(xg.squeeze(), zg.squeeze())
for i in range(xg.shape[0]):
    for j in range(zg.shape[1]):
        x = xg[i,0]
        z = zg[0,j]
        print(x,z)
        #print(x,y) - need to rescale from [0,1] to [-1,1]
        mask_grid['g'][i,j] = interp_smooth(x, z)

# %%

fig = plt.figure(figsize=[9,9])
ax = fig.add_subplot(1,1,1)

# xmesh, zmesh = np.meshgrid(xg, zg)
# xorder = np.argsort(xg)
# zorder = np.argsort(zg)
xmesh, zmesh = dpt.quad_mesh(xg[:,0], zg[0,:])
maskgraph=np.transpose(mask_grid['g'])
ax.pcolormesh(xmesh, zmesh, maskgraph,shading='auto')

# %%
import h5py
with h5py.File('mask.h5', 'w') as f:
    f['mask'] = mask_grid['g']
    f['x'] = x
    f['z'] = z

# %%
import numpy as np
import dedalus.public as d3
from scipy.interpolate import RegularGridInterpolator
import logging
logger = logging.getLogger(__name__)
import copy
import h5py
import matplotlib
import re

# Parameters
Lx, Lz = 4,1
Nx, Nz = 1024, 256
Ra_M = 1e5
D_0 = 0
D_H = 1/3
M_0 = 1
M_H = 0
N_s2=4/3
Qrad=0.0028
gamma=1

Prandtl = 1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = min(0.1, 0.25/gamma)
dtype = np.float64

# %%
# Bases
coords = d3.CartesianCoordinates('x','z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
D = dist.Field(name='D', bases=(xbasis,zbasis))
M = dist.Field(name='M', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
Z = dist.Field(name='Z', bases=zbasis)
T = dist.Field(name='T', bases=(xbasis,zbasis))
C = dist.Field(name='C', bases=(xbasis,zbasis))

tau_p = dist.Field(name='tau_p')
tau_B1 = dist.Field(name='tau_B1', bases=xbasis)
tau_B2 = dist.Field(name='tau_B2', bases=xbasis)
tau_D1 = dist.Field(name='tau_D1', bases=xbasis)
tau_D2 = dist.Field(name='tau_D2', bases=xbasis)
tau_M1 = dist.Field(name='tau_M1', bases=xbasis)
tau_M2 = dist.Field(name='tau_M2', bases=xbasis)
tau_u1 = dist.VectorField(coords,name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords,name='tau_u2', bases=xbasis)
tau_u3 = dist.Field(name='tau_u3', bases=xbasis)
tau_u4 = dist.Field(name='tau_u4', bases=xbasis)
tau_T1 = dist.Field(name='tau_t1', bases=xbasis)
tau_T2 = dist.Field(name='tau_t2', bases=xbasis)
tau_C1 = dist.Field(name='tau_c1', bases=xbasis)
tau_C2 = dist.Field(name='tau_c2', bases=xbasis)

# Substitutions    
#Kuo_Bretherton Equilibrium
kappa = (Ra_M * Prandtl/((M_0-M_H)*Lz**3))**(-1/2)
nu = (Ra_M / (Prandtl*(M_0-M_H)*Lz**3))**(-1/2)
print('kappa',kappa)
print('nu',nu)



x,z = dist.local_grids(xbasis,zbasis)

# Z.change_scales(3/2)


ex,ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

B_op = (np.absolute(D - M - N_s2*Z)+ M + D - N_s2*Z)/2
lq = B_op/2 + np.absolute(B_op)



Max = lambda A,B: (abs(A-N_s2*Z-B)+A-N_s2*Z+B)/2
eva = lambda A: A.evaluate()

dz= lambda A: d3.Differentiate(A, coords['z'])
dx= lambda A: d3.Differentiate(A, coords['x'])


ux=u@ex
uz=u@ez
dxux=dx(ux)
dzux=dz(ux)
dxuz=dx(uz)
dzuz=dz(uz)

grad_u = d3.grad(u) + ez* lift(tau_u1) # First-order reduction
grad_ux = grad_u@ex # First-order reduction
grad_uz = grad_u@ez # First-order reduction
grad_M = d3.grad(M) + ez*lift(tau_M1) # First-order reduction
grad_D = d3.grad(D) + ez*lift(tau_D1) # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1)
grad_C = d3.grad(C) + ez*lift(tau_C1)



mask = dist.Field(bases=(xbasis,zbasis))
grid_slices = dist.layouts[-1].slices(mask.domain, dealias)
mask_file = 'mask.h5'
with h5py.File(mask_file) as f:
    logger.info('loading mask from {}'.format(mask_file))
    mask.change_scales(dealias)
    mask['g'] = f['mask'][:,grid_slices[-1]]
mask = d3.Grid(mask).evaluate()

# # Example data for interpolation
# x_points = np.linspace(-1, 1, num=xg)  # Adjusted range to include possible negative and positive values
# y_points = np.linspace(-1, 1, num=zg)    # Adjusted range similarly
# data_on_grid = np.random.rand(xg, zg)  # Example data, replace with actual

# Define the interpolation function using RegularGridInterpolator
# def interp_smooth(x, y):
#     x = np.clip(x, x_points.min(), x_points.max())  # Clamping x within the interpolation range
#     y = np.clip(y, y_points.min(), y_points.max())  # Clamping y similarly
#     interpolator = RegularGridInterpolator((x_points, y_points), data_on_grid)
#     return interpolator([[x, y]])[0]


# %%

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, M, u, tau_p, tau_M1, tau_M2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) = - u@grad(M) - mask*gamma*M")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p)  + lift(tau_u2) = M*ez- u@grad(u) - mask*gamma*u")
problem.add_equation("u(z=0) = 0")
problem.add_equation("uz(z=Lz) = 0")
problem.add_equation("dz(ux)(z=Lz)=0")
problem.add_equation("M(z=0) = M_0")
problem.add_equation("M(z=Lz) = M_H")

problem.add_equation("integ(p) = 0") # Pressure gauge

# %%
# %%
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# %%
# Initial condition
M.fill_random('g', seed=28, distribution='normal', scale=1e-3) # Random noise
M.low_pass_filter(scales=0.25)
Z.change_scales(dealias)
M.change_scales(dealias)
M['g'] *= Z['g']* (Lz - Z['g']) # Damp noise at walls
M['g'] *=1e-3*(1-mask['g']) # Apply mask
M['g'] += (M_H-M_0)*Z['g'] # Add linear background
# M.change_scales(1)
# Z.change_scales(1)


# %%
# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots',sim_dt=0.025, max_writes=1)
snapshots.add_tasks(solver.state,layout='g')

# %%
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.5, threshold=0.05,
             max_change=1.1, min_change=0, max_dt=max_timestep)
CFL.add_velocity(u)

# %%
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(u@u)/nu, name='Re')


# %%
# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# %%



