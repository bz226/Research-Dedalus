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
Nx, Nz = 512, 128
Ra_M = 1e5
# D_0 = 0
# D_H = 1/3
M_0 = 1
M_H = 0
N_s2=4/3
Qrad=0.0028
gamma=100
gamma_s=1

Prandtl = 1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = min(0.125, 0.25/gamma)
dtype = np.float64

mask_dir= "/home/zb2113/Research-Dedalus/2d-AR=64/masks"
save_dir= "/scratch/zb2113/DedalusData/mountain"
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
time = dist.Field(name='time', bases=(xbasis,zbasis))
X = dist.Field(name='X', bases=xbasis)

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
tau_t1 = dist.Field(name='tau_t1', bases=xbasis)
tau_t2 = dist.Field(name='tau_t2', bases=xbasis)
F = dist.Field(name='F', bases=(xbasis,zbasis))
M_s = dist.Field(name='M_s', bases=(xbasis,zbasis))
u_s = dist.Field(name='u_s', bases=(xbasis,zbasis))

# Substitutions    
#Kuo_Bretherton Equilibrium
kappa = (Ra_M * Prandtl/((M_0-M_H)*Lz**3))**(-1/2)
nu = (Ra_M / (Prandtl*(M_0-M_H)*Lz**3))**(-1/2)
print('kappa',kappa)
print('nu',nu)



x,z = dist.local_grids(xbasis,zbasis)

# Z.change_scales(3/2)
Z['g']=z
X['g']=x

ex,ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

B_op = (np.absolute(D - M - N_s2*Z)+ M + D - N_s2*Z)/2
lq = B_op/2 + np.absolute(B_op)

# F=(max((Lx/10-x)/(Lx/10),0)+max((-Lx+Lx/10+x)/(Lx/10),0))

F['g']= (Lx/10-x)/(Lx/10)/2 +np.absolute((Lx/10-x)/(Lx/10))/2 + (-Lx+Lx/10+x)/(Lx/10)/2 + np.absolute((-Lx+Lx/10+x)/(Lx/10))/2
M_s['g']=z
u_s['g']=10*z

max = lambda A,B: (abs(A-N_s2*z-B)+A-N_s2*z+B)/2
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
sponge = dist.Field(bases=(xbasis,zbasis))
grid_slices = dist.layouts[-1].slices(mask.domain, dealias)
#Mountain
mask_file = mask_dir+'/mask.h5'
with h5py.File(mask_file) as f:
    logger.info('loading mask from {}'.format(mask_file))
    mask.change_scales(dealias)
    mask['g'] = f['mask'][:,grid_slices[-1]]
mask = d3.Grid(mask).evaluate()
#Sponge 
mask_file = mask_dir+'/mask_sp.h5'
with h5py.File(mask_file) as f:
    logger.info('loading mask from {}'.format(mask_file))
    sponge.change_scales(dealias)
    sponge['g'] = f['mask'][:,grid_slices[-1]]
    

sponge = d3.Grid(sponge).evaluate()




# %%

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, M, u, time, tau_p, tau_M1, tau_M2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
# problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) = - u@grad(M) - mask*gamma*(M-M_0)  -F*sponge*gamma_s*(M-(Lz-Z)/Lz)")
# problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) = - u@grad(M) - mask*gamma*(M-M_0)  -sponge*gamma*(M-(Z-Lz)/Lz)")
# problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) = - u@grad(M) - mask*gamma*(M-M_0)  -sponge*gamma*M")
problem.add_equation("dt(M) - kappa*div(grad_M) + lift(tau_M2) = - u@grad(M)  -F*sponge*gamma_s*(M-M_s)")
# problem.add_equation("dt(u) - nu*div(grad_u) + grad(p)  + lift(tau_u2) -M*ez = - u@grad(u) - mask*gamma*u- F*sponge*gamma_s*(u-10/1*Z*ex)")
# problem.add_equation("dt(u) - nu*div(grad_u) + grad(p)  + lift(tau_u2) -M*ez = - u@grad(u) - mask*gamma*u- sponge*gamma*(u-10/1*Z*ex)")
# problem.add_equation("dt(u) - nu*div(grad_u) + grad(p)  + lift(tau_u2) -M*ez = - u@grad(u) - mask*gamma*u- sponge*gamma*(u-5/1*Z*ex)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p)  + lift(tau_u2) -M*ez = - u@grad(u) - F*sponge*gamma_s*(u-u_s*ex)")
problem.add_equation("dt(time) = 1 ")
problem.add_equation("u(z=0) = 0")
problem.add_equation("uz(z=Lz) = 0")
problem.add_equation("dz(ux)(z=Lz)=0")
problem.add_equation("M(z=0) = M_0")
problem.add_equation("M(z=Lz) = M_H")
# problem.add_equation("dx(time)(z=0) = 0")
# problem.add_equation("dz(time)(z=0) = 0")
# problem.add_equation("dx(time)(z=Lz) = 0")
# problem.add_equation("dz(time)(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# %%
# %%
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# %%
# Initial condition
M.fill_random('g', seed=28, distribution='normal', scale=1e-3) # Random noise
M['g'] *= z * (Lz - z) # Damp noise at walls
M['g'] += (M_H-M_0)*z+M_0 # Add linear background
M.change_scales(dealias)

# M['g'] *=(1-mask['g']) # Apply mask
M['g'] *= (1-sponge['g']) # Apply sponge
time['g']=0

# M.change_scales(1)
# Z.change_scales(1)


# %%
# Analysis
snapshots = solver.evaluator.add_file_handler(save_dir+'/snapshots',sim_dt=0.25, max_writes=1)
snapshots.add_tasks(solver.state,layout='g')
snapshots.add_task(u@u, layout='g', name='u square')
snapshots.add_task(u@ez, layout='g', name='uz')
snapshots.add_task(u@ex, layout='g', name='ux')
snapshots.add_task(F*sponge, layout='g', name='Fsponge')



maskcheck = solver.evaluator.add_file_handler(save_dir+'/maskcheck',sim_dt=5, max_writes=1)
maskcheck.add_task(mask, layout='g', name='mask')
maskcheck.add_task(sponge, layout='g', name='sponge')


# %%
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.1, min_change=0, max_dt=max_timestep)
CFL.add_velocity(u)

# %%
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
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