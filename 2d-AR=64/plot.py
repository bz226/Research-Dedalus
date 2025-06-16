from lib.dedalus_Plot import Plot
Plotter=Plot(save_dir="/scratch/zb2113/DedalusData/2D/MRBC_2D_Kappa_1.0e-04_Pr_1.0e+00_QR_1.0e-04_DH_5.0e-01_MH_-5.0e-01_deltaM_0.0e+00_Lx_6.4e+01_Nz_64")
Plotter.get_sim_time()
# print(Plotter.sim_time)
Plotter.plot_all_snapshots('G', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/mountain")
# Plotter.plot_all_snapshots('M', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/mountain")