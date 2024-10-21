from lib.dedalus_Plot import Plot
folder="mountain2"
Plotter=Plot(save_dir="/scratch/zb2113/DedalusData/"+folder)
Plotter.get_sim_time()
print(Plotter.sim_time)
Plotter.plot_all_snapshots('ux', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder ,levelnum=20)
Plotter.plot_all_snapshots('uz', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder,levelnum=20)
Plotter.plot_all_snapshots('M', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder,levelnum=20)
Plotter.animate('ux',use_existing_pics=True,output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder)
Plotter.animate('uz',use_existing_pics=True,output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder)
Plotter.animate('M',use_existing_pics=True,output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder)

from lib.dedalus_Plot import Plot
folder="mountain3"
Plotter=Plot(save_dir="/scratch/zb2113/DedalusData/"+folder)
Plotter.get_sim_time()
print(Plotter.sim_time)
Plotter.plot_all_snapshots('ux', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder ,levelnum=20)
Plotter.plot_all_snapshots('uz', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder,levelnum=20)
Plotter.plot_all_snapshots('M', output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder,levelnum=20)
Plotter.animate('ux',use_existing_pics=True,output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder)
Plotter.animate('uz',use_existing_pics=True,output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder)
Plotter.animate('M',use_existing_pics=True,output_dir="/home/zb2113/Dedalus-Postanalysis/2D/"+folder)