"""
Plot 2D cartesian snapshots.

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['average buoyancy','buoyancy']
    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Layout
    nrows, ncols = 2, 1
    image = plot_tools.Box(4, 1)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)


    # Plot writes
    with h5py.File(filename, mode='r') as file:
        
        for index in range(start, start+count):
            avgb = file['tasks']['average buoyancy']
            t = avgb.dims[0]['sim_time']
            x = avgb.dims[1]
            z = avgb.dims[2]
            # Plot data
            avb_phase=avgb[:,0,:]
            plt.figure(figsize=(6,7), dpi=100)
            plt.pcolormesh(x[:], z[:], avb_phase, shading='nearest', cmap='twilight_shifted')
            plt.colorbar(label='average buoyancy')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('Average buoyancy')
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

