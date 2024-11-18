import os
import re
import numpy as np
import h5py
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from PIL import Image
import moviepy

class Plot:
    def __init__(self, save_dir=None, handler="snapshots", dimension=2):
        if save_dir is None:
            save_dir = os.getcwd()
        self.save_dir = save_dir
        self.handler = handler
        self.folder_dir = os.path.join(save_dir, handler)
        self.file_paths = self.sort_files_in_directory()
        self.x = None
        self.y = None
        self.z = None
        self.scalekeys = None
        self.taskkeys = None
        self.dimension = dimension
        self.sim_time = None
        self.get_grid_data()
        self.get_sim_time()
        # Cache for loaded data
        self._data_cache = {}

    def sort_files_in_directory(self):
        file_paths = [
            os.path.join(self.folder_dir, file)
            for file in listdir(self.folder_dir)
            if os.path.isfile(os.path.join(self.folder_dir, file)) and file.endswith('.h5')
        ]
        file_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        return file_paths

    def get_grid_data(self):
        file = self.file_paths[-1]
        with h5py.File(file, mode='r') as file:
            print(list(file.keys()))
            self.scalekeys = list(file['scales'].keys())
            self.taskkeys = list(file['tasks'].keys())
            print("Scale keys:", self.scalekeys)
            print("Task keys:", self.taskkeys)
            
            if self.dimension == 2:
                xhash = self.scalekeys[-2]
                zhash = self.scalekeys[-1]
                self.x = np.array(file['scales'][xhash])
                self.z = np.array(file['scales'][zhash])
            elif self.dimension == 3:
                xhash = self.scalekeys[-3]
                yhash = self.scalekeys[-2]
                zhash = self.scalekeys[-1]
                self.x = np.array(file['scales'][xhash])
                self.y = np.array(file['scales'][yhash])
                self.z = np.array(file['scales'][zhash])
            else:
                raise ValueError("Unsupported number of dimensions. Please use 2 or 3.")

    def get_sim_time(self):
        self.sim_time = []
        for file_path in self.file_paths:
            with h5py.File(file_path, mode='r') as file:
                st = file['scales/sim_time']
                self.sim_time.extend(np.array(st))

    def load_data(self, task_name):
        """
        Load data with caching mechanism to prevent multiple loads of the same data.
        """
        # Check if data is already in cache
        if task_name in self._data_cache:
            return self._data_cache[task_name]

        # Load data if not in cache
        data = []
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                task = file['tasks'][task_name][:]
                data.append(task)
        
        data = np.concatenate(data, axis=0)
        # Store in cache
        self._data_cache[task_name] = data
        return data

    def clear_cache(self):
        """
        Clear the data cache to free memory when needed.
        """
        self._data_cache.clear()

    def plot_all_snapshots(self, task_name, output_dir=None, cmap='RdBu_r', vmin=None, vmax=None, 
                          levelnum=10, figure_size=(10, 8), concentration=1.0):
        """
        Optimized version of plot_all_snapshots with:
        - Single data load with caching
        - Figure reuse
        - Proper memory management
        - Efficient min/max calculation
        """
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(self.save_dir, task_name)
        os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
        output_dir = os.path.join(output_dir, task_name)

        # Load data once and cache it
        print("Loading data...")
        data = self.load_data(task_name)
        
        # Calculate global min/max efficiently
        if vmin is None or vmax is None:
            print("Calculating global min/max...")
            global_min = np.min(data)
            global_max = np.max(data)
            vmin = global_min if vmin is None else vmin
            vmax = global_max if vmax is None else vmax
        
        # Calculate levels once
        levels = self.nonlinear_space(a=vmin, b=vmax, n=levelnum, concentration=concentration)
        
        # Create figure and axes once
        print("Creating figure...")
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create a progress counter
        total_frames = len(self.sim_time)
        
        # Process all snapshots
        print(f"Processing {total_frames} snapshots...")
        for t in range(total_frames):
            # Clear previous plot content but keep the figure
            ax.clear()
            
            # Create new contour plot
            cont = ax.contourf(self.x, self.z, data[t].T, 
                             cmap=cmap, 
                             levels=levels)
            
            # Add colorbar (only on first iteration)
            if t == 0:
                plt.colorbar(cont, label=task_name)
            
            # Set labels and title
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_title(f"{task_name}, t = {self.sim_time[t]:.2f}")
            
            # Save the current frame
            plt.savefig(os.path.join(output_dir, f'{task_name}_{t:04d}.png'), 
                       dpi=200, 
                       bbox_inches='tight')
            
            # Clean up contour collections to free memory
            for coll in cont.collections:
                coll.remove()
            
            # Print progress
            if (t + 1) % 10 == 0:
                print(f"Progress: {t + 1}/{total_frames} frames processed")
        
        # Clean up
        plt.close(fig)
        print("Finished processing all snapshots!")

    def animate(self, task_name, output_dir=None, fps=10, use_existing_pics=True, output_type='gif'):
        """
        Create an animation for a specific task.

        Args:
            task_name (str): Name of the task to animate.
            output_file (str, optional): Path to save the animation. Defaults to None.
            fps (int, optional): Frames per second. Defaults to 10.
            use_existing_pics (bool, optional): Use existing pictures if available. Defaults to True.
            output_type (str, optional): Type of output file ('gif' or 'mp4'). Defaults to 'gif'.
        """
        if output_dir is None:
            output_file = os.path.join(self.save_dir, f'{task_name}_animation.{output_type}')
            pics_folder = os.path.join(self.save_dir, task_name)
            
        else:
            output_file = os.path.join(output_dir, f'{task_name}_animation.{output_type}')
            pics_folder = os.path.join(output_dir, task_name)


        
        existing_pics = sorted(glob.glob(os.path.join(pics_folder, f'{task_name}_*.png')))

        if use_existing_pics and existing_pics:
            print(f"Using existing pictures from {pics_folder}")
            self.create_animation_from_pics(existing_pics, output_file, fps, output_type)
        else:
            print("Generating new animation from data")
            self.create_animation_from_data(task_name, output_file, fps, output_type)

    def create_animation_from_pics(self, pic_files, output_file, fps, output_type):
        images = [Image.open(f) for f in pic_files]
        
        if output_type.lower() == 'gif':
            images[0].save(output_file, save_all=True, append_images=images[1:], 
                           duration=1000/fps, loop=0)
        elif output_type.lower() == 'mp4':
            import moviepy.editor as mpy
            clip = mpy.ImageSequenceClip(pic_files, fps=fps)
            clip.write_videofile(output_file)
        else:
            raise ValueError("Output type must be either 'gif' or 'mp4'")

        print(f"Animation saved as {output_file}")
        
        # Use moviepy for MP4
        elif output_type.lower() == 'mp4':
            import moviepy.editor as mpy
            
            # Create clip with specific codec and quality settings
            clip = mpy.ImageSequenceClip(pic_files, fps=fps)
            clip.write_videofile(output_file, 
                               codec='libx264',
                               bitrate='16M',  # high bitrate for better quality
                               preset='slower'  # slower encoding = better compression
                               )
        else:
            raise ValueError("Output type must be either 'gif' or 'mp4'")
    
        print(f"Animation saved as {output_file}")

    def create_animation_from_data(self, task_name, output_file, fps, output_type):
        """
        Create animation directly from data.

        Args:
            task_name (str): Name of the task to animate.
            output_file (str): Path to save the animation.
            fps (int): Frames per second.
            output_type (str): Type of output file ('gif' or 'mp4').
        """
        data = self.load_data(task_name)
        # print(data.shape)

        fig, ax = plt.subplots(figsize=(10, 8))

        vmin, vmax = self.get_global_minmax(task_name)
        
        im = ax.imshow(data[0].T, cmap='RdBu_r', aspect='auto', origin='lower', 
                       vmin=vmin, vmax=vmax, extent=[self.x[0], self.x[-1], self.z[0], self.z[-1]])
        plt.colorbar(im, label=task_name)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        title = ax.set_title(f"{task_name}, t = {self.sim_time[0]:.2f}")

        def update(frame):
            im.set_array(data[frame].T)
            title.set_text(f"{task_name}, t = {self.sim_time[frame]:.2f}")
            return im, title
        

        anim = animation.FuncAnimation(fig, update, frames=len(self.sim_time), blit=True)

        if output_type.lower() == 'gif':
            writer = animation.PillowWriter(fps=fps)
        elif output_type.lower() == 'mp4':
            writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
        else:
            raise ValueError("Output type must be either 'gif' or 'mp4'")

        anim.save(output_file, writer=writer)
        plt.close(fig)
        print(f"Animation saved as {output_file}")
        
    @staticmethod
    def nonlinear_space(a, b, n, concentration=0.5):
        """
        Generate a non-linear space with more points around the center.
        
        Parameters:
        a, b : float
            The start and end of the interval.
        n : int
            The number of points to generate.
        concentration : float
            Controls the degree of concentration at the center. 
            Higher values increase central concentration.
            Controls the distribution of points. 
            > 1: more points near the center
            < 1: more points near the boundaries
            = 1: approximately linear distributiontion.

        Returns:
        array : numpy array
            An array of n points between a and b, with more points near the center.
        """
        
        if n == 1:
            return np.linspace(a, b, n) 
        
        # Generate a linear space from -1 to 1
        x = np.linspace(-1, 1, n)
        
        # Apply sinh function to concentrate points
        y = np.sinh(concentration * x) / np.sinh(concentration)
        
        # Scale and shift to the desired interval [a, b]
        return a + (b - a) * (y + 1) / 2