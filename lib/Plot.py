import os
import re

class Plot:
    
    def sort_files_in_directory(folder_dir, file_extension='.h5'):
        """
        Sort files in a directory based on the numeric part of their filenames.
        
        Args:
        folder_dir (str): Path to the directory containing the files.
        file_extension (str): File extension to filter by (default is '.h5').
        
        Returns:
        list: Sorted list of full file paths.
        """
        # Get all files with the specified extension
        file_paths = [
            os.path.join(folder_dir, file)
            for file in os.listdir(folder_dir)
            if os.path.isfile(os.path.join(folder_dir, file)) and file.endswith(file_extension)
        ]
        
        # Sort files based on the numeric part of their filenames
        file_paths.sort(key=lambda f: int(re.sub('\D', '', os.path.basename(f))))
        
        return file_paths
    
    