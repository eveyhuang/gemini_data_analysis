#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path

def remove_split_folders(directory):
    """
    Recursively removes all folders that start with 'split-' and their contents.
    
    Args:
        directory (Path): Path object of the directory to process
    """
    try:
        # Walk through all directories from bottom to top
        for root, dirs, files in os.walk(directory, topdown=False):
            # Check if the current directory starts with 'split-'
            if os.path.basename(root).startswith('split-'):
                try:
                    shutil.rmtree(root)
                    print(f"Removed split folder and contents: {root}")
                except Exception as e:
                    print(f"Error removing split folder {root}: {e}")
    except Exception as e:
        print(f"An error occurred while removing split folders: {e}")

def remove_empty_folders(directory):
    """
    Recursively removes empty folders from the given directory.
    
    Args:
        directory (Path): Path object of the directory to process
    """
    try:
        # Walk through all directories from bottom to top
        for root, dirs, files in os.walk(directory, topdown=False):
            # Check if the directory is empty
            if not os.listdir(root):
                try:
                    os.rmdir(root)
                    print(f"Removed empty folder: {root}")
                except Exception as e:
                    print(f"Error removing empty folder {root}: {e}")
    except Exception as e:
        print(f"An error occurred while removing empty folders: {e}")

def remove_mp4_files(directory):
    """
    Recursively removes all .mp4 files from the given directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory to process
    """
    try:
        # Convert the directory path to a Path object
        root_dir = Path(directory)
        
        # Check if the directory exists
        if not root_dir.exists():
            print(f"Error: Directory '{directory}' does not exist.")
            return
        
        # Count the number of files removed
        files_removed = 0
        
        # Walk through all directories and files
        for path in root_dir.rglob("*.mp4"):
            try:
                # Remove the file
                path.unlink()
                print(f"Removed: {path}")
                files_removed += 1
            except Exception as e:
                print(f"Error removing {path}: {e}")
        
        print(f"\nRemoved {files_removed} MP4 files.")
        
        # Remove split- folders and their contents
        print("\nRemoving split- folders and their contents...")
        remove_split_folders(root_dir)
        
        # Remove empty folders after removing MP4 files and split folders
        print("\nRemoving empty folders...")
        remove_empty_folders(root_dir)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_mp4_files.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    remove_mp4_files(directory_path) 