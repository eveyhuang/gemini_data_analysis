#!/usr/bin/env python3
import os
import re
import sys
import unicodedata
import argparse
from pathlib import Path

def sanitize_name(name, replace_char='_'):
    """
    Sanitize a name by replacing spaces and hyphens with underscores.
    
    Args:
        name: The name to sanitize
        replace_char: Character to use as replacement for spaces and hyphens
        
    Returns:
        A sanitized version of the name
    """
    # Normalize Unicode characters (e.g., convert 'é' to 'e')
    name = unicodedata.normalize('NFKD', name)
    
    sanitized = name.replace(' ', replace_char).replace('-', replace_char).replace('._', replace_char)
    sanitized = re.sub(f'{replace_char}+', replace_char, sanitized)
    sanitized = sanitized.strip(replace_char)
    return sanitized

def get_all_paths(directory):
    """
    Get all paths in the directory, sorted by depth (deepest first).
    This ensures we process child items before their parents.
    
    Args:
        directory: The root directory to process
        
    Returns:
        List of (path, is_dir) tuples, sorted by depth (deepest first)
    """
    all_paths = []
    base_depth = directory.rstrip(os.sep).count(os.sep)
    
    for root, dirs, files in os.walk(directory, topdown=False):
        # Add files
        for name in files:
            if name != '.DS_Store':  # Skip .DS_Store files
                path = os.path.join(root, name)
                depth = path.count(os.sep) - base_depth
                all_paths.append((path, False, depth))
        
        # Add directories
        for name in dirs:
            path = os.path.join(root, name)
            depth = path.count(os.sep) - base_depth
            all_paths.append((path, True, depth))
    
    # Sort by depth (descending) and then by path
    return sorted(all_paths, key=lambda x: (-x[2], x[0]))

def process_directory(directory, dry_run=False, verbose=False):
    """
    Process all files and directories in the given directory.
    
    Args:
        directory: The directory to process
        dry_run: If True, only print what would be done without making changes
        verbose: If True, print detailed information about each file processed
    """
    renamed_count = 0
    directory = os.path.abspath(directory)
    
    # Get all paths sorted by depth
    all_paths = get_all_paths(directory)
    
    # Keep track of renamed paths to update subsequent paths
    path_updates = {}
    
    # Process each path
    for path, is_dir, _ in all_paths:
        # Get the parent directory and current name
        parent_dir = os.path.dirname(path)
        current_name = os.path.basename(path)
        
        # Update parent_dir based on any previous renames
        for old_path, new_path in path_updates.items():
            if parent_dir.startswith(old_path):
                parent_dir = parent_dir.replace(old_path, new_path, 1)
        
        # Sanitize the name
        sanitized_name = sanitize_name(current_name)
        
        # Skip if no changes needed
        if current_name == sanitized_name:
            if verbose:
                print(f"✓ {path} (no changes needed)")
            continue
        
        # Create the new path
        new_path = os.path.join(parent_dir, sanitized_name)
        
        if dry_run:
            print(f"Would rename: {path} -> {new_path}")
            path_updates[path] = new_path
            renamed_count += 1
        else:
            try:
                # Create parent directory if it doesn't exist
                os.makedirs(parent_dir, exist_ok=True)
                
                # Check if target exists
                if os.path.exists(new_path) and path != new_path:
                    print(f"Error: Cannot rename {path} -> {new_path} (target already exists)")
                    continue
                
                # Rename the file or directory
                os.rename(path, new_path)
                print(f"Renamed: {path} -> {new_path}")
                path_updates[path] = new_path
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {path}: {e}")
    
    return renamed_count

def main():
    parser = argparse.ArgumentParser(description='Sanitize filenames and directory names by removing problematic characters and replacing spaces with underscores.')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return 1
    
    print(f"Processing directory: {args.directory}")
    if args.dry_run:
        print("DRY RUN - no changes will be made")
    
    renamed_count = process_directory(args.directory, args.dry_run, args.verbose)
    
    if args.dry_run:
        print(f"Would rename {renamed_count} files/directories")
    else:
        print(f"Renamed {renamed_count} files/directories")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 