#!/usr/bin/env python3
import os
import re
import sys
import unicodedata
import argparse
from pathlib import Path

def sanitize_filename(filename, replace_char='_'):
    """
    Sanitize a filename by removing or replacing problematic characters.
    
    Args:
        filename: The filename to sanitize
        replace_char: Character to use as replacement for invalid characters
        
    Returns:
        A sanitized version of the filename
    """
    # Get the base name and extension
    base, ext = os.path.splitext(filename)
    
    # Normalize Unicode characters (e.g., convert 'é' to 'e')
    base = unicodedata.normalize('NFKD', base)
    
    # Replace problematic characters with the replacement character
    # This regex matches any character that's not alphanumeric, hyphen, underscore, or period
    sanitized_base = re.sub(r'[^\w\-\.]', replace_char, base)
    
    # Remove multiple consecutive replacement characters
    sanitized_base = re.sub(f'{replace_char}+', replace_char, sanitized_base)
    
    # Remove leading/trailing replacement characters
    sanitized_base = sanitized_base.strip(replace_char)
    
    # Combine with the extension
    return sanitized_base + ext

def process_directory(directory, dry_run=False, verbose=False):
    """
    Process all files and directories in the given directory.
    
    Args:
        directory: The directory to process
        dry_run: If True, only print what would be done without making changes
        verbose: If True, print detailed information about each file processed
    """
    # Convert to Path object for easier handling
    dir_path = Path(directory)
    
    # Get all files and directories, sorted to process directories first
    all_items = []
    for root, dirs, files in os.walk(directory, topdown=False):
        # Add directories first (to process from bottom up)
        for dir_name in dirs:
            all_items.append((os.path.join(root, dir_name), True))
        # Then add files
        for file_name in files:
            all_items.append((os.path.join(root, file_name), False))
    
    # Process each item
    renamed_count = 0
    for item_path, is_dir in all_items:
        item_name = os.path.basename(item_path)
        sanitized_name = sanitize_filename(item_name)
        
        # Skip if no changes needed
        if item_name == sanitized_name:
            if verbose:
                print(f"✓ {item_path} (no changes needed)")
            continue
        
        # Create the new path
        new_path = os.path.join(os.path.dirname(item_path), sanitized_name)
        
        if dry_run:
            print(f"Would rename: {item_path} -> {new_path}")
        else:
            try:
                os.rename(item_path, new_path)
                print(f"Renamed: {item_path} -> {new_path}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {item_path}: {e}")
    
    return renamed_count

def main():
    parser = argparse.ArgumentParser(description='Sanitize filenames in a directory by removing problematic characters.')
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