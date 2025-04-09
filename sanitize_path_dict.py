#!/usr/bin/env python3
import os
import re
import sys
import json
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

def sanitize_path_dict(path_dict_file, dry_run=False, verbose=False):
    """
    Sanitize file paths and names in a path_dict.json file.
    
    Args:
        path_dict_file: Path to the path_dict.json file
        dry_run: If True, only print what would be done without making changes
        verbose: If True, print detailed information about each file processed
        
    Returns:
        True if the file was processed successfully, False otherwise
    """
    try:
        # Read the path_dict.json file
        with open(path_dict_file, 'r', encoding='utf-8') as f:
            path_dict = json.load(f)
        
        # Track changes for reporting
        changes = []
        
        # Process each entry in the path_dict
        for key, value in list(path_dict.items()):
            # Sanitize the key (which is often a file path or name)
            sanitized_key = sanitize_filename(key)
            
            # Process each item in the value list
            for i, item in enumerate(value):
                if isinstance(item, list) and len(item) >= 2:
                    # The second element is typically the file path
                    file_path = item[1]
                    file_name = os.path.basename(file_path)
                    
                    # Sanitize the file name
                    sanitized_file_name = sanitize_filename(file_name)
                    
                    # Create a new file path with the sanitized file name
                    dir_path = os.path.dirname(file_path)
                    sanitized_file_path = os.path.join(dir_path, sanitized_file_name)
                    
                    # Update the item with the sanitized file path
                    item[1] = sanitized_file_path
                    
                    # If the file name changed, update the first element too (if it's a file name)
                    if item[0] == file_name:
                        item[0] = sanitized_file_name
                    
                    # Track the change
                    if file_name != sanitized_file_name:
                        changes.append((file_path, sanitized_file_path))
        
        # If the key changed, update the path_dict
        if key != sanitized_key:
            path_dict[sanitized_key] = path_dict.pop(key)
            changes.append((key, sanitized_key))
        
        # Print changes if in verbose mode
        if verbose and changes:
            print(f"Changes in {path_dict_file}:")
            for old, new in changes:
                print(f"  {old} -> {new}")
        
        # Save the updated path_dict if not in dry run mode
        if not dry_run and changes:
            with open(path_dict_file, 'w', encoding='utf-8') as f:
                json.dump(path_dict, f, indent=4)
            print(f"Updated {path_dict_file} with {len(changes)} changes")
        elif dry_run and changes:
            print(f"Would update {path_dict_file} with {len(changes)} changes")
        elif verbose:
            print(f"No changes needed in {path_dict_file}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {path_dict_file}: {e}")
        return False

def process_directory(directory, dry_run=False, verbose=False):
    """
    Process all path_dict.json files in the given directory.
    
    Args:
        directory: The directory to process
        dry_run: If True, only print what would be done without making changes
        verbose: If True, print detailed information about each file processed
        
    Returns:
        Number of files processed successfully
    """
    # Convert to Path object for easier handling
    dir_path = Path(directory)
    
    # Find all path_dict.json files
    path_dict_files = list(dir_path.glob('**/*path_dict.json'))
    
    if not path_dict_files:
        print(f"No path_dict.json files found in {directory}")
        return 0
    
    # Process each path_dict.json file
    success_count = 0
    for file_path in path_dict_files:
        if verbose:
            print(f"Processing {file_path}...")
        
        if sanitize_path_dict(file_path, dry_run, verbose):
            success_count += 1
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Sanitize file paths and names in path_dict.json files.')
    parser.add_argument('directory', help='Directory containing path_dict.json files')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return 1
    
    print(f"Processing directory: {args.directory}")
    if args.dry_run:
        print("DRY RUN - no changes will be made")
    
    success_count = process_directory(args.directory, args.dry_run, args.verbose)
    
    if args.dry_run:
        print(f"Would process {success_count} path_dict.json files")
    else:
        print(f"Processed {success_count} path_dict.json files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 