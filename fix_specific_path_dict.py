#!/usr/bin/env python3
import os
import sys
import json
import re
import unicodedata
import argparse
from pathlib import Path

def sanitize_filename_general(filename, replace_char='_'):
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

def is_latin1_compatible(filename):
    """
    Check if a filename can be encoded in latin-1.
    
    Args:
        filename: The filename to check
        
    Returns:
        True if the filename can be encoded in latin-1, False otherwise
    """
    try:
        filename.encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False

def sanitize_filename_latin1(filename):
    """
    Create a safe version of a filename by replacing non-latin-1 characters.
    
    Args:
        filename: The original filename
        
    Returns:
        A safe version of the filename
    """
    # Get the base name and extension
    base, ext = os.path.splitext(filename)
    
    # Replace problematic characters with underscores
    safe_base = ''
    for char in base:
        try:
            char.encode('latin-1')
            safe_base += char
        except UnicodeEncodeError:
            safe_base += '_'
    
    # Remove multiple consecutive underscores
    safe_base = '_'.join(filter(None, safe_base.split('_')))
    
    # Combine with the extension
    return safe_base + ext

def sanitize_filename_ascii(filename):
    """
    Create a safe version of a filename by replacing non-ASCII characters.
    
    Args:
        filename: The original filename
        
    Returns:
        A safe version of the filename
    """
    # Get the base name and extension
    base, ext = os.path.splitext(filename)
    
    # Replace problematic characters with underscores
    safe_base = ''
    for char in base:
        try:
            char.encode('ascii')
            safe_base += char
        except UnicodeEncodeError:
            safe_base += '_'
    
    # Remove multiple consecutive underscores
    safe_base = '_'.join(filter(None, safe_base.split('_')))
    
    # Combine with the extension
    return safe_base + ext

def fix_path_dict(path_dict_file, method='general', dry_run=False, verbose=False):
    """
    Fix encoding issues in a path_dict.json file.
    
    Args:
        path_dict_file: Path to the path_dict.json file
        method: Method to use for sanitizing filenames ('general', 'latin1', or 'ascii')
        dry_run: If True, only print what would be done without making changes
        verbose: If True, print detailed information about each file processed
        
    Returns:
        True if the file was processed successfully, False otherwise
    """
    try:
        # Read the path_dict.json file
        with open(path_dict_file, 'r', encoding='utf-8') as f:
            path_dict = json.load(f)
        
        # Select the appropriate sanitization method
        if method == 'general':
            sanitize_fn = sanitize_filename_general
        elif method == 'latin1':
            sanitize_fn = sanitize_filename_latin1
        elif method == 'ascii':
            sanitize_fn = sanitize_filename_ascii
        else:
            print(f"Error: Unknown method '{method}'")
            return False
        
        # Track changes for reporting
        changes = []
        
        # Process each entry in the path_dict
        for key, value in list(path_dict.items()):
            # Sanitize the key (which is often a file path or name)
            sanitized_key = sanitize_fn(key)
            
            # Process each item in the value list
            for i, item in enumerate(value):
                if isinstance(item, list) and len(item) >= 2:
                    # The second element is typically the file path
                    file_path = item[1]
                    file_name = os.path.basename(file_path)
                    
                    # Sanitize the file name
                    sanitized_file_name = sanitize_fn(file_name)
                    
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

def main():
    parser = argparse.ArgumentParser(description='Fix encoding issues in a path_dict.json file.')
    parser.add_argument('path_dict_file', help='Path to the path_dict.json file')
    parser.add_argument('--method', choices=['general', 'latin1', 'ascii'], default='general',
                        help='Method to use for sanitizing filenames (default: general)')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.path_dict_file):
        print(f"Error: {args.path_dict_file} is not a valid file")
        return 1
    
    if not args.path_dict_file.endswith('.json'):
        print(f"Warning: {args.path_dict_file} does not appear to be a JSON file")
    
    print(f"Processing {args.path_dict_file} using {args.method} method")
    if args.dry_run:
        print("DRY RUN - no changes will be made")
    
    success = fix_path_dict(args.path_dict_file, args.method, args.dry_run, args.verbose)
    
    if success:
        print("Processing completed successfully")
        return 0
    else:
        print("Processing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 