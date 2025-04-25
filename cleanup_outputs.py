import os
import json
import shutil
from pathlib import Path
import unicodedata
import re
def is_valid_json(file_path):
    """Check if a file contains valid JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            json.loads(content)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

def get_json_size(file_path):
    """Get the size of the JSON content in characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return len(content)
    except (UnicodeDecodeError, FileNotFoundError):
        return 0

def normalize_name(name, replace_char='_'):
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

def should_merge_dirs(dir1, dir2):
    """Check if two directories should be merged based on their normalized names."""
    name1 = normalize_name(dir1.name)
    name2 = normalize_name(dir2.name)
    return name1 == name2

def merge_similar_folders(base_dir):
    """Find and merge folders that differ only by space/underscore."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # First, process the current directory level
    try:
        all_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    except Exception as e:
        print(f"Error accessing directory {base_dir}: {e}")
        return
    
    # Group directories by their normalized names
    dir_groups = {}
    for d in all_dirs:
        normalized_name = normalize_name(d.name)
        if normalized_name not in dir_groups:
            dir_groups[normalized_name] = []
        dir_groups[normalized_name].append(d)
    
    # Process groups with multiple directories
    for normalized_name, dirs in dir_groups.items():
        if len(dirs) > 1 or dirs[0].name != normalized_name:  # Also process single dirs that aren't normalized
            print(f"\nFound similar folders: {[d.name for d in dirs]}")
            
            # Always create or use the normalized name directory
            target_dir = base_dir / normalized_name
            if not target_dir.exists():
                target_dir.mkdir()
                print(f"Created normalized target directory: {target_dir.name}")
            
            # Move all files from source directories to target
            for source_dir in dirs:
                if source_dir != target_dir and source_dir.exists():
                    print(f"Moving files from {source_dir.name} to {target_dir.name}")
                    try:
                        # Move all contents from source to target
                        for item in source_dir.iterdir():
                            target_path = target_dir / normalize_name(item.name)
                            if target_path.exists():
                                # If both exist, keep the one with more content
                                if item.is_file():
                                    source_size = item.stat().st_size
                                    target_size = target_path.stat().st_size
                                    if source_size > target_size:
                                        target_path.unlink()
                                        shutil.move(str(item), str(target_path))
                                        print(f"  Replaced: {target_path.name} with larger version")
                                    else:
                                        print(f"  Keeping existing file: {target_path.name}")
                            else:
                                # If target doesn't exist, just move the file/directory
                                try:
                                    if item.is_file():
                                        shutil.move(str(item), str(target_path))
                                        print(f"  Moved file: {item.name} -> {target_path.name}")
                                    elif item.is_dir():
                                        # For directories, merge recursively
                                        if target_path.exists():
                                            merge_similar_folders(item)  # Merge contents first
                                        else:
                                            shutil.move(str(item), str(target_path))
                                            print(f"  Moved directory: {item.name} -> {target_path.name}")
                                except Exception as e:
                                    print(f"  Error moving {item.name}: {e}")
                        
                        # Remove the source directory after moving all contents
                        try:
                            shutil.rmtree(source_dir)
                            print(f"Removed source directory: {source_dir.name}")
                        except Exception as e:
                            print(f"Error removing directory {source_dir.name}: {e}")
                            
                    except Exception as e:
                        print(f"Error processing {source_dir.name}: {e}")
    
    # Now process remaining subdirectories recursively
    for d in base_dir.iterdir():
        if d.is_dir():
            merge_similar_folders(d)

def process_repetitive_files(pair):
    """Process a pair of repetitive files and keep the better one based on priority:
    1. Longest valid JSON with content
    2. Valid JSON with content
    3. Non-valid JSON with content
    4. Empty or invalid JSON
    """
    print("\nDetailed file analysis:")
    # Score each file based on priority
    file_scores = []
    for file in pair:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                content_length = len(content)
                
                print(f"\nFile: {os.path.basename(file)}")
                print(f"Content length: {content_length}")
                print(f"First 100 chars: {content[:100]}")
                
                # Check if content is empty or just {}
                if content == '{}' or not content:
                    score = (0, 0)  # Priority 4: Empty JSON
                    print("Status: Empty JSON")
                else:
                    # Try to parse as JSON
                    try:
                        json.loads(content)
                        # Valid JSON with content - score based on length
                        score = (3, content_length)  # Priority 1: Valid JSON with content
                        print("Status: Valid JSON with content")
                    except json.JSONDecodeError as e:
                        # Non-valid JSON with content - prioritize by length
                        score = (2, content_length)  # Priority 3: Non-valid JSON with content
                        print(f"Status: Invalid JSON with content (Error: {str(e)})")
        except Exception as e:
            score = (0, 0)  # Priority 4: Can't read file
            print(f"Status: Error reading file ({str(e)})")
            
        print(f"Score: {score}")
        file_scores.append((file, score))
    
    # Sort by score (higher is better)
    file_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFinal decision:")
    for i, (file, score) in enumerate(file_scores):
        status = "KEEP" if i == 0 else "REMOVE"
        print(f"{status}: {os.path.basename(file)} (Score: {score})")
    
    # The best file is the first one
    keep_file = file_scores[0][0]
    remove_files = [f for f, _ in file_scores[1:]]
    
    return keep_file, remove_files

def find_repetitive_files(directory):
    """Find pairs of repetitive JSON files in a directory."""
    repetitive_pairs = []
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # Group files by their normalized name
    file_groups = {}
    for file in files:
        # Handle prefixes (all_, verbal_)
        if file.startswith('all_'):
            prefix = 'all_'
            rest = file[4:]  # Remove 'all_' prefix
        elif file.startswith('verbal_'):
            prefix = 'verbal_'
            rest = file[7:]  # Remove 'verbal_' prefix
        else:
            prefix = ''
            rest = file
            
        # Handle chunk numbers
        if '_chunk' in rest:
            base_part, chunk_part = rest.rsplit('_chunk', 1)
            # Normalize the base part by replacing spaces with underscores
            normalized_base = normalize_name(base_part)
            # Reconstruct the normalized key
            group_key = f"{prefix}{normalized_base}_chunk{chunk_part}"
        else:
            # For files without chunks, just normalize the whole name
            normalized_base = normalize_name(rest)
            group_key = f"{prefix}{normalized_base}"
        
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(file)
    
    # Find groups with multiple files
    for group_key, group in file_groups.items():
        if len(group) > 1:
            # Sort the group to ensure consistent order
            group.sort()
            repetitive_pairs.append([os.path.join(directory, f) for f in group])
    
    return repetitive_pairs

def remove_debug_and_bak_files(directory):
    """Remove all .debug and .bak files in the directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.debug') or file.endswith('.bak'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

def remove_mp4_from_filenames(directory):
    """Remove 'mp4' from JSON filenames in the directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4.json'):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, file.replace('.mp4.json', '.json'))
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file} -> {file.replace('.mp4.json', '.json')}")
                except Exception as e:
                    print(f"Error renaming {file}: {e}")

def process_directory(base_dir):
    """Process all subdirectories in the base directory."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Then merge similar folders
    print("\nMerging similar folders...")
    merge_similar_folders(base_dir)
    
    # First remove 'mp4' from JSON filenames
    print("\nRemoving 'mp4' from JSON filenames...")
    remove_mp4_from_filenames(base_dir)
    
    # Process repetitive files in each directory
    print("\nProcessing repetitive files...")
    for root, _, _ in os.walk(base_dir):
        print(f"\nChecking directory: {root}")
        repetitive_pairs = find_repetitive_files(root)
        if repetitive_pairs:
            print(f"Found {len(repetitive_pairs)} groups of repetitive files")
            for pair in repetitive_pairs:
                print(f"\nProcessing group: {[os.path.basename(f) for f in pair]}")
                keep_file, remove_files = process_repetitive_files(pair)
                # Remove the files that we don't want to keep
                for file_to_remove in remove_files:
                    try:
                        os.remove(file_to_remove)
                        print(f"Removed: {os.path.basename(file_to_remove)}")
                    except Exception as e:
                        print(f"Error removing {os.path.basename(file_to_remove)}: {e}")
    
    # Finally remove debug and bak files
    print("\nRemoving debug and bak files...")
    remove_debug_and_bak_files(base_dir)

def main():
    base_dir = "/Users/eveyhuang/Documents/NICO/gemini_code/outputs/2021MND"
    process_directory(base_dir)

if __name__ == "__main__":
    main() 