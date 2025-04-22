import os
import json
import shutil
from pathlib import Path

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

def merge_similar_folders(base_dir):
    """Find and merge folders that differ only by space/underscore."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Process each subdirectory
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        # Get all subdirectories within this directory
        try:
            subdirs = [d for d in subdir.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"Error accessing directory {subdir}: {e}")
            continue
        
        # Group directories by their normalized names (replacing spaces with underscores)
        dir_groups = {}
        for d in subdirs:
            normalized_name = d.name.replace(' ', '_')
            if normalized_name not in dir_groups:
                dir_groups[normalized_name] = []
            dir_groups[normalized_name].append(d)
        
        # Process groups with multiple directories
        for normalized_name, dirs in dir_groups.items():
            if len(dirs) > 1:
                print(f"\nFound similar folders for: {normalized_name}")
                # Find the target directory (the one with underscore)
                target_dir = next((d for d in dirs if '_' in d.name), None)
                if not target_dir:
                    # If no underscore version exists, use the first one
                    target_dir = dirs[0]
                
                print(f"Target directory: {target_dir.name}")
                
                # Move files from other directories to target
                for source_dir in dirs:
                    if source_dir != target_dir:
                        print(f"Moving files from {source_dir.name} to {target_dir.name}")
                        for file in source_dir.iterdir():
                            if file.is_file():
                                target_path = target_dir / file.name
                                if target_path.exists():
                                    print(f"  Warning: {file.name} already exists in target, skipping")
                                else:
                                    shutil.move(str(file), str(target_path))
                                    print(f"  Moved: {file.name}")
                        
                        # Remove the now-empty source directory
                        try:
                            source_dir.rmdir()
                            print(f"Removed empty directory: {source_dir.name}")
                        except Exception as e:
                            print(f"Error removing directory {source_dir.name}: {e}")

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
            normalized_base = base_part.replace(' ', '_')
            # Reconstruct the normalized key
            group_key = f"{prefix}{normalized_base}_chunk{chunk_part}"
        else:
            # For files without chunks, just normalize the whole name
            normalized_base = rest.replace(' ', '_')
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
    
    # First merge similar folders
    print("Merging similar folders...")
    merge_similar_folders(base_dir)
    
    # Then remove debug and bak files
    print("\nRemoving debug and bak files...")
    remove_debug_and_bak_files(base_dir)
    
    # Process each subdirectory for repetitive files
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            print(f"\nProcessing directory: {subdir}")
            repetitive_pairs = find_repetitive_files(subdir)
            
            if not repetitive_pairs:
                print("No repetitive files found.")
                continue
            
            print(f"Found {len(repetitive_pairs)} pairs of repetitive files.")
            
            for pair in repetitive_pairs:
                print(f"\nProcessing pair: {[os.path.basename(f) for f in pair]}")
                keep_file, remove_files = process_repetitive_files(pair)
                
                if keep_file:
                    print(f"Keeping: {os.path.basename(keep_file)}")
                    for file in remove_files:
                        try:
                            os.remove(file)
                            print(f"Removed: {os.path.basename(file)}")
                        except Exception as e:
                            print(f"Error removing {file}: {e}")
                else:
                    print("No valid files to keep in this pair.")
    
    # Finally, remove 'mp4' from remaining JSON filenames
    print("\nRemoving 'mp4' from JSON filenames...")
    remove_mp4_from_filenames(base_dir)

def main():
   
    base_dir = "/Users/eveyhuang/Documents/NICO/gemini_code/outputs/2020NES"
    process_directory(base_dir)

if __name__ == "__main__":
    main() 