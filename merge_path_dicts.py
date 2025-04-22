import json
import os

def normalize_string(s):
    """Replace spaces with underscores in a string"""
    return s.replace(' ', '_')

def remove_duplicates(file_lists):
    """Remove duplicate entries from a list of file entries based on normalized filenames"""
    seen = {}
    unique_lists = []
    
    for file_list in file_lists:
        normalized_name = normalize_string(file_list[0])
        # If we've seen this name before and the current entry has True in position 3,
        # replace the old entry
        if normalized_name in seen:
            if file_list[3] and not unique_lists[seen[normalized_name]][3]:
                unique_lists[seen[normalized_name]] = file_list
        else:
            seen[normalized_name] = len(unique_lists)
            unique_lists.append(file_list)
    
    return unique_lists

def merge_path_dicts(original_file, copy_file, output_file):
    """
    Merge two path dictionary files with specific rules:
    1. For existing keys:
       - If list[0] matches (accounting for space/underscore differences), keep version with true in list[3]
       - If list[0] differs only in extension, keep mp4 over mkv
       - Otherwise add new list[0] entries
    2. For new keys, add directly
    3. Remove any duplicate entries in the process
    """
    # Read original file
    with open(original_file, 'r') as f:
        original_dict = json.load(f)
    
    # Read copy file
    with open(copy_file, 'r') as f:
        copy_dict = json.load(f)
    
    # Create merged dictionary starting with original, removing any duplicates
    merged_dict = {key: remove_duplicates(value) for key, value in original_dict.items()}
    
    # Create normalized key mapping for original dict
    normalized_keys = {normalize_string(key): key for key in merged_dict.keys()}
    
    # Process each key-value pair from copy
    for copy_key, copy_lists in copy_dict.items():
        normalized_copy_key = normalize_string(copy_key)
        # Remove duplicates from copy lists first
        unique_copy_lists = remove_duplicates(copy_lists)
        
        if normalized_copy_key in normalized_keys:
            # Key exists in original (after normalization), need to merge lists
            original_key = normalized_keys[normalized_copy_key]
            original_lists = merged_dict[original_key]
            
            # Create a map of normalized list[0] to full list for original entries
            original_map = {}
            for lst in original_lists:
                name = lst[0]
                normalized_name = normalize_string(name)
                base_name = normalized_name.rsplit('.', 1)[0]  # Remove extension
                original_map[normalized_name] = lst
                original_map[base_name] = lst  # Also map without extension
            
            # Process each list from copy
            for copy_list in unique_copy_lists:
                copy_name = copy_list[0]
                normalized_copy_name = normalize_string(copy_name)
                copy_base = normalized_copy_name.rsplit('.', 1)[0]  # Remove extension
                copy_ext = copy_name.rsplit('.', 1)[1] if '.' in copy_name else ''
                
                if normalized_copy_name in original_map:
                    # Exact match (after normalization) - keep version with true in list[3]
                    orig_list = original_map[normalized_copy_name]
                    if copy_list[3] and not orig_list[3]:
                        # Replace with copy version
                        print(f"File {copy_name} exists in original but not analyzed, replacing with copy version")
                        idx = original_lists.index(orig_list)
                        original_lists[idx] = copy_list
                elif copy_base in original_map:
                    # Base name matches - check extensions
                    orig_list = original_map[copy_base]
                    orig_ext = orig_list[0].rsplit('.', 1)[1]
                    
                    # Keep mp4 over mkv
                    if orig_ext == 'mkv' and copy_ext == 'mp4':
                        idx = original_lists.index(orig_list)
                        original_lists[idx] = copy_list
                else:
                    # No match - add new entry
                    print(f"File {copy_name} doesn't exist in original, adding new entry")
                    original_lists.append(copy_list)
            
            # Remove any duplicates that might have been introduced during merging
            merged_dict[original_key] = remove_duplicates(original_lists)
        else:
            # Key doesn't exist in original, add directly (after removing duplicates)
            print(f"Key {copy_key} doesn't exist in original, adding directly")
            merged_dict[copy_key] = unique_copy_lists
    
    # Write merged dictionary to new file
    with open(output_file, 'w') as f:
        json.dump(merged_dict, f, indent=4)

def main():
    original_file = "2020NES_path_dict.json"
    copy_file = "2020NES_path_dict copy.json"
    output_file = "2020NES_path_dict_merged.json"
    
    merge_path_dicts(original_file, copy_file, output_file)
    print(f"Merged dictionary saved to {output_file}")

if __name__ == "__main__":
    main() 