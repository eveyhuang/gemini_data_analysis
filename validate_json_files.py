import json
import os
from pathlib import Path
import unicodedata
import re
import argparse

def is_valid_json_file(file_path):
    """
    Check if a file contains valid and non-empty JSON.
    Returns True only if the file contains valid JSON and is not empty ({}).
    """
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
            # Check if the JSON is empty (just {})
            if content == {} or content == [] or not content:
                return False
            return True
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return False

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
    # return name

def validate_and_update_path_dict(path_dict_file, base_output_dir):
    """
    Validate JSON files referenced in path dict and update their status.
    Returns a new dictionary with updated validation status.
    """
    # Read the original path dictionary
    with open(path_dict_file, 'r') as f:
        path_dict = json.load(f)
    
    # Create a copy to modify
    updated_dict = path_dict.copy()
    
    # Track changes made
    changes_made = []
    
    # Process each key-value pair
    for key, file_lists in updated_dict.items():
        # Get the directory name from the key (split by / and take the last part)
        # dir_name = f"output-{sanitize_name(key)}"
        dir_name = f"output_{sanitize_name(key)}"
        
        
        for file_list in file_lists:
            # Get the base filename without extension
            base_name = os.path.splitext(file_list[0])[0]
            base_name = sanitize_name(base_name)
            
            # Construct the expected JSON file path
            # json_file = os.path.join(base_output_dir, dir_name, f"{base_name}.mp4.json")
            json_file = os.path.join(base_output_dir, dir_name, f"{base_name}.json")
            
            # Check if the JSON file exists and is valid
            if file_list[3]:  # Only check files marked as True
                if not is_valid_json_file(json_file):
                    # Update status to False
                    file_list[3] = False
                    changes_made.append({
                        'key': key,
                        'file': file_list[0],
                        'json_path': json_file
                    })
            else:
                if is_valid_json_file(json_file):
                    file_list[3] = True
                    changes_made.append({
                        'key': key,
                        'file': file_list[0],
                        'json_path': json_file
                    })
    # Print changes
    if changes_made:
        print("\nThe following files were found and changed analysis status:")
        for change in changes_made:
            print(f"Directory: {change['key']}")
            print(f"File: {change['file']}")
            print(f"JSON path: {change['json_path']}")
            print("-" * 80)
    else:
        print("\nNo new valid JSON files found among those marked as False.")
    
    return updated_dict

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Validate JSON files and update path dictionary.')
    parser.add_argument('path_dict_file', help='Path to the path dictionary JSON file')
    parser.add_argument('output_dir', help='Path to the output directory')
    
    args = parser.parse_args()
    
    # Extract base name from path_dict_file to construct output file name
    path_dict_base = os.path.splitext(os.path.basename(args.path_dict_file))[0]
    if path_dict_base.endswith('_path_dict'):
        base_name = path_dict_base[:-10]  # Remove '_path_dict' suffix
    else:
        base_name = path_dict_base
    output_file = f"{base_name}_path_dict_validated.json"
    
    # Get absolute paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    path_dict_path = os.path.join(workspace_dir, args.path_dict_file)
    base_output_dir = os.path.join(workspace_dir, args.output_dir)
    
    print(f"Validating JSON files...")
    print(f"Path dictionary: {path_dict_path}")
    print(f"Output directory: {base_output_dir}")
    
    # Validate and update the dictionary
    updated_dict = validate_and_update_path_dict(path_dict_path, base_output_dir)
    
    # Save the updated dictionary
    output_path = os.path.join(workspace_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(updated_dict, f, indent=4)
    
    print(f"\nUpdated path dictionary saved to: {output_file}")

if __name__ == "__main__":
    main() 