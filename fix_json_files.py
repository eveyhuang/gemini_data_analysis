import os
import json
import re
import shutil
import sys

def fix_quotes_in_text(text):
    """Fix quotes in text content to ensure valid JSON."""
    # First try to extract the meeting_annotations directly using regex
    try:
        # Remove escaped newlines and code block markers first
        text = text.replace('\\n', '')
        text = text.replace('```json', '').replace('```', '')
        
        # Extract the meeting_annotations array
        pattern = r'"meeting_annotations"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Wrap the content in a proper JSON structure
            full_json = f'{{"meeting_annotations": [{content}]}}'
            # Try to parse to validate
            parsed = json.loads(full_json)
            return json.dumps(parsed, indent=2)
    except Exception:
        pass

    # If that fails, try to parse the whole thing as JSON
    try:
        # Clean up the text first
        text = text.replace('\\n', '').replace('\n', '')  # Remove both escaped and real newlines
        text = text.replace('\\"', '"')  # Fix escaped quotes
        text = text.strip()
        
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if "text" in parsed:
                # If it's wrapped in a text field, try to parse that
                inner_text = parsed["text"]
                inner_text = inner_text.replace('```json', '').replace('```', '')
                try:
                    inner_parsed = json.loads(inner_text)
                    if "meeting_annotations" in inner_parsed:
                        return json.dumps(inner_parsed, indent=2)
                except json.JSONDecodeError:
                    pass
            elif "meeting_annotations" in parsed:
                return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass

    # If all attempts fail, wrap in an error structure
    error_dict = {
        "error": "Failed to parse JSON",
        "original_text": text[:200]  # Include first 200 chars of problematic text
    }
    return json.dumps(error_dict, indent=2)

def clean_json_file(file_path):
    """Clean a JSON file by creating a backup and fixing the content."""
    # Create backup
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as-is first
        try:
            json.loads(content)
            return True  # If it parses, we're done
        except json.JSONDecodeError:
            pass  # Continue with cleaning if parsing fails
        
        # Clean the content
        cleaned_content = fix_quotes_in_text(content)
        
        # Write the cleaned content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        # Verify the cleaned content is valid JSON
        try:
            json.loads(cleaned_content)
            print(f"Successfully cleaned {file_path}")
            return True
        except json.JSONDecodeError as e:
            print(f"Error: Cleaned content is still not valid JSON in {file_path}: {str(e)}")
            print("Original content:", content[:200])  # Print first 200 chars for debugging
            print("Cleaned content:", cleaned_content[:200])
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def process_directory(base_dir):
    """Process all JSON files in the directory and its subdirectories."""
    success_count = 0
    fail_count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json') and not (file.startswith('all_') or file.startswith('verbal_')):
                file_path = os.path.join(root, file)
                print(f"\nProcessing: {file_path}")
                if clean_json_file(file_path):
                    success_count += 1
                else:
                    fail_count += 1
    
    print(f"\nProcessing complete. Successfully cleaned {success_count} files. Failed to clean {fail_count} files.")

def main():
    """Main function to process JSON files."""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            clean_json_file(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        # Test with a specific file
        test_file = "outputs/2020NES/output-2020-11-06-NES-S3/2020-11-06_10-47-07/2020-11-06_10-47-07_chunk3.json"
        if os.path.isfile(test_file):
            print(f"Testing with file: {test_file}")
            clean_json_file(test_file)
        else:
            print("No file specified and test file not found")

if __name__ == '__main__':
    main() 