import os
import json
import re
import shutil
import sys

def fix_quotes_in_text(text):
    """Fix quotes in text content to ensure valid JSON."""
    # First try to extract JSON from code blocks if present
    if '```json' in text:
        # Split by code block markers and get the content
        parts = text.split('```')
        for part in parts:
            if part.strip().startswith('json'):
                # Remove the 'json' prefix and try to clean that content
                content = part.replace('json', '', 1).strip()
                try:
                    # Try to parse it directly first
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    # If that fails, continue with cleaning
                    pass
            elif len(part.strip()) > 0:
                try:
                    # Try to parse non-empty parts directly
                    json.loads(part.strip())
                    return part.strip()
                except json.JSONDecodeError:
                    pass

    # If we get here, either there were no code blocks or they weren't valid JSON
    # Remove any remaining code block markers and 'json' markers
    text = text.replace('```json', '').replace('```', '')
    
    # Fix various quote issues
    text = text.replace('"', '"').replace('"', '"')  # Fix curly quotes
    text = text.replace(''', "'").replace(''', "'")  # Fix curly apostrophes
    
    # Remove any outer JSON wrapper if it exists
    if text.strip().startswith('{"meeting_annotations":') and not text.strip().startswith('{"meeting_annotations":['):
        try:
            # Try to parse as JSON to extract inner content
            data = json.loads(text)
            if isinstance(data, dict) and 'meeting_annotations' in data:
                inner_content = data['meeting_annotations']
                if isinstance(inner_content, str):
                    text = inner_content.strip()
        except json.JSONDecodeError:
            pass

    # Clean up any trailing/leading whitespace
    text = text.strip()
    
    # Ensure the content is a JSON array if it contains meeting annotations
    if '"meeting_annotations":' in text and not text.strip().startswith('{"meeting_annotations":['):
        text = '{"meeting_annotations":' + text + '}'
    
    return text

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
        except json.JSONDecodeError as e:
            pass  # Continue with cleaning if parsing fails
        
        # Clean the content
        cleaned_content = fix_quotes_in_text(content)
        
        # Verify the cleaned content is valid JSON
        try:
            json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            print(f"Error cleaning {file_path}: {str(e)}")
            print("Original content:", content[:200])  # Print first 200 chars for debugging
            print("Cleaned content:", cleaned_content[:200])
            return False
        
        # Write the cleaned content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def process_directory(base_dir):
    """Process all JSON files in the directory"""
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json') and not (file.startswith('all_') or file.startswith('verbal_')):
                file_path = os.path.join(root, file)
                clean_json_file(file_path)

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