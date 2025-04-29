import json
import re

# Input and output file paths
input_path = 'outputs/2021MND/output_2021_04_23_MND_S2/bot1_2021_04_23_11_15_35/bot1_2021_04_23_11_15_35_chunk3.mp4.json'
output_path = 'outputs/2021MND/output_2021_04_23_MND_S2/bot1_2021_04_23_11_15_35_chunk3_fixed.json'

def clean_json_text(text):
    # Remove ```json and ```
    text = text.strip()
    if text.startswith('```json'):
        text = text[len('```json'):].strip()
    if text.endswith('```'):
        text = text[:-len('```')].strip()
    
    # Find JSON body inside {}
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx == -1 or end_idx == 0:
        raise ValueError("Could not find valid JSON object in text")
    
    text = text[start_idx:end_idx]
    
    # Remove stray line breaks and normalize spaces
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    
    # Fix missing commas between key-value pairs
    # This regex looks for a closing quote followed by whitespace and another quote
    text = re.sub(r'\"([^\"]*)\"\s+\"', r'\"\1\", \"', text)
    
    # Fix missing commas between array elements
    text = re.sub(r'}\s+{', r'}, {', text)
    
    # Fix missing comma before closing array bracket
    text = re.sub(r'}\s*\]', r'}]', text)
    
    # Fix missing comma before closing object bracket
    text = re.sub(r'}\s*}', r'}}', text)
    
    # Add comma before closing array bracket if missing
    if '}]' not in text and text.endswith('}'):
        text = text[:-1] + '}]'
    
    return text

try:
    # Step 1: Load the outer JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        outer_json = json.load(f)

    # Step 2: Extract the 'text' field
    raw_text = outer_json.get('text', '')
    if not raw_text:
        raise ValueError("No text field found in JSON")

    # Step 3: Clean the JSON text
    cleaned_text = clean_json_text(raw_text)
    
    # Debug: Print the last 100 characters of cleaned text
    print("Last 100 characters of cleaned text:")
    print(cleaned_text[-100:])
    
    # Step 4: Parse the cleaned JSON
    try:
        real_json = json.loads(cleaned_text)
        
        # Validate the structure
        if not isinstance(real_json, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        if 'meeting_annotations' not in real_json:
            raise ValueError("Missing 'meeting_annotations' field")
        
        # Step 5: Save the cleaned JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(real_json, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully cleaned and saved to {output_path}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Error position: {e.pos}")
        print(f"Context around error: {cleaned_text[max(0, e.pos-50):min(len(cleaned_text), e.pos+50)]}")
        print("\nFull cleaned text length:", len(cleaned_text))
        print("Last 200 characters of cleaned text:")
        print(cleaned_text[-200:])
        raise

except Exception as e:
    print(f"❌ Error processing file: {e}")
    raise