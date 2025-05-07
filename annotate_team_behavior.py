import ffmpeg
import re
import time, json, os
from google import genai
from dotenv import load_dotenv
from google.genai.errors import ServerError
import subprocess
import unicodedata
import argparse


def init():
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    code_book = """
        (1) present new idea: introduces a novel concept, approach, or hypothesis not previously mentioned. Example: "What if we used reinforcement learning instead?"
        (2) expand on existing idea: builds on a previously mentioned idea, adding details, variations, or improvements. Example: "Yes, and if we train it with synthetic data, we might improve accuracy."
        (3) provide supporting evidence: provides data, references, or logical reasoning to strengthen an idea. Example: "A recent Nature paper shows that this method outperforms others."
        (4) explain or define term or concept: explains a concept, method, or terminology for clarity. Example: "When I say 'feature selection,' I mean choosing the most important variables."
        (5) ask clarifying question: requests explanation or elaboration on a prior statement. Example:"Can you explain what you mean by 'latent variable modeling'?"
        (6) propose decision: suggests a concrete choice for the group. "I think we should prioritize dataset A."
        (7) confirm decision: explicitly agrees with a proposed decision, finalizing it. Example: "Yes, let's go with that approach."
        (9) express alternative decision: rejects a prior decision and suggests another. Example: "Instead of dataset A, we should use dataset B because it has more variability."
        (10) express agreement: explicitely agrees with proposed idea or decision. Example: "I agree with your approach."
        (11) assign task: assigns responsibility for a task to a team member. Example: "Alex, can you handle data processing?"
        (12) offer constructive criticism: critiques with the intent to improve. Example: "This model has too many parameters, maybe we can prune them."
        (13) reject idea: dismisses or rejects an idea but does not offer a new one or ways to improve. "I don't think that will work"
        (14) resolve conflict: mediates between opposing ideas to reach concensus. "we can test both and compare the results."
        (15) express frustation: verbalizes irritation, impatience, or dissatisfaction. "We have been going in circles on this issue."
        (16) acknowledge contribution: verbally recognizes another person's input, but not agreeing or expanding. "That is a great point."
        (17) encourage particpatioin: invites someone else to contribute. "Alex, what do you think?"
        (18) express enthusiasm: expresses excitement, optimism, or encouragement. "This is an exciting direction!"
        (19) express humor: makes a joke or laughs. example: "well, at least this model isn't as bad as our last one!"
    """

    return client, code_book

# Save the path dictionary to a JSON file
def save_path_dict(path_dict, file_name, destdir):
    with open(f"{destdir}/{file_name}", 'w') as json_file:
        json.dump(path_dict, json_file, indent=4)




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

# save path dict file
def save_path_dict(path_dict, file_name, destdir):
    with open(f"{destdir}/{file_name}", 'w') as json_file:
        json.dump(path_dict, json_file, indent=4)

# Annotate utterances with Gemini
def annotate_utterances(client, merged_list, codebook):
    """
    Iterates over each utterance, using the full list as context.
    Returns structured annotations in JSON format.
    """
    # print(f"Annotating {len(merged_list)} utterances")
    annotations = []  # Store annotations for all utterances

    for i, utterance in enumerate(merged_list):
        # Prepare the prompt
        comm_prompt = f"""
        You are an expert in interaction analysis, team science, and qualitative coding. Your task is to analyze an 
        utterance from a scientific collaboration meeting and annotate it using the codes in the provided codebook.

        **Annotation Guidelines:**
        - If the utterance has multiple sentences, assign one most relevant code for each sentence. If no code applies, write 'None'.
        - For each code you choose, provide:
          1. **Code Name** 
          2. **Explanation**: Justify why this code applies in 1 sentence, using relevant prior context.
        - Ensure that **each annotation follows the structured JSON format**.

        ** Codebook for Annotation:**
        {codebook}

        ** Full Conversation Context:**
        {json.dumps(merged_list[:i+1], indent=2)}  # Include past utterances up to the current one

        ** Utterance to Annotate:**
        "{utterance}"

        **Expected Output:**
        Output a structured JSON file where each coded category is a key and the explanation is the value. If there are multiple explanations for this code, summarize them into one coherent sentence.
        Do not include codes that are not relevant to this utterance in your output.
        """

        # Call Gemini API (adjust depending on your implementation)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[comm_prompt],
            config={
                'response_mime_type':'application/json',
                'temperature':0,
                'max_output_tokens':8192,      
            },)
        annotation = json.loads(response.text)  # Parse response as JSON

        annotations.append({
            "utterance": utterance,
            "annotations": annotation  # Store structured annotation
        })

    return annotations  # Return all annotations

# Extract the chunk number from a string
def extract_chunk_number(string):
    match = re.search(r'chunk(\d+)', string)
    if match:
        return int(match.group(1))
    else:
        return None

# Add two time strings in HH:MM format
def add_times(time1, time2):
    # Split the time strings into hours and minutes
    hours1, minutes1 = map(int, time1.split(':'))
    hours2, minutes2 = map(int, time2.split(':'))
    
    # Add the hours and minutes separately
    total_hours = hours1 + hours2
    total_minutes = minutes1 + minutes2
    
    # If total minutes are 60 or more, convert to hours
    if total_minutes >= 60:
        total_hours += total_minutes // 60
        total_minutes = total_minutes % 60
    
    # Format the result as "HH:MM"
    return f"{total_hours:02}:{total_minutes:02}"

class InvalidJsonContentError(Exception):
    """Exception raised when JSON file is invalid or empty."""
    pass

# Load JSON files from a directory and sort them by chunk number
def load_json_files(directory):
    # Get all JSON files except those starting with 'all' or 'verbal'
    json_files = [f for f in os.listdir(directory) if f.endswith('.json') and not (f.startswith('all') or f.startswith('verbal'))]
    
    if not json_files:
        raise InvalidJsonContentError(f"No JSON files found in directory: {directory}")
    
    # Sort files by chunk number, handling files without chunk numbers
    def sort_key(filename):
        chunk_num = extract_chunk_number(filename)
        return chunk_num if chunk_num is not None else 1
    
    json_files.sort(key=sort_key)
    
    result = []
    invalid_files = []
    
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path) as f:
                data = json.load(f)
                # Check if JSON is empty or has no content
                if data == {} or data == [] or not data:
                    invalid_files.append(f"{json_file} (empty content)")
                    continue
                
                chunk_number = extract_chunk_number(json_file)
                
                # Only process files with valid chunk numbers
                if chunk_number is not None:
                    # Ensure the result list is long enough
                    while len(result) < chunk_number:
                        result.append(None)
                    result[chunk_number-1] = data
                else:
                    result.append(data)
                
        except json.JSONDecodeError:
            invalid_files.append(f"{json_file} (invalid JSON)")
        except Exception as e:
            invalid_files.append(f"{json_file} (error: {str(e)})")
    
    if invalid_files:
        raise InvalidJsonContentError(
            f"Found invalid or empty JSON files in {directory}:\n" + 
            "\n".join(f"- {f}" for f in invalid_files)
        )
    
    if not result:
        raise InvalidJsonContentError(f"No valid JSON files found in {directory}")
    
    return result

def get_json_subfolder(output_folder):
    """Find the subdirectories containing JSON files."""
    list_of_folders = []
    try:
        # List all items in the output folder
        items = os.listdir(output_folder)
        # Look for a directory that might contain JSON files
        for item in items:
            item_path = os.path.join(output_folder, item)
            if os.path.isdir(item_path):
                # Check if this directory contains any JSON files
                if any(f.endswith('.json') for f in os.listdir(item_path)):
                    list_of_folders.append(item_path)
        return list_of_folders
    except Exception as e:
        print(f"Error accessing directory {output_folder}: {e}")
        return None

def merge_output_json(output_folder):
    """
    Merge output JSON files from the correct subdirectory.
    Args:
        output_folder: Path to the output folder containing a subdirectory with JSON files
    """
    # Find the subdirectory containing JSON files
    
    # Now load and process JSON files from the correct subdirectory
    data_list = load_json_files(output_folder)
    # print(f"Merging {len(data_list)} chunks in {output_folder}")
    num_chunks = len(data_list)
    full_output = []
    utterance_list = []

    for m in range(num_chunks):
        ck = data_list[m]
        if ck:
            for key in ck.keys():
                d_list = ck[key]
                
                for i in d_list:
                    data = i.copy()
                    # Remove brackets from timestamp if present
                    try:
                        timestamp = data['timestamp'].strip('[]')
                        sp_time = timestamp.split('-')
                        if len(sp_time)==2:
                            data['start_time'] = add_times(sp_time[0], str(10*m)+':00')
                            data['end_time'] = add_times(sp_time[1], str(10*m)+':00')
                        else:
                            data['start_time'] = add_times(sp_time[0], str(10*m)+':00')
                            
                        full_output.append(data)
                        utterance_list.append(f"{data['speaker']}: {data['transcript']} ")
                    except KeyError:
                        print(f"KeyError in {output_folder} for chunk {m}. Original data: {data}")
                        continue
        else:
            print(f"Having issues with data in directory: {output_folder}")
                
    return (full_output, utterance_list)

def get_output_folders(base_dir):
    """Get all output folders in the base directory."""
    output_folders = []
    try:
        for item in os.listdir(base_dir):
            if item.startswith('output_') or item.startswith('output-'):
                full_path = os.path.join(base_dir, item)
                if os.path.isdir(full_path):
                    output_folders.append(full_path)
    except Exception as e:
        print(f"Error accessing directory {base_dir}: {e}")
    return output_folders


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

def annotate_and_merge_outputs(client, output_dir, codebook):
    """
    Annotate and merge outputs for all subfolders in the output directory.
    Args:
        client: Gemini API client
        output_dir: Base directory containing output folders
        codebook: Codebook for annotation
    """
    output_folders = get_output_folders(output_dir)
    if not output_folders:
        # already in an output folder
        # check if the files in output_dir are json
        if any(f.endswith('.json') for f in os.listdir(output_dir)):
            print(f"Found JSON files in {output_dir}, processing...")
            output_folders = [output_dir]
        else:
            print(f"No JSON files found in {output_dir}, skipping...")
            return

    print(f"Found {len(output_folders)} output folders to process")
    
    for output_subfolder in output_folders:
        folder_name = os.path.basename(output_subfolder)
        print(f"\nProcessing sub-folder: {folder_name}")
        
        # Get the subdirectory containing JSON files
        json_dir = get_json_subfolder(output_subfolder)
        if not json_dir:
            if any(f.endswith('.json') for f in os.listdir(output_subfolder)):
                json_dir = [output_subfolder]
            else:
                print(f"No JSON files found in {output_subfolder}, skipping...")
                continue

        for folder in json_dir:    
            # Get the base name for the JSON files from the json_dir name
            json_dir_name = os.path.basename(folder)
            
            # Create verbal and all files in the JSON directory
            verbal_file = os.path.join(folder, f"verbal_{json_dir_name}.json")
            all_file = os.path.join(folder, f"all_{json_dir_name}.json")
            
            if not is_valid_json_file(verbal_file):
                print(f"No existing/valid verbal file in {folder}, annotating now...")
                try:
                    merged_output = merge_output_json(folder)
                except InvalidJsonContentError as e:
                    print(f"Skipping {json_dir} due to invalid JSON content: {str(e)}")
                    continue
                    
                verbal_annotations = []
                
                # print(f"Annotating verbal behaviors for {json_dir_name}")
                annotations = annotate_utterances(client, merged_output[1], codebook)
                verbal_annotations = annotations
                output_file = verbal_file
                with open(output_file, "w") as f:
                    print(f"Saved verbal annotations to {output_file}")
                    json.dump(annotations, f, indent=4)
                
                    
                if len(verbal_annotations) == len(merged_output[0]):
                    all_anno = merged_output[0].copy()
                    for i in range(len(verbal_annotations)):
                        anno = verbal_annotations[i]['annotations']
                        all_anno[i]['annotations'] = anno
                    
                    with open(all_file, "w") as f:
                        print(f"Merged verbal annotations with existing video annotations to {all_file}")
                        json.dump(all_anno, f, indent=4)
                
            else:
                print(f"Already annotated files in {folder}, skipping...")
        
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Annotate and merge team behavior analysis outputs.')
    parser.add_argument('output_dir', help='Path to the directory containing output folders')
    
    args = parser.parse_args()
    
    # Initialize client and codebook
    client, codebook = init()
    
    # Process the output directory
    print(f"Processing outputs in: {args.output_dir}")
    annotate_and_merge_outputs(client, args.output_dir, codebook)
    print("\nAnnotation and merging complete!")

if __name__ == '__main__':
    main()