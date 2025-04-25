import ffmpeg
import re
import time, json, os
from google import genai
from dotenv import load_dotenv
from google.genai.errors import ServerError
import subprocess
import unicodedata


def init():
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    ## behavior focused prompt
    prompt = """
    Objective:
    You are an expert in interaction analysis and team research. You are provided with recording of a zoom meeting between a group of scientists collaborating on novel ideas to address scientific challenges. 
    Your objective is to annotate behavior and verbal cues to help us understand this team's behavior and processes.
    Each time someone speaks in this video, provide the following structured annotation:
    (1) speaker: Full names of speaker.
    (2) timestamp: startiing and ending time of this person's speech in [MM:SS-MM:SS] format, before the speaker changes
    (3) transcript: Verbatim speech transcript. Remove filler words unless meaningful.
    (4) speaking duration: the total number of seconds the speaker talks in this segment
    (4) nods_others: Count of head nods from other participants during this speaker's turn.
    (5) smile_self: Percentage of time this speaker was smiling during their turn.
    (6) smile_other: Percentage of time at least one other person was smiling.
    (7) distracted_others: Number of people looking away or using their phone during this speaker's turn.
    (8) hand_gesture: what type of hand gesture did the speaker use? (Raising Hand, Pointing, Open Palms, Thumbs up, Crossed Arms, None)
    (9) interuption: Was this an interruption? (Yes/No) – if the speaker started talking before the previous one finished.
    (10) overlap: Was this turn overlapped? (Yes/No) – if another person spoke at the same time
    (11) screenshare: Did anyone share their screen during this segment? (Yes/No)
    (12) screenshare_content: If there was screenshare, summarize the content shared on the screen and changes made to the content within the segment in no more than 3 sentences. Otherwise, write "None".
    
    Notes:
    If uncertain about a label due to poor visibility, return [low confidence] next to the annotation.
    Ensure timestamps, speaker IDs, and behavior annotations are consistent throughout the video.
    For transcripts, remove filler words unless meaningful and return one long string without line breaks or paragraph breaks.s

    Input:
    A video recording of a zoom meeting among a team of scientists from diverse backgrounds engaged in a collaborative task.

    Output Format:
    Return a JSON object with the key 'meeting_annotations' and list of annotations as value.
    """

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

    return client, prompt, code_book

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

# get all the video files (scialog directory is categorized by folders of each conference)
def get_video_in_folders(directory):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []

    # Get all items in the given directory
    items = os.listdir(directory)
    
    # First check for folders
    folder_names = [f for f in items if os.path.isdir(os.path.join(directory, f))]
    
    # If folders exist, process videos in folders
    if folder_names:
        for folder in folder_names:
            folder_path = os.path.join(directory, folder)
            # Get all files in the current folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension.lower() in video_extensions:
                    video_files.append((os.path.join(folder_path, file), folder_path, f"{folder}/{file_name}", file))
    
    # If no folders or additional files exist in root directory, check for direct video files
    files_in_root = [f for f in items if os.path.isfile(os.path.join(directory, f))]
    for file in files_in_root:
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in video_extensions:
            # For files in root, use the file name as both folder and file identifier
            video_files.append((os.path.join(directory, file), directory, f"{file_name}", file))

    return video_files

# get videos files in a directory (used for split directory with splitted videos)
def get_videos(directory):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []

    # Get all files in the given directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in files:
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in video_extensions:
            if file_extension.lower() == '.mkv':
                # Check if MP4 version exists
                mp4_path = os.path.join(directory, file_name + '.mp4')
                if not os.path.exists(mp4_path):
                    # Convert MKV to MP4
                    converted_path = convert_mkv_to_mp4(os.path.join(directory, file))
                    if converted_path:
                        video_files.append(os.path.basename(converted_path))
                else:
                    # Use existing MP4 version
                    video_files.append(file_name + '.mp4')
            elif file_extension.lower() != '.mkv':  # Skip MKV files but include other video formats
                video_files.append(file)
        
    return video_files

# convert mkv to mp4 using ffmpeg
def convert_mkv_to_mp4(input_path):
    """
    Converts an MKV video file to MP4 using ffmpeg.

    Args:
        input_path (str): Path to the .mkv file

    Returns:
        str: Path to the converted .mp4 file
    """
    if not input_path.lower().endswith(".mkv"):
        raise ValueError("Input file must be an MKV file.")
    
    output_path = os.path.splitext(input_path)[0] + ".mp4"
    
    command = [
        "ffmpeg",
        "-i", input_path,
        "-codec", "copy",  # Copy without re-encoding
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return None

# Create or update the path dictionary with video file paths and their split chunks
def create_or_update_path_dict(directory, cur_dir):
    folder_name = os.path.basename(directory)
    path_dict_file = os.path.join(cur_dir, f"{folder_name}_path_dict.json")

    # Check if path_dict.json exists
    if os.path.exists(path_dict_file):
        # Load existing path_dict
        with open(path_dict_file, 'r') as f:
            path_dict = json.load(f)
    else:
        # Create a new path_dict
        path_dict = {}

    # Get list of video files in the directory
    video_files = get_video_in_folders(directory)
    
    # Track processed files to avoid duplicates
    processed_files = set()

    for video_tuple in video_files:
        video_path, folder_path, video_file_name, full_filename = video_tuple
        
        # Skip if we've already processed this file
        if video_path in processed_files:
            continue
            
        file_name, file_extension = os.path.splitext(full_filename)
        
        # Skip MKV files if MP4 version exists
        if file_extension.lower() == '.mkv':
            mp4_path = video_path.replace('.mkv', '.mp4')
            if os.path.exists(mp4_path):
                continue
                
        path_key_name = video_file_name
        
        # Get the split directory for this video file
        split_dir = os.path.join(folder_path, f"split-{file_name}")
        new_chunk_paths = []
        
        # if the split directory exists, get the list of chunk files and sort them by chunk number
        if os.path.exists(split_dir):
            # Get list of chunk files in the split directory
            chunk_files = get_videos(split_dir)
            # Filter out MKV files if MP4 version exists
            chunk_files = [f for f in chunk_files if not (f.endswith('.mkv') and os.path.exists(f.replace('.mkv', '.mp4')))]
            
            # Remove duplicates while preserving order
            chunk_files = list(dict.fromkeys(chunk_files))
            
            # Sort chunk files by chunk number
            chunk_files.sort(key=lambda x: int(x.split('chunk')[1].split('.')[0]))
            
            # Create list of [chunk file name, full path to this video, gemini upload file name, analysis status] for each chunk file
            # Use a set to track paths we've already added
            added_paths = set()
            for chunk_file in chunk_files:
                chunk_path = os.path.join(split_dir, chunk_file)
                if chunk_path not in added_paths:
                    new_chunk_paths.append([chunk_file, chunk_path, ' ', False])
                    added_paths.add(chunk_path)
        else:
            # if the split directory does not exist, means that the video has not been split (length is short)
            # Convert MKV to MP4 if needed
            if video_path.endswith('.mkv'):
                mp4_path = video_path.replace('.mkv', '.mp4')
                if not os.path.exists(mp4_path):
                    print(f"Converting {video_path} to mp4")
                    video_path = convert_mkv_to_mp4(video_path)
                else:
                    print(f"Found {video_path} already converted to mp4")
                    video_path = mp4_path
                    
            new_chunk_paths = [[full_filename, video_path, ' ', False]]

        # Update path_dict, preserving analysis status if it exists
        if path_key_name in path_dict:
            old_chunk_paths = path_dict[path_key_name]
            # Create a map of existing analysis status using the full path as key
            status_map = {path[1]: (path[2], path[3]) for path in old_chunk_paths}
            
            # Update new paths with existing status if available
            for chunk_path in new_chunk_paths:
                if chunk_path[1] in status_map:
                    chunk_path[2], chunk_path[3] = status_map[chunk_path[1]]
                    
        # Remove any duplicates in new_chunk_paths while preserving order
        seen_paths = set()
        unique_chunk_paths = []
        for chunk_path in new_chunk_paths:
            if chunk_path[1] not in seen_paths:
                unique_chunk_paths.append(chunk_path)
                seen_paths.add(chunk_path[1])
                
        path_dict[path_key_name] = unique_chunk_paths
        processed_files.add(video_path)
        
    return path_dict

# Split a video into chunks of specified length
def split_video(video_full_path, duration, chunk_length=10*60):
    
    # Calculate the number of chunks
    num_chunks = int(duration // chunk_length) + 1
    
    # Get the file name and directory
    file_name, file_extension = os.path.splitext(os.path.basename(video_full_path))

    # if the file is in mkv format, convert it to mp4 using ffmpeg
    if file_extension == '.mkv':
        video_full_path = convert_mkv_to_mp4(video_full_path)

    directory = os.path.dirname(video_full_path)
    
    # Create a directory to store the split videos
    split_dir = os.path.join(directory, f"split-{file_name}")
    os.makedirs(split_dir, exist_ok=True)
    
    # Split the video into chunks
    chunk_paths = []
    for i in range(num_chunks):
        start_time = i * chunk_length
        output_file_name = os.path.join(split_dir, f"{file_name}_chunk{i+1}{file_extension}")
        ffmpeg.input(video_full_path, ss=start_time, t=chunk_length).output(output_file_name).run()
        chunk_paths.append(output_file_name)
        print(f"Created chunk: {output_file_name}")
    
    return chunk_paths

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

# Process all videos in a directory, splitting them if necessary
def process_videos_in_directory(directory):
    
    # path to video, path_to_folder, video file name for path_dict, original filename
    video_files = get_video_in_folders(directory)

    split_videos_dict = {}
    
    for video_file in video_files:
        video_full_path = video_file[0]
        file_name, file_extension = os.path.splitext(video_file[3])
        split_dir = os.path.join(directory, f"split-{file_name}")
        # print(f"Split directory is {split_dir}")
        if not os.path.exists(split_dir):
            try:
                probe = ffmpeg.probe(video_full_path)
                duration = float(probe['format']['duration'])
                if duration > 10 * 60:
                    print(f"Splitting video: {video_file}")
                    split_videos_dict[video_file] = split_video(video_full_path, duration)
                else:
                    print(f"Video {video_file} is shorter than 10 minutes, no need to split.")
                    split_videos_dict[video_file] = [video_full_path]
            except Exception as e:
                print(f"Having issues with video: {video_full_path}")
                print(f"Here is the exception: {e}")
            
        else:
            print(f"Found a folder with splitted videios for {video_file} already.")
    
    return split_videos_dict

# Get the list of files on gemini
def safe_list_files(client):
    try:
        # Get the list of files
        files = client.files.list()
        
        # Filter out any files with problematic names
        safe_files = []
        for file in files:
            try:
                # Use our sanitization function
                safe_name = sanitize_filename(file.name)
                safe_files.append(safe_name)
            except Exception as e:
                # Skip files with problematic names
                print(f"Skipping file {file} with problematic name: {e}")
                continue
                
        return safe_files
    except Exception as e:
        print(f"When getting all files on gemini, received unexpected error: {e}")
        return []  # Return empty list instead of None

# Sanitize a filename to ensure it contains only ASCII characters
def sanitize_filename(name):
    """
    Sanitize a filename to ensure it contains only ASCII characters.
    Replace or remove problematic characters.
    """
    if name is None:
        return None
        
    try:
        # Try to encode the name to ASCII, replacing non-ASCII chars
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        return safe_name
    except Exception as e:
        print(f"Error sanitizing filename {name}: {e}")
        
    
# Return the video file from Gemini, uploading it if necessary
def get_gemini_video(client, file_name, file_path, gemini_name):
    """Return the video file from Gemini, uploading it if necessary"""
    video_file = None
    
    # Sanitize the gemini_name if it exists
    safe_gemini_name = sanitize_filename(gemini_name)
    
    # If gemini_name is not empty or just whitespace, try to get the file
    if safe_gemini_name and safe_gemini_name.strip():
        try:

            gemini_file = client.files.get(name=safe_gemini_name)
            if gemini_file:
                print(f"{file_name} already uploaded to Gemini, returning that...")
                return gemini_file, safe_gemini_name
        except Exception as e:
            print(f"Could not get file with name {safe_gemini_name}: {e}")
    
    # If we couldn't get the file, upload it
    print(f"Uploading {file_name} to Gemini")
    try:
        safe_file_path = sanitize_filename(file_path)
        video_file = client.files.upload(file=safe_file_path)
        print(f"Completed upload: {video_file.uri}")  
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video_file = client.files.get(name=video_file.name)
            safe_gemini_name = sanitize_filename(video_file.name)
        if video_file.state.name == "FAILED":
            print(f"File processing failed for {file_name}")
    except Exception as e:
        print(f"When processing {file_name} encountered the following error: {e}")
        
    return video_file, safe_gemini_name


# Analyze a video using the Gemini API
def gemini_analyze_video(client, prompt, video_file, filename, max_tries = 3, delay=1):
    print(f"Making LLM inference request for {filename}...")
    for attempt in range(max_tries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',  # Using the generally available model
                contents=[prompt, video_file],
                config={
                    'temperature':0,
                    'max_output_tokens':8192,      
                },)
            print("Got a response! ...")
            return response
            
        except Exception as e:
            print(f"Error in making LLM request for {filename}, trying again... ")
            if attempt < max_tries-1:
                time.sleep(delay)
            else:
                print(f"Couldn't get response even after three tries: {filename}. Error: {e}")
                return None

# Analyze all videos in the path dictionary
def analyze_video(client, path_dict, prompt, dir):
    cur_dir = os.getcwd()
    n_path_dict = path_dict.copy()
    folder_name = os.path.basename(dir)
    
    # Create the base outputs directory
    base_output_dir = os.path.join(cur_dir, "outputs", folder_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    for file_name in n_path_dict.keys():
        list_chunks = n_path_dict[file_name]
        # Sanitize the output directory name
        safe_file_name = sanitize_name(file_name)
        output_dir = os.path.join(base_output_dir, f"output_{safe_file_name}")
        os.makedirs(output_dir, exist_ok=True)

        for m in range(len(list_chunks)):
            # [chunk file name, full path to this video, gemini upload file name, analysis status]
            file_name = list_chunks[m][0]
            fileName, file_extension = os.path.splitext(file_name)
            file_path = list_chunks[m][1]
            gemini_name = list_chunks[m][2]
            analyzed = list_chunks[m][3]
            
            if not analyzed:
                print(f"Analyzing {file_name}")
                video_file, gemini_name = get_gemini_video(client, file_name, file_path, gemini_name)
                list_chunks[m][2] = gemini_name
                if video_file:
                    response = gemini_analyze_video(client, prompt, video_file, file_name)
                    if response:
                        print(f"Trying to save output for {file_name} to json file")
                        try:
                            # Remove trailing slash and use os.path.join
                            save_to_json(response.text, file_name, output_dir)
                            list_chunks[m][3] = True
                        except ValueError:
                            response = gemini_analyze_video(client, prompt, video_file, file_name)
                            if response:
                                try:
                                    save_to_json(response.text, fileName, output_dir)
                                    list_chunks[m][3] = True
                                except Exception as e:
                                    print(f"Still can't get the output to workout: {file_name}")
                                    list_chunks[m][3] = False
                                    continue
                            else: 
                                list_chunks[m][3] = False
                    else:
                        list_chunks[m][3] = False
                save_path_dict(n_path_dict, f"{folder_name}_path_dict.json", cur_dir)
            else:
                print(f"{file_name} already analyzed, moving on..")
                continue
    print("Analysis all finished. Returning updated path_dict.")
    
    return n_path_dict

# Save data to a JSON file in the specified directory
def save_to_json(text, file_name, output_dir):
    """
    Save the analysis text to a JSON file, handling various response formats
    
    Args:
        text (str): The text to save (can be JSON or plain text)
        file_name (str): The name of the file being analyzed
        output_dir (str): The directory to save the output to
    """
    # Ensure the output directory exists
    output_dir = sanitize_name(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = sanitize_name(file_name)
    # Construct the output file path using os.path.join
    output_file = os.path.join(output_dir, f"{file_name}.json")
    
    # Clean the input text
    text = text.strip()  # Remove any leading/trailing whitespace
    
    # Handle code block markers
    if '```' in text:
        # Split by ``` and look for the JSON content
        blocks = text.split('```')
        for block in blocks:
            # Skip empty blocks
            if not block.strip():
                continue
            # If block starts with 'json', remove it
            block = block.strip().removeprefix('json').strip()
            try:
                # Try to parse this block as JSON
                parsed_json = json.loads(block)
                # If we successfully parsed JSON, use this block
                text = block
                break
            except json.JSONDecodeError:
                continue
    
    # Try to parse as JSON first
    try:
        parsed_json = json.loads(text)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved JSON to {output_file}")
        return
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Attempting to fix the JSON format...")
        # Attempt to fix the JSON format by removing any trailing characters
        text_fixed = text[:e.pos]
        try:
            parsed_json = json.loads(text_fixed)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved fixed JSON to {output_file}")
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON format: {e}")
            print("Saving as plain text wrapped in a JSON object")
            # Fall back to saving as plain text in a JSON wrapper
            output_dict = {"text": text}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=4)

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
        return chunk_num if chunk_num is not None else float('inf')
    
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
        raise InvalidJsonContentError(f"No valid JSON files with chunk numbers found in {directory}")
    
    return result

# Merge output JSON files into a single list of annotations
def merge_output_json(directory):
    data_list = load_json_files(directory)
    num_chunks = len(data_list)
    full_output = []
    utterance_list=[]

    for m in range(num_chunks):
        ck = data_list[m]
        if ck:
            for key in ck.keys():
                d_list = ck[key]
                
                for i in d_list:
                    data = i.copy()
                    sp_time = data['timestamp'].split('-')
                    if len(sp_time)==2:
                        data['start_time'] = add_times(sp_time[0], str(10*m)+':00')
                        data['end_time'] = add_times(sp_time[1], str(10*m)+':00')
                    else:
                        data['start_time'] = add_times(sp_time[0], str(10*m)+':00')
                        
                    full_output.append(data)
                    utterance_list.append(f"{data['speaker']}: {data['transcript']} ")
        else:
            print(f"Having issues with data in directory: {directory}")
                
    return (full_output, utterance_list)

# Annotate and merge the output JSON files
def annotate_and_merge(client, path_dict, directory, codebook):
    for key in path_dict.keys():
        split_vid_name = key.split("/")
        if len(split_vid_name)>1:
            vid_name = split_vid_name[1]
        else:
            vid_name = split_vid_name[0]

        output_subfolder = os.path.join(directory, f"output-{key}")
        verbal_file = os.path.join(output_subfolder, f"verbal_{vid_name}.json")
        all_file = os.path.join(output_subfolder,f"all_{vid_name}.json" )
        
        if os.path.exists(output_subfolder):
            try:
                merged_output = merge_output_json(output_subfolder)
            except InvalidJsonContentError as e:
                print(f"Skipping {output_subfolder} due to invalid JSON content: {str(e)}")
                continue
                
            if not os.path.exists(all_file):
                verbal_annotations = []
                if not os.path.exists(verbal_file):
                    print(f"Annotating verbal behvaiors for {key}")
                    annotations = annotate_utterances(client, merged_output[1],codebook)
                    verbal_annotations = annotations
                    output_file = verbal_file
                    with open(output_file, "w") as f:
                        json.dump(annotations, f, indent=4)
                else:
                    print(f"{verbal_file} already exists, loading that to merge...")
                    with open(verbal_file, 'r') as f:
                        verbal_annotations = json.load(f)
                    
                if len(verbal_annotations)== len(merged_output[0]):
                    all_anno = merged_output[0].copy()
                    for i in range(len(verbal_annotations)):
                        anno = verbal_annotations[i]['annotations']
                        all_anno[i]['annotations'] = anno
                    print(f"Merged verbal annotations with existing video annotations.")
                    with open(all_file, "w") as f:
                        json.dump(all_anno, f, indent=4)
            else:
                print(f"{all_file} already exists, skipping...")
        else:
            print(f"{output_subfolder} does not exist, skipping...")


def main(vid_dir, process_video):

    folder_name = os.path.basename(vid_dir)
    client, prompt, codebook = init()
    cur_dir = os.getcwd()
    if process_video == 'yes':
        process_videos_in_directory(vid_dir)
    path_dict = create_or_update_path_dict(vid_dir, cur_dir)
    save_path_dict(path_dict, f"{folder_name}_path_dict.json", cur_dir)

    new_path_dict = analyze_video(client, path_dict, prompt, vid_dir)
    save_path_dict(new_path_dict, f"{folder_name}_path_dict.json", cur_dir)

    
    return path_dict


if __name__ == '__main__':
    dir = input("Please provide the FULL PATH to the directory where videos are stored (do NOT wrap it in quotes): ")
    process_video = input("Process video? yes/no ")
    main(dir, process_video)