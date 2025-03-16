import ffmpeg
import re
import time, json, os
from google import genai
from dotenv import load_dotenv


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
    (1) speaker: Initials of speaker.
    (2) timestamp: startiing and ending time of this person's speech in [MM:SS-MM:SS] format, before the speaker changes
    (3) transcript: Verbatim speech transcript. Remove filler words unless meaningful.
    (4) speaking duration: the total number of seconds the speaker talks in this segment
    (4) nods_others: Count of head nods from other participants during this speaker’s turn.
    (5) smile_self: Percentage of time this speaker was smiling during their turn.
    (6) smile_other: Percentage of time at least one other person was smiling.
    (7) distracted_others: Number of people looking away or using their phone during this speaker’s turn.
    (8) hand_gesture: what type of hand gesture did the speaker use? (Raising Hand, Pointing, Open Palms, Thumbs up, Crossed Arms, None)
    (9) interuption: Was this an interruption? (Yes/No) – if the speaker started talking before the previous one finished.
    (10) overlap: Was this turn overlapped? (Yes/No) – if another person spoke at the same time
    (11) screenshare: Did anyone share their screen during this segment? (Yes/No)
    (12) screenshare_content: If there was screenshare, summarize the content shared on the screen and changes made to the content within the segment in no more than 3 sentences. Otherwise, write "None".
    
    Notes:
    If uncertain about a label due to poor visibility, return [low confidence] next to the annotation.
    Ensure timestamps, speaker IDs, and behavior annotations are consistent throughout the video.

    Input:
    A video recording of a zoom meeting among a team of scientists from diverse backgrounds engaged in a collaborative task.

    Output Format:
    Return a JSON object with the key 'meeting_annotations' and list of annotations as value.
    """

    code_book = """
        (1) present new idea: introduces a novel concept, approach, or hypothesis not previously mentioned. Example: "“What if we used reinforcement learning instead?”
        (2) expand on existing idea: builds on a previously mentioned idea, adding details, variations, or improvements. Example: “Yes, and if we train it with synthetic data, we might improve accuracy.”
        (3) provide supporting evidence: provides data, references, or logical reasoning to strengthen an idea. Example: “A recent Nature paper shows that this method outperforms others.”
        (4) explain or define term or concept: explains a concept, method, or terminology for clarity. Example: “When I say ‘feature selection,’ I mean choosing the most important variables.”
        (5) ask clarifying question: requests explanation or elaboration on a prior statement. Example:“Can you explain what you mean by ‘latent variable modeling’?”
        (6) propose decision: suggests a concrete choice for the group. “I think we should prioritize dataset A.”
        (7) confirm decision: explicitly agrees with a proposed decision, finalizing it. Example: “Yes, let’s go with that approach.”
        (9) express alternative decision: rejects a prior decision and suggests another. Example: “Instead of dataset A, we should use dataset B because it has more variability.”
        (10) express agreement: explicitely agrees with proposed idea or decision. Example: "I agree with your approach."
        (11) assign task: assigns responsibility for a task to a team member. Example: "Alex, can you handle data processing?"
        (12) offer constructive criticism: critiques with the intent to improve. Example: "This model has too many parameters, maybe we can prune them."
        (13) reject idea: dismisses or rejects an idea but does not offer a new one or ways to improve. "I don't think that will work"
        (14) resolve conflict: mediates between opposing ideas to reach concensus. "we can test both and compare the results."
        (15) express frustation: verbalizes irritation, impatience, or dissatisfaction. "We have been going in circles on this issue."
        (16) acknowledge contribution: verbally recognizes another person’s input, but not agreeing or expanding. "That is a great point."
        (17) encourage particpatioin: invites someone else to contribute. "Alex, what do you think?"
        (18) express enthusiasm: expresses excitement, optimism, or encouragement. "This is an exciting direction!"
        (19) express humor: makes a joke or laughs. example: "well, at least this model isn't as bad as our last one!"
    """

    return client, prompt, code_book

# Save the path dictionary to a JSON file
def save_path_dict(path_dict, file_name, destdir):
    with open(f"{destdir}/{file_name}", 'w') as json_file:
        json.dump(path_dict, json_file, indent=4)

# get all the video files (scialog directory is categorized by folders of each conference)
def get_video_in_folders(directory):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []

    # Get all folders in the given directory
    folder_names = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    for folder in folder_names:
        folder_path = os.path.join(directory, folder)
        # Get all files in the current folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                file_name, file_extension = os.path.splitext(file)
                # path to video, path_to_folder, video file name for path_dict, original filename
                video_files.append((os.path.join(folder_path, file), folder_path, f"{folder}/{file_name}", file))

    return video_files

# get videos files in a directory (used for split directory with splitted videos)
def get_videos(directory):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []

    # Get all files in the given directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in files:
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(file)

    return video_files

# Create or update the path dictionary with video file paths and their split chunks
def create_or_update_path_dict(directory, cur_dir):
    path_dict_file = os.path.join(cur_dir, "path_dict.json")
    
    # Check if path_dict.json exists
    if os.path.exists(path_dict_file):
        # Load existing path_dict
        with open(path_dict_file, 'r') as f:
            path_dict = json.load(f)
    else:
        # Create a new path_dict
        path_dict = {}

    # Get list of video files in the directory
     # path to video, path_to_folder, video file name for path_dict, original filename
    video_files = get_video_in_folders(directory)

    for video_file in video_files:
        file_name, file_extension = os.path.splitext(video_file[3])
        path_key_name = video_file[2]
        folder_dir = video_file[1]
        if path_key_name not in path_dict.keys():
            # Get the split directory for this video file
            split_dir = os.path.join(folder_dir, f"split-{file_name}")
            # print(f"Split directory is {split_dir}")
            if os.path.exists(split_dir):
                # Get list of chunk files in the split directory
                chunk_files = get_videos(split_dir)
                chunk_files.sort(key=lambda x: int(x.split('chunk')[1].split('.')[0]))  # Sort chunk files by chunk number
                
                # Create list of [chunk name, full path to this video, gemini upload file name, analysis status] for each chunk file
                chunk_paths = [[chunk_file, os.path.join(split_dir, chunk_file), ' ', False] for chunk_file in chunk_files]
                
                # Add to path_dict
                path_dict[path_key_name] = chunk_paths
    # print(f"updated path dict: {path_dict}")
    return path_dict

# Split a video into chunks of specified length
def split_video(video_full_path, duration, chunk_length=10*60):
    
    # Calculate the number of chunks
    num_chunks = int(duration // chunk_length) + 1
    
    # Get the file name and directory
    file_name, file_extension = os.path.splitext(os.path.basename(video_full_path))
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

# Return the video file from Gemini, uploading it if necessary
def get_gemini_video(client, file_name, file_path, gemini_name):
    # files that have already been uploaded to gemini
    existing_files = [x.name for x in client.files.list()]
    video_file = None
    gemini_name = ''
    if gemini_name in existing_files:
        print(f"{file_name} already uploaded to Gemini, returning that...")
        video_file = client.files.get(name=gemini_name)
    else:
        print(f"Uploading {file_name} to Gemini")
        try:
            video_file = client.files.upload(file=file_path)
            print(f"Completed upload: {video_file.uri}")  
            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(10)
                video_file = client.files.get(name=video_file.name)
                gemini_name = video_file.name
            if video_file.state.name == "FAILED":
                print("File processing failed.")
        except Exception as e:
            print(f"File processing failed for {file_name}")
            
    return video_file, gemini_name

# Analyze a video using the Gemini API
def gemini_analyze_video(client, prompt, video_file, filename, max_tries = 3, delay=1):
    print(f"Making LLM inference request for {filename}...")
    for attempt in range(max_tries):
        try:
            response = client.models.generate_content(
                model='gemini-1.5-pro',
                contents=[prompt, video_file],
                config={
                    'response_mime_type':'application/json',
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
                print(f"Couldn't get response even after three tries: {filename}")
                return None

# Analyze all videos in the path dictionary
def analyze_video(client, path_dict, prompt, dir):
    cur_dir = os.getcwd()
    n_path_dict = path_dict.copy()
    for file_name in n_path_dict.keys():
        list_chunks = n_path_dict[file_name]
        output_dir = f"{cur_dir}/output-{file_name}"
        os.makedirs(output_dir, exist_ok=True)
        for m in range(len(list_chunks)):
            file_name = list_chunks[m][0]
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
                            save_to_json(response.text, file_name, f"{output_dir}/")
                            list_chunks[m][3] = True
                        except ValueError:
                            response = gemini_analyze_video(client, prompt, video_file, file_name)
                            if response:
                                save_to_json(response.text, file_name, f"{output_dir}/")
                                list_chunks[m][3] = True
                            else: 
                                list_chunks[m][3] = False
                    else:
                        list_chunks[m][3] = False
                save_path_dict(n_path_dict, "path_dict.json", cur_dir)
            else:
                print(f"{file_name} already analyzed, moving on..")
                continue
    print("Analysis all finished. Returning updated path_dict.")
    # print(f"Updated path dict: {n_path_dict}")
    return n_path_dict

# Save data to a JSON file in the specified directory
def save_to_json(response, filename, destdir):
    output_file_path= destdir + filename+".json"
    jres = response

    # Ensure the response text is properly formatted JSON
    try:
        parsed_json = json.loads(jres)
        with open(output_file_path, 'w') as json_file:
            json.dump(parsed_json, json_file, indent=4)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Attempting to fix the JSON format...")
        # Attempt to fix the JSON format by removing any trailing characters
        jres_fixed = jres[:e.pos]
        try:
            parsed_json = json.loads(jres_fixed)
            with open(output_file_path, 'w') as json_file:
                json.dump(parsed_json, json_file, indent=4)
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON format: {e}")
            print("Saving the original response text to the file for manual inspection.")
            with open(output_file_path, 'w') as json_file:
                json_file.write(jres)
            raise ValueError("Could not fix the error; try to run the prompt again.")

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
            model='gemini-1.5-pro',
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

# Load JSON files from a directory and sort them by chunk number
def load_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json') and not (f.startswith('all') or f.startswith('verbal'))]
    json_files.sort(key=lambda x: int(re.search(r'chunk(\d+)', x).group(1)))
    
    result = []
    for json_file in json_files:
        with open(os.path.join(directory, json_file)) as f:
            data = json.load(f)
            chunk_number = extract_chunk_number(json_file)

            while len(result) < chunk_number:
                result.append(None)
            result[chunk_number-1] = data
    
    return result

# Merge output JSON files into a single list of annotations
def merge_output_json(directory):
    data_list = load_json_files(directory)
    num_chunks = len(data_list)
    full_output = []
    utterance_list=[]

    for m in range(num_chunks):
        ck = data_list[m]
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
                
    return (full_output, utterance_list)

# Annotate and merge the output JSON files
def annotate_and_merge(client, path_dict, directory, codebook):
    for key in path_dict.keys():
        output_subfolder = os.path.join(directory, f"output-{key}")
        verbal_file = os.path.join(output_subfolder, f"verbal_{key}.json")
        all_file = os.path.join(output_subfolder,f"all_{key}.json" )
        if os.path.exists(output_subfolder):
            merged_output = merge_output_json(output_subfolder)
            if not os.path.exists(all_file):
                verbal_annotations = []
                if not os.path.exists(verbal_file):
                    print(f"Annotating verbal behvaiors for {key}")
                    annotations = annotate_utterances(client, merged_output[1],codebook)
                    verbal_annotations = annotations
                    output_file = f"{output_subfolder}/verbal_{key}.json"
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


def main(vid_dir):
    client, prompt, codebook = init()
    cur_dir = os.getcwd()
    process_videos_in_directory(vid_dir)
    path_dict = create_or_update_path_dict(vid_dir, cur_dir)
    save_path_dict(path_dict, "path_dict.json", cur_dir)

    new_path_dict = analyze_video(client, path_dict, prompt, vid_dir)
    save_path_dict(new_path_dict, "path_dict.json", cur_dir)
    annotate_and_merge(client, new_path_dict, cur_dir, codebook)
    return new_path_dict

if __name__ == '__main__':
    dir = input("Please provide the FULL PATH to the directory where videos are stored (do NOT wrap it in quotes): ")
    main(dir)