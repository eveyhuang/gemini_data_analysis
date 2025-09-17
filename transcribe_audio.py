import ffmpeg
import re
import time, json, os
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from dotenv import load_dotenv
import subprocess
import unicodedata
import argparse


def init():
    """Initialize Deepgram client"""
    load_dotenv()
    DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
    
    client = DeepgramClient(DEEPGRAM_API_KEY)
    return client


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
                    if file_extension.lower() == '.mkv':
                        # convert to mp4 first if it is a mkv file
                        mp4_path = os.path.join(folder_path, file_name + '.mp4')
                        if not os.path.exists(mp4_path):
                            mp4_path = convert_mkv_to_mp4(os.path.join(folder_path, file))
                        video_files.append((mp4_path, folder_path, f"{folder}/{file_name}", file))
                    else:
                        video_files.append((os.path.join(folder_path, file), folder_path, f"{folder}/{file_name}", file))
    
    # If no folders or additional files exist in root directory, check for direct video files
    files_in_root = [f for f in items if os.path.isfile(os.path.join(directory, f))]
    for file in files_in_root:
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in video_extensions:
            if file_extension.lower() == '.mkv':
                # convert to mp4 first if it is a mkv file
                mp4_path = os.path.join(directory, file_name + '.mp4')
                if not os.path.exists(mp4_path):
                    mp4_path = convert_mkv_to_mp4(os.path.join(directory, file))
                video_files.append((mp4_path, directory, f"{file_name}", file))
            else:
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
            else:  # Skip MKV files but include other video formats
                video_files.append(file)
        
    return video_files


# convert mkv to mp4 using ffmpeg
def convert_mkv_to_mp4(input_path):
    """
    Converts an MKV video file to MP4 using ffmpeg.
    Re-encodes audio to AAC to ensure compatibility.
    """
    import subprocess
    import os

    if not input_path.lower().endswith(".mkv"):
        raise ValueError("Input file must be an MKV file.")
    
    output_path = os.path.splitext(input_path)[0] + ".mp4"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",        # Copy video stream
        "-c:a", "aac",         # Re-encode audio to AAC
        "-b:a", "192k",        # Set audio bitrate (optional)
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return None


# Extract audio from video file
def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path: Path to the input video file
        audio_path: Path where the audio file should be saved
        
    Returns:
        Path to the extracted audio file if successful, None otherwise
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Extract audio using ffmpeg
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='pcm_s16le',  # Use PCM 16-bit little-endian for better compatibility
            ar=16000,            # Sample rate 16kHz (good for speech)
            ac=1                 # Mono channel
        ).overwrite_output().run()
        
        print(f"Audio extracted successfully: {audio_path}")
        return audio_path
        
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
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
        try:
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
                
                # Create list of [chunk file name, full path to this video, audio file path, transcription status] for each chunk file
                # Use a set to track paths we've already added
                added_paths = set()
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(split_dir, chunk_file)
                    if chunk_path not in added_paths:
                        # Create audio file path
                        audio_file_name = os.path.splitext(chunk_file)[0] + '.wav'
                        audio_path = os.path.join(split_dir, 'audio', audio_file_name)
                        new_chunk_paths.append([chunk_file, chunk_path, audio_path, False])
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
                        
                # Create audio file path for unsplit video
                audio_file_name = os.path.splitext(full_filename)[0] + '.wav'
                audio_path = os.path.join(folder_path, 'audio', audio_file_name)
                new_chunk_paths = [[full_filename, video_path, audio_path, False]]

            # Update path_dict, preserving transcription status if it exists
            if path_key_name in path_dict:
                old_chunk_paths = path_dict[path_key_name]
                # Create a map of existing transcription status using the full path as key
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

        except Exception as e:
            print(f"Error processing video file {video_file_name}: {e}")
            continue
        
    return path_dict


# Split a video into chunks of specified length
def split_video(video_full_path, duration, chunk_length=10*60):
    
    # Calculate the number of chunks
    num_chunks = int(duration // chunk_length) + 1
    
    # Get the file name and directory
    file_name, _ = os.path.splitext(os.path.basename(video_full_path))
    directory = os.path.dirname(video_full_path)
    
    # Create a directory to store the split videos
    split_dir = os.path.join(directory, f"split-{file_name}")
    os.makedirs(split_dir, exist_ok=True)
    
    # Split the video into chunks
    chunk_paths = []
    for i in range(num_chunks):
        start_time = i * chunk_length
        output_file_name = os.path.join(split_dir, f"{file_name}_chunk{i+1}.mp4")
        # Add -n flag to skip existing files
        ffmpeg.input(video_full_path, ss=start_time, t=chunk_length).output(output_file_name, n=None).run()
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
        if video_full_path:
            # Check if the file is MKV and needs conversion
            if file_extension.lower() == '.mkv':
                mp4_path = video_full_path.replace('.mkv', '.mp4')
                if not os.path.exists(mp4_path):
                    print(f"Converting MKV to MP4: {video_full_path}")
                    video_full_path = convert_mkv_to_mp4(video_full_path)
                else:
                    video_full_path = mp4_path
            
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
                print(f"Found a folder with splitted videos for {video_file} already.")
    
    return split_videos_dict


# Transcribe audio using Deepgram API
def transcribe_audio(client, audio_path, max_tries=3, delay=1):
    """
    Transcribe audio file using Deepgram API
    
    Args:
        client: Deepgram client instance
        audio_path: Path to the audio file
        max_tries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Transcription result or None if failed
    """
    print(f"Transcribing audio: {audio_path}")
    
    for attempt in range(max_tries):
        try:
            with open(audio_path, 'rb') as audio:
                buffer_data = audio.read()
            
            payload: FileSource = {
                "buffer": buffer_data,
            }
            
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                punctuate=True,
                diarize=True,  # Enable speaker diarization
                utterances=True,  # Get utterances with timestamps
                language="en"
            )
            
            # Use the new REST API instead of deprecated prerecorded
            response = client.listen.rest.v("1").transcribe_file(payload, options)
            print("Transcription completed successfully!")
            return response
            
        except Exception as e:
            print(f"Error transcribing {audio_path} due to error {e}, trying again...")
            if attempt < max_tries - 1:
                time.sleep(delay)
            else:
                print(f"Couldn't transcribe even after {max_tries} tries: {audio_path}. Error: {e}")
                return None


# Process all audio files in the path dictionary
def transcribe_audio_files(client, path_dict, dir):
    cur_dir = os.getcwd()
    n_path_dict = path_dict.copy()
    folder_name = os.path.basename(dir)
    
    # Create the base outputs directory
    base_output_dir = os.path.join(cur_dir, "outputs", f"NEW_{folder_name}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    for file_name in n_path_dict.keys():
        list_chunks = n_path_dict[file_name]
        # Sanitize the output directory name
        safe_file_name = sanitize_name(file_name)
        output_dir = os.path.join(base_output_dir, f"output_{safe_file_name}")
        os.makedirs(output_dir, exist_ok=True)

        for m in range(len(list_chunks)):
            # [chunk file name, full path to this video, audio file path, transcription status]
            chunk_file_name = list_chunks[m][0]
            fileName, file_extension = os.path.splitext(chunk_file_name)
            video_path = list_chunks[m][1]
            audio_path = list_chunks[m][2]
            transcribed = list_chunks[m][3]
            
            if not transcribed:
                print(f"Processing {chunk_file_name}")
                
                # Extract audio if it doesn't exist
                if not os.path.exists(audio_path):
                    print(f"Extracting audio from {chunk_file_name}")
                    extracted_audio = extract_audio_from_video(video_path, audio_path)
                    if not extracted_audio:
                        print(f"Failed to extract audio from {chunk_file_name}")
                        list_chunks[m][3] = False
                        continue
                
                # Transcribe audio
                response = transcribe_audio(client, audio_path)
                if response:
                    print(f"Transcription response: {response}")
                    print(f"Saving transcription for {chunk_file_name}")
                    try:
                        save_transcription_to_json(response, fileName, output_dir)
                        list_chunks[m][3] = True
                    except Exception as e:
                        print(f"Error saving transcription for {chunk_file_name}: {e}")
                        list_chunks[m][3] = False
                else:
                    print(f"No transcription response for {chunk_file_name}")
                    list_chunks[m][3] = False
                    
                save_path_dict(n_path_dict, f"{folder_name}_path_dict.json", cur_dir)
            else:
                print(f"{chunk_file_name} already transcribed, moving on..")
                continue
                
    print("Transcription finished. Returning updated path_dict.")
    return n_path_dict


# Save transcription data to a JSON file
def save_transcription_to_json(transcription_response, file_name, output_dir):
    """
    Save the transcription response to a JSON file in the meeting_annotations format
    
    Args:
        transcription_response: Deepgram transcription response
        file_name: The name of the file being transcribed
        output_dir: The directory to save the output to
    """
    # Ensure the output directory exists
    output_dir = sanitize_name(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = sanitize_name(file_name)
    # Construct the output file path using os.path.join
    output_file = os.path.join(output_dir, f"{file_name}.json")

    try:
        # The new Deepgram API returns a dictionary directly
        if hasattr(transcription_response, 'to_dict'):
            # If it's a response object with to_dict method
            transcription_data = transcription_response.to_dict()
        elif isinstance(transcription_response, dict):
            # If it's already a dictionary
            transcription_data = transcription_response
        else:
            # Try to access the response data directly
            transcription_data = {
                "metadata": getattr(transcription_response, 'metadata', {}),
                "results": getattr(transcription_response, 'results', {})
            }
        
        # Process the transcription data into meeting_annotations format
        meeting_annotations = []
        
        # Process each channel (usually just one for mono audio)
        for channel in transcription_data.get('results', {}).get('channels', []):
            for alternative in channel.get('alternatives', []):
                # Group words by speaker and create utterances
                current_speaker = None
                current_utterance = []
                current_start = None
                current_end = None
                
                for word in alternative.get('words', []):
                    # Handle different speaker field names
                    speaker = word.get('speaker', word.get('speaker_label', 'Unknown'))
                    
                    # If speaker changes or this is the first word, start a new utterance
                    if current_speaker != speaker and current_utterance:
                        # Save the previous utterance
                        utterance_data = {
                            'speaker': current_speaker,
                            'timestamp': f"{format_time(current_start)}-{format_time(current_end)}" if current_start and current_end else "00:00-00:00",
                            'transcript': ' '.join(current_utterance),
                            'speaking_duration': int(current_end - current_start) if current_start and current_end else 0
                        }
                        meeting_annotations.append(utterance_data)
                        
                        # Reset for new utterance
                        current_utterance = []
                        current_start = None
                        current_end = None
                    
                    # Add word to current utterance
                    current_utterance.append(word['word'])
                    current_speaker = speaker
                    
                    # Update timing
                    if current_start is None:
                        current_start = word['start']
                    current_end = word['end']
                
                # Don't forget the last utterance
                if current_utterance:
                    utterance_data = {
                        'speaker': current_speaker,
                        'timestamp': f"{format_time(current_start)}-{format_time(current_end)}" if current_start and current_end else "00:00-00:00",
                        'transcript': ' '.join(current_utterance),
                        'speaking_duration': int(current_end - current_start) if current_start and current_end else 0
                    }
                    meeting_annotations.append(utterance_data)
        
        # Create the final output structure
        output_data = {
            "meeting_annotations": meeting_annotations
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved transcription to {output_file}")
        
    except Exception as e:
        print(f"Error processing transcription data: {e}")
        print(f"Response type: {type(transcription_response)}")
        print(f"Response attributes: {dir(transcription_response) if hasattr(transcription_response, '__dict__') else 'No attributes'}")
        
        # Save raw response as fallback
        output_file = os.path.join(output_dir, f"ATTN_{file_name}.json")
        try:
            # Try to convert to dict if possible
            if hasattr(transcription_response, 'to_dict'):
                raw_data = transcription_response.to_dict()
            elif hasattr(transcription_response, '__dict__'):
                raw_data = transcription_response.__dict__
            else:
                raw_data = str(transcription_response)
            
            with open(output_file, 'w') as json_file:
                json.dump(raw_data, json_file, indent=4, default=str)
        except Exception as save_error:
            print(f"Error saving fallback file: {save_error}")
            with open(output_file, 'w') as json_file:
                json_file.write(str(transcription_response))


def format_time(seconds):
    """
    Convert seconds to MM:SS format
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        Formatted time string in MM:SS format
    """
    if seconds is None:
        return "00:00"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


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


# Merge transcription JSON files into a single list of utterances
def merge_transcription_json(directory):
    """
    Merge transcription JSON files into a single list of utterances with speaker information
    """
    data_list = load_json_files(directory)
    num_chunks = len(data_list)
    full_output = []
    utterance_list = []

    for m in range(num_chunks):
        ck = data_list[m]
        if ck:
            # Check if this is the new meeting_annotations format
            if 'meeting_annotations' in ck:
                # Process meeting_annotations format
                for annotation in ck['meeting_annotations']:
                    # Adjust timestamps for chunk offset (each chunk is 10 minutes)
                    chunk_offset_minutes = 10 * m
                    start_time = annotation.get('timestamp', '00:00-00:00').split('-')[0]
                    end_time = annotation.get('timestamp', '00:00-00:00').split('-')[1] if '-' in annotation.get('timestamp', '00:00-00:00') else start_time
                    
                    # Add chunk offset to timestamps
                    adjusted_start = add_times(start_time, f"{chunk_offset_minutes}:00")
                    adjusted_end = add_times(end_time, f"{chunk_offset_minutes}:00")
                    
                    utterance_data = {
                        'speaker': annotation.get('speaker', 'Unknown'),
                        'transcript': annotation.get('transcript', ''),
                        'start_time': adjusted_start,
                        'end_time': adjusted_end,
                        'timestamp': f"{adjusted_start}-{adjusted_end}",
                        'speaking_duration': annotation.get('speaking_duration', 0)
                    }
                    full_output.append(utterance_data)
                    utterance_list.append(f"{annotation.get('speaker', 'Unknown')}: {annotation.get('transcript', '')}")
            else:
                # Fallback to old format processing (for backward compatibility)
                for channel in ck.get('results', {}).get('channels', []):
                    for alternative in channel.get('alternatives', []):
                        # Group words by speaker and create utterances
                        current_speaker = None
                        current_utterance = []
                        current_start = None
                        current_end = None
                        
                        for word in alternative.get('words', []):
                            # Handle different speaker field names
                            speaker = word.get('speaker', word.get('speaker_label', 'Unknown'))
                            
                            # If speaker changes or this is the first word, start a new utterance
                            if current_speaker != speaker and current_utterance:
                                # Save the previous utterance
                                utterance_data = {
                                    'speaker': current_speaker,
                                    'transcript': ' '.join(current_utterance),
                                    'start_time': current_start,
                                    'end_time': current_end,
                                    'timestamp': f"{current_start}-{current_end}" if current_start and current_end else "00:00-00:00",
                                    'speaking_duration': current_end - current_start if current_start and current_end else 0
                                }
                                full_output.append(utterance_data)
                                utterance_list.append(f"{current_speaker}: {' '.join(current_utterance)}")
                                
                                # Reset for new utterance
                                current_utterance = []
                                current_start = None
                                current_end = None
                            
                            # Add word to current utterance
                            current_utterance.append(word['word'])
                            current_speaker = speaker
                            
                            # Update timing
                            if current_start is None:
                                current_start = word['start']
                            current_end = word['end']
                        
                        # Don't forget the last utterance
                        if current_utterance:
                            utterance_data = {
                                'speaker': current_speaker,
                                'transcript': ' '.join(current_utterance),
                                'start_time': current_start,
                                'end_time': current_end,
                                'timestamp': f"{current_start}-{current_end}" if current_start and current_end else "00:00-00:00",
                                'speaking_duration': current_end - current_start if current_start and current_end else 0
                            }
                            full_output.append(utterance_data)
                            utterance_list.append(f"{current_speaker}: {' '.join(current_utterance)}")
        else:
            print(f"Having issues with data in directory: {directory}")
                
    return (full_output, utterance_list)


def main(vid_dir, process_video, transcribe_audio_flag):
    folder_name = os.path.basename(vid_dir)
    client = init()
    cur_dir = os.getcwd()
    
    if process_video == 'yes':
        process_videos_in_directory(vid_dir)
    
    path_dict = create_or_update_path_dict(vid_dir, cur_dir)
    save_path_dict(path_dict, f"{folder_name}_path_dict.json", cur_dir)
    
    if transcribe_audio_flag == 'yes':
        path_dict = transcribe_audio_files(client, path_dict, vid_dir)
        save_path_dict(path_dict, f"{folder_name}_path_dict.json", cur_dir)

    return path_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio and transcribe videos using Deepgram API')
    parser.add_argument('--dir', required=True, 
                       help='Full path to the directory where videos are stored')
    parser.add_argument('--process-video', choices=['yes', 'no'], required=True,
                       help='Whether to process video (yes/no)')
    parser.add_argument('--transcribe-audio', choices=['yes', 'no'], required=True,
                       help='Whether to transcribe audio (yes/no)')
    
    args = parser.parse_args()
    
    main(args.dir, args.process_video, args.transcribe_audio)
