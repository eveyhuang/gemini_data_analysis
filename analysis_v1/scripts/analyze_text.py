import json
import os
import time
from google import genai
from dotenv import load_dotenv
import unicodedata
import re
from datetime import datetime

def init(prompt_type='scialog'):
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    ## behavior focused prompt
    scialog_prompt = """
    Objective:
    You are an expert in interaction analysis and team research. You are provided with a transcript of a meeting between a group of scientists collaborating on novel ideas to address scientific challenges. 
    Your objective is to annotate behavior and verbal cues to help us understand this team's behavior and processes.
    Each time someone speaks in this transcript, provide the following structured annotation:
    speaker: Full names of speaker.
    timestamp: startiing and ending time of this person's speech in MM:SS-MM:SS format, before the speaker changes
    transcript: Verbatim speech transcript. Remove filler words unless meaningful.
    speaking duration: the total number of seconds the speaker talks in this segment
    nods_others: Count of head nods from other participants during this speaker's turn.
    smile_self: Percentage of time this speaker was smiling during their turn.
    smile_other: Percentage of time at least one other person was smiling.
    distracted_others: Number of people looking away or using their phone during this speaker's turn.
    hand_gesture: what type of hand gesture did the speaker use? (Raising Hand, Pointing, Open Palms, Thumbs up, Crossed Arms, None)
    interuption: Was this an interruption? (Yes/No) – if the speaker started talking before the previous one finished.
    overlap: Was this turn overlapped? (Yes/No) – if another person spoke at the same time
    screenshare: Did anyone share their screen during this segment? (Yes/No)
    screenshare_content: If there was screenshare, summarize the content shared on the screen and changes made to the content within the segment in no more than 3 sentences. Otherwise, write "None".
    
    Notes:
    If uncertain about a label due to poor visibility, return [low confidence] next to the annotation.
    Ensure timestamps, speaker IDs, and behavior annotations are consistent throughout the transcript.
    For transcripts, remove filler words unless meaningful and return one long string without line breaks or paragraph breaks.

    Input:
    A transcript of a meeting among a team of scientists from diverse backgrounds engaged in a collaborative task.

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
        (15) express frustration: verbalizes irritation, impatience, or dissatisfaction. "We have been going in circles on this issue."
        (16) acknowledge contribution: verbally recognizes another person's input, but not agreeing or expanding. "That is a great point."
        (17) encourage participation: invites someone else to contribute. "Alex, what do you think?"
        (18) express enthusiasm: expresses excitement, optimism, or encouragement. "This is an exciting direction!"
        (19) express humor: makes a joke or laughs. example: "well, at least this model isn't as bad as our last one!"
    """

    covid_prompt = """
    Objective:
    You are provided with a transcript of a conversation between three people: a study administrator, a conversationpartner (name in video as '1045'), and a participant (name in video with five characters0.)
    Your objective is to analyze the transcript carefully, especially pay attention to the participant's behavior and code using the codebook provided below:
    (1) technology: The extent to which the participant experiences technical difficulties (e.g., audio cutting out, video cutting out, sound issues, etc.).  Use 1-7 scale: 1 = no difficulties, 7 = extreme difficulties.
    (2) cur_neg_affect: The extent to which the participant discloses negative affect about the pandemic/masks/vaccine/boosters in the present. To what degree does s/he exhibit or disclose negative affective states such as being: angry, frustrated, sad, etc. Use 1-7 scale: 1 = not at all negative, 4 = somewhat negative, 7 = extremely negative. Examples: 1 = living life as it was before/normal, 2 = booster symptoms, 3 = COVID is still here/wearing mask, 4 = negative societal implications, 5 = political comments about lockdown, 6/7 = super sick, hospitalized, family member die, etc.
    (3) past_neg_affect: The extent to which the participant discloses feeling negative affect about the pandemic/masks/vaccine/boosters when recalling the past. To what degree does s/he describe feeling negative affective states such as being: angry, frustrated, sad, uneasy, uncertain, hesitant, etc. 1-7 scale: 1 = not at all negative, 4 = somewhat negative, 7 = extremely negative. Examples: 3 - 4 = missing graduation, prom, etc.
    (4) communal_orientation: The extent to which the participant displays behaviors or makes statements that reflect a desire for social connectedness with their conversation partner (e.g., being kind, showing concern and being sympathetic to partner's needs). Use 1-7 scale: 1 = not at all, 7 = extremely communal. Examples:  1 - 2 = Just answering the questions, cold in responses, not wanting to talk to partner beyond the time; 3, 4, 5 = talking and asking questions but not giving affirmations, doesn't seem entirely engaged; 6 - 7 = building off of story conversation partner told, trying to relate to partner.
    (5) engagement: The extent to which the participant is engaged in the conversation (e.g., throwing themselves into the conversation and genuinely conversing with their partner, trying to keep the conversation going, etc.). 1-7 scale: 1 = disengaged, 7 = engaged
    (6) defensiveness: The extent to which the participant is defensive in justifying their beliefs/perspective. (e.g., "How could you say that…", "I don't think you understand what I'm saying . . .", etc. 1-7 scale: 1 = not at all defensive, 7 = very defensive
    (7) questions: all questions that participants asks their partner throughout the conversation.
    (8) perspective_statements: Statements that participant says with an attempt to understand the other person's perspective (i.e., engages in perspective taking, or understanding of the others' thoughts and/or emotions) with their partner (e.g., using language such as: "I know how you feel…", "I imagine that feels. . .", "I hear what you are saying . . .", "I have never experienced that but I could imagine that…", etc.).
    (9) similarity_statements: Statement that participant says to acknowledge some sort of similarity between themselves and their partner (e.g., saying things such as: "I also had a bad reaction to the vaccine", "I feel the same way", "That happened to me too", or otherwise attempting to relate to the other by pointing out common ground such as "Oh you're in Texas? My best friend lives in Texas.").
    (10) vaccine_importance: At the very end of the conversation, to what extent does the participant believe that vaccines are important. 1-7 scale: 1 = not at all important, 4 = somewhat important, 7 = extremely important, -1 = unclear or not discussed at all
    (11) booster_importance: At the very end of the conversation, to what extent does the participant believe that boosters are important. same 1-7 scale as (10).
    (12) cognitive_complexity: To what extent during the conversation does the participant demonstrate cognitive complexity (i.e., consideration, analysis, recognizing nuance, playing with different ideas, and/or understanding the various perspectives in the conversation). 1-7 scale: 1 = not at all, 4 = moderate amount, 7 = very much. 
    (13) general_notes: any general notes about this conversation that stuck out to you.
    
    Notes:
    If you are ever unclear about a code, use -1 rather than forcing a code.
    For code 1-6 and 10-12, also include a one-sentence explanation of your rating. 
    For 7-9, include the time stamp of each transcript in the list. 

    Input:
    A transcript of a conversation between two people having a conversation about COVID. 
    
    Output Format:
    Return a JSON object with each of the code name as key.
    For code 1-6 and 10-12, a list of rating and a one-sentence explanation as value.
    For code 7-9, a list of transcripts and time stamp for each transcript as value.
    """

    # Select prompt based on prompt_type
    if prompt_type.lower() == 'covid':
        prompt = covid_prompt
    else:  # default to scialog
        prompt = scialog_prompt

    return client, prompt, code_book

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

def parse_vtt_time(time_str):
    """Convert VTT timestamp to MM:SS format."""
    try:
        # Parse the VTT timestamp (HH:MM:SS.mmm)
        dt = datetime.strptime(time_str.strip(), '%H:%M:%S.%f')
        return f"{dt.minute:02d}:{dt.second:02d}"
    except ValueError:
        try:
            # Try without milliseconds
            dt = datetime.strptime(time_str.strip(), '%H:%M:%S')
            return f"{dt.minute:02d}:{dt.second:02d}"
        except ValueError:
            return "00:00"

def parse_vtt_file(file_path):
    """Parse a VTT file and return a formatted transcript."""
    transcript = []
    current_speaker = None
    current_text = []
    current_times = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Skip the WEBVTT header
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == 'WEBVTT':
                start_idx = i + 1
                break
        
        for line in lines[start_idx:]:
            line = line.strip()
            
            # Skip empty lines and WEBVTT metadata
            if not line or line.startswith('NOTE') or line.startswith('STYLE'):
                continue
                
            # Check for timestamp line
            if '-->' in line:
                if current_text and current_times:
                    transcript.append({
                        'speaker': current_speaker or 'Unknown',
                        'timestamp': current_times,
                        'transcript': ' '.join(current_text)
                    })
                current_text = []
                times = line.split('-->')
                start_time = parse_vtt_time(times[0])
                end_time = parse_vtt_time(times[1])
                current_times = f"{start_time}-{end_time}"
                continue
            
            # Check for speaker line (usually in format "Speaker: ")
            if ':' in line and not line[0].isdigit():
                if current_text and current_times:
                    transcript.append({
                        'speaker': current_speaker or 'Unknown',
                        'timestamp': current_times,
                        'transcript': ' '.join(current_text)
                    })
                current_speaker = line.split(':')[0].strip()
                current_text = []
                continue
            
            # Add text to current segment
            if line and not line.startswith('WEBVTT'):
                current_text.append(line)
        
        # Add the last segment
        if current_text and current_times:
            transcript.append({
                'speaker': current_speaker or 'Unknown',
                'timestamp': current_times,
                'transcript': ' '.join(current_text)
            })
            
        return transcript
    except Exception as e:
        print(f"Error parsing VTT file {file_path}: {e}")
        return []

def get_text_files(directory):
    """Get all VTT files in the given directory and its subdirectories."""
    vtt_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.vtt'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)
                vtt_files.append((full_path, relative_path))
    
    return vtt_files

def create_or_update_path_dict(directory, cur_dir):
    """Create or update the path dictionary with text file paths."""
    folder_name = os.path.basename(directory)
    path_dict_file = os.path.join(cur_dir, f"TEXT_{folder_name}_path_dict.json")

    # Check if path_dict.json exists
    if os.path.exists(path_dict_file):
        with open(path_dict_file, 'r') as f:
            path_dict = json.load(f)
    else:
        path_dict = {}

    # Get list of text files in the directory
    text_files = get_text_files(directory)
    
    for file_path, relative_path in text_files:
        file_name = os.path.basename(file_path)
        path_key_name = relative_path
        
        # Update path_dict with new file
        if path_key_name not in path_dict:
            path_dict[path_key_name] = [[file_name, file_path, '', False]]
        else:
            # Check if this file is already in the path_dict
            file_exists = False
            for entry in path_dict[path_key_name]:
                if entry[1] == file_path:
                    file_exists = True
                    break
            
            if not file_exists:
                path_dict[path_key_name].append([file_name, file_path, '', False])

    return path_dict

def save_path_dict(path_dict, file_name, destdir):
    """Save the path dictionary to a JSON file."""
    with open(f"{destdir}/{file_name}", 'w') as json_file:
        json.dump(path_dict, json_file, indent=4)

def gemini_analyze_text(client, prompt, text_content, filename, max_tries=3, delay=1):
    """Analyze VTT content using the Gemini API."""
    print(f"Making LLM inference request for {filename}...")
    
    # Convert the transcript list to a formatted string
    formatted_transcript = "\n".join([
        f"{entry['speaker']} ({entry['timestamp']}): {entry['transcript']}"
        for entry in text_content
    ])
    
    for attempt in range(max_tries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[prompt, formatted_transcript],
                config={
                    'temperature': 0,
                    'max_output_tokens': 8192,
                },)
            print("Got a response! ...")
            return response
            
        except Exception as e:
            print(f"Error in making LLM request for {filename} due to error {e}, trying again... ")
            if attempt < max_tries-1:
                time.sleep(delay)
            else:
                print(f"Couldn't get response even after three tries: {filename}. Error: {e}")
                return None

def save_to_json(text, file_name, output_dir):
    """Save the analysis text to a JSON file."""
    output_dir = sanitize_name(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = sanitize_name(file_name)
    output_file = os.path.join(output_dir, f"{file_name}.json")

    if text:
        try:
            parsed_json = json.loads(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved JSON to {output_file}")
            return
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Attempting to fix the JSON format...")
            
            try:
                # Try to find the start of JSON content
                start_idx = text.find('{')
                if start_idx == -1:
                    start_idx = text.find('[')
                if start_idx != -1:
                    text = text[start_idx:]
                    parsed_json = json.loads(text)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(parsed_json, f, ensure_ascii=False, indent=4)
                    print(f"Successfully saved fixed JSON to {output_file}")
                else:
                    raise ValueError("No JSON content found in text")
            except Exception as e:
                output_file = os.path.join(output_dir, f"ATTN_{file_name}.json")
                print(f"Failed to fix JSON format: {e}. Saving the original response text to the file for manual inspection.")
                with open(output_file, 'w') as json_file:
                    json_file.write(text)
    else:
        raise ValueError(f"No text to save for {file_name}")

def analyze_text(client, path_dict, prompt, dir):
    """Analyze all VTT files in the path dictionary."""
    cur_dir = os.getcwd()
    n_path_dict = path_dict.copy()
    folder_name = os.path.basename(dir)
    
    # Create the base outputs directory
    base_output_dir = os.path.join(cur_dir, "outputs", f"TEXT_{folder_name}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    for file_name in n_path_dict.keys():
        list_files = n_path_dict[file_name]
        safe_file_name = sanitize_name(file_name)
        output_dir = os.path.join(base_output_dir, f"output_{safe_file_name}")
        os.makedirs(output_dir, exist_ok=True)

        for m in range(len(list_files)):
            file_name = list_files[m][0]
            file_path = list_files[m][1]
            analyzed = list_files[m][3]
            
            if not analyzed:
                print(f"Analyzing {file_name}")
                try:
                    # Parse VTT file instead of reading raw text
                    transcript = parse_vtt_file(file_path)
                    if not transcript:
                        print(f"Failed to parse VTT file: {file_name}")
                        list_files[m][3] = False
                        continue
                    
                    response = gemini_analyze_text(client, prompt, transcript, file_name)
                    if response and response.text:
                        print(f"Trying to save output for {file_name} to json file")
                        try:
                            save_to_json(response.text, file_name, output_dir)
                            list_files[m][3] = True
                        except ValueError:
                            list_files[m][3] = False
                    else:
                        print(f"No response for {file_name}. Response: {response}")
                        list_files[m][3] = False
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    list_files[m][3] = False
                
                save_path_dict(n_path_dict, f"TEXT_{folder_name}_path_dict.json", cur_dir)
            else:
                print(f"{file_name} already analyzed, moving on..")
                continue
    
    print("Analysis all finished. Returning updated path_dict.")
    return n_path_dict

def main(text_dir, prompt_type='scialog'):
    """Main function to process text files."""
    folder_name = os.path.basename(text_dir)
    client, prompt, codebook = init(prompt_type)
    cur_dir = os.getcwd()
    
    path_dict = create_or_update_path_dict(text_dir, cur_dir)
    save_path_dict(path_dict, f"TEXT_{folder_name}_path_dict.json", cur_dir)

    new_path_dict = analyze_text(client, path_dict, prompt, text_dir)
    save_path_dict(new_path_dict, f"TEXT_{folder_name}_path_dict.json", cur_dir)

    return path_dict

if __name__ == '__main__':
    dir = input("Please provide the FULL PATH to the directory where VTT files are stored (do NOT wrap it in quotes): ")
    prompt_type = input("Choose prompt type (scialog/covid): ").lower()
    if prompt_type not in ['scialog', 'covid']:
        print("Invalid prompt type. Defaulting to 'scialog'")
        prompt_type = 'scialog'
    main(dir, prompt_type) 