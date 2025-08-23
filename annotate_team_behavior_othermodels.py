import ffmpeg
import re
import time, json, os
from langchain_community.chat_models import ChatLiteLLM
from dotenv import load_dotenv
import subprocess
import unicodedata
import argparse


def init():
    load_dotenv()
    NCEMS_API_KEY = os.environ.get("NCEMS_API_KEY")
    NCEMS_API_URL = os.environ.get("NCEMS_API_URL")
    
    # Initialize LangChain with AI-VERDE API
    llm = ChatLiteLLM(
        model="litellm_proxy/js2/llama-4-scout",  # You can change this model as needed
        api_key=NCEMS_API_KEY,
        api_base=NCEMS_API_URL
    )

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

    code_book_v2 = """
    (1) propose new idea: introducing NEW ideas, suggestions, solutions, or approaches that have not been previously discussed in the meeting. Key Distinguisher: NEWNESS - the content has not been mentioned before. Example: "I think we should try a completely different approach..."; "Here's a new idea - what about..." ;
    (2) develop idea: expanding, building upon, or elaborating existing ideas through reasoning, examples, clarification, and evidence.  Example: "This appraoch would work for our problem because... "; "Let me give you a concrete example of how that would look..."; "A Nature paper showed this method outperforms others.";
    (3) ask question: Request information, clarification, or expertise from other team members on a prior statement or idea proposed by another group member. Example:"Can you explain what you mean by 'latent variable modeling'?"; "do we have data that is needed to train this model?"; "what is your thought on this approach?";
    (4) signal expertise: Explicitly stating one's own or others' expertise or qualifications related to the task. Example: "Emily is our expert in bio-engineering, she could take the lead here"; "I came from a bio-chemistry background and have worked on.. ";
    (5) identify gap: explicitly recognizing one's own or the group's lack of knowledge, skill, resource, or familiarity in a particular domain, approach or topic. Examples: "I'm not very familiar with this topic"; "This isn't really my area of expertise"; "we don't have the data for that".
    (6) acknowledge contribution: verbally recognizes another group member's input, but not agreeing or expanding. Example: "Lisa has previously brought up the idea of using oxygen as the core material"; Not this code if the utterance only has a few words such as "okay", "mhm", "I see" or acknowledge one's own input, such as "I wrote down some ideas".
    (7) supportive response: Expressing agreement, validation, or positive evaluation for other group members' contributions without adding new content. NOT this code if the speaker simply says "yep" or "yes" to acknowledge a prior statement. Example: "I agree with your approach"; "You made agreat point".
    (8) critical response: Questioning, challenging, disagreeing with, or providing negative evaluation of ideas, approaches, or information provided by other group members.  Example: "I'm concerned about the feasibility of..."; "Have we considered the risks of...?"; "You might be missing a very important limitation of this approach". 
    (9) offer feedback: Provide specific suggestions for improvement, modification of existing ideas or approaches proposed by other group members. NOT This Code If: Pure criticism without suggestions or simple agreement. Examples: "here's how we could strengthen the idea..."; 
    (10) summarize conversation: Summarize what has been previously discussed by the group. For example: "So far we have talked about the possibility of training an AI model and limitations of data for traininig."
    (11) express humor: makes a joke. example: "well, at least this model isn't as bad as our last one!" Not this code if the speaker is simply laughing, such as "haha" or if only the tone sounds humorous but not explicitly making a joke.
    (12) encourage participation: invites someone else in the group to contribute their expertise, opinions or ideas. "Alex, what do you think?"; "Anyone has any thoughts?";
    (13) process management: Managing meeting flow, time, structure, or organizing group activities. Examples: “Let’s keep to the agenda”;“We have 5 minutes left.” NOT This Code If: Clarifying goals or assigning tasks, or simply greeting the group.
    (14) assign task: assigns responsibility, roles, deadlines, or action items to the group or a group member. Example: "Alex, can you handle data processing?"; NOT This Code If: Defining what needs to be accomplished (goals);
    (15) clarify goal:  Defining, clarifying, or seeking clarity on objectives, outcomes, expectations, or success criteria for the group to achieve. Key Distinguisher: DEFINING what needs to be accomplished by the group. Examples: "Let's be clear about what we're trying to achieve here..."; "our goal is to ...";
    (16) confirm decision: Explicitly committing to, choosing, or confirming a single idea, approach, goal, or course of action as a group outcome. NOT This Code If: The group is still brainstorming or discussing multiple options without clear closure or commitment. Examples: “It sounds like everyone agrees we’ll proceed with the third option.”

    """

    return llm, code_book_v2

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

def extract_json_from_response(response_content):
    """
    Extract JSON from response content that may be wrapped in markdown code blocks.
    Handles cases where multiple JSON blocks are present.
    
    Args:
        response_content: The response content from the LLM
        
    Returns:
        Parsed JSON object (takes the last valid JSON found)
    """
    content = response_content.strip()
    
    # Find all JSON code blocks in the response
    json_blocks = []
    
    # Pattern to match ```json ... ``` blocks
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_matches = re.findall(json_pattern, content)
    
    # Pattern to match ``` ... ``` blocks (without json specifier)
    code_pattern = r'```\s*([\s\S]*?)\s*```'
    code_matches = re.findall(code_pattern, content)
    
    # Combine all matches
    all_blocks = json_matches + code_matches
    
    # Also look for JSON objects that might not be in code blocks
    # Pattern to match { ... } JSON objects
    json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_object_matches = re.findall(json_object_pattern, content)
    
    # Add JSON object matches to all_blocks
    all_blocks.extend(json_object_matches)
    
    # Try to parse each block as JSON, keeping track of the last valid one
    last_valid_json = None
    final_answer_json = None
    
    for i, block in enumerate(all_blocks):
        block = block.strip()
        
        # Check if this block is marked as final answer
        is_final_answer = False
        if i < len(all_blocks) - 1:  # Not the last block
            # Look for "final answer" markers in the surrounding text
            block_start = content.find(block)
            if block_start != -1:
                # Check text after this block for final answer markers
                after_block = content[block_start + len(block):block_start + len(block) + 100]
                if any(marker in after_block.lower() for marker in ['final answer', 'final result', 'conclusion', 'therefore']):
                    is_final_answer = True
        
        try:
            parsed = json.loads(block)
            # If it's a list, take the first item or convert to dict
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    parsed_dict = parsed[0]  # Store first item as dict
                else:
                    parsed_dict = {"result": parsed}  # Convert list to dict
            elif isinstance(parsed, dict):
                parsed_dict = parsed
            else:
                parsed_dict = {"result": parsed}
            
            # Store as last valid JSON
            last_valid_json = parsed_dict
            
            # If this is marked as final answer, store it separately
            if is_final_answer:
                final_answer_json = parsed_dict
                
        except json.JSONDecodeError:
            continue
    
    # Return the final answer if found, otherwise the last valid JSON
    if final_answer_json is not None:
        # print("final_answer: ", final_answer_json)
        return final_answer_json
    elif last_valid_json is not None:
        # print("last_valid_json: ", last_valid_json)
        return last_valid_json
    
    # If no valid JSON found in code blocks, try parsing the entire content
    try:
        # Remove all markdown code blocks and try to parse what's left
        cleaned_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL).strip()
        if cleaned_content:
            # print("cleaned_content: ", cleaned_content)
            return json.loads(cleaned_content)
    except json.JSONDecodeError:
    
        pass
    
    # Try parsing the content directly as JSON (in case it's already clean JSON)
    try:
        json_content = json.loads(content)
        # print("json_content: ", json_content)
        return json_content
    except json.JSONDecodeError as e:
        print(f"Failed to parse content as JSON: {e}")
        print(f"Content was: {content}")
        # Return empty dict as fallback
        return {}

# Annotate utterances with AI-VERDE API via LangChain
def annotate_utterances(llm, merged_list, codebook):
    """
    Iterates over each utterance, using the full list as context.
    Returns structured annotations in JSON format.
    """
    # print(f"Annotating {len(merged_list)} utterances")
    annotations = []  # Store annotations for all utterances

    for i, utterance in enumerate(merged_list):
        # Prepare the prompt
        comm_prompt = f"""
        You are an expert in interaction analysis, team science, and qualitative coding. Your task is to analyze an utterance from a scientific collaboration meeting and annotate it using the codes in the provided codebook.
        Each utterance is structured with "speaker name: what the speaker said".
        Apply code based onwhat is explicitly observed from the utterance, not inferred intent or motivation. Do not force a code that is not explicitly observed in the utterance.
        Use the full conversation context to understand what has been previously discussed when deciding the most accurate code that applies to the utterance.

        **Annotation Guidelines:** 
        - If no code applies, write 'None' as the code name.
        - Only choose multiple codes (no more than 3)if they are all explicitly observed in the utterance.
        - If what the speaker said only has a few words such as "yep", "umm", "I see", always default to the code "None".
        - For each code you choose, provide a json object with the following fields:
            code name: the name of the code from the codebook that applies to the utterance;
            explanation: Justify your reasoning on why this code applies in 1 sentence, using evidence from the utterance and context, and definitions from the codebook;
        - Ensure that each annotation follows the structured JSON format.

        ** Codebook for Annotation:**
        {codebook}

        ** Full Conversation Context:**
        {json.dumps(merged_list[:i+1], indent=2)}  # Include past utterances up to the current one

        ** Utterance to Annotate:**
        "{utterance}"

        **Expected Output:**
        You may reason through your analysis and provide multiple JSON blocks if needed to show your thinking process.
        However, your FINAL answer should be a single JSON object where each coded category is a key and the explanation is the value.
        If multiple codes apply, include the codes and explanations in the same JSON object.
        If no code applies, use {{"None": "No relevant code applies to this utterance"}}.
        Make sure your final answer JSON black is clearly marked as the final answer.
        Example format: {{"code name": "explanation", "another code": "another explanation"}}
        """

        # Call AI-VERDE API via LangChain
        response = llm.invoke(comm_prompt)    
        
        annotation = extract_json_from_response(response.content)  # Parse response as JSON
        print("utterance: ", utterance)
        print("annotation: ", annotation)

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
    # Get all JSON files except those starting with 'all', 'verbal', or 'verbal_llama'
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

def annotate_and_merge_outputs(llm, output_dir, codebook):
    """
    Annotate and merge outputs for all subfolders in the output directory.
    Args:
        llm: LangChain LLM client for AI-VERDE API
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
            verbal_file = os.path.join(folder, f"verbal_llama_{json_dir_name}.json")
            all_file = os.path.join(folder, f"all_llama_{json_dir_name}.json")
            
            if not is_valid_json_file(verbal_file):
                print(f"No existing/valid verbal_llama file in {folder}, annotating now...")
                try:
                    merged_output = merge_output_json(folder)
                except InvalidJsonContentError as e:
                    print(f"Skipping {json_dir} due to invalid JSON content: {str(e)}")
                    continue
                    
                verbal_annotations = []
                
                # print(f"Annotating verbal behaviors for {json_dir_name}")
                annotations = annotate_utterances(llm, merged_output[1], codebook)
                verbal_annotations = annotations
                output_file = verbal_file
                with open(output_file, "w") as f:
                    print(f"Saved verbal_llama annotations to {output_file}")
                    json.dump(annotations, f, indent=4)
                
                    
                if len(verbal_annotations) == len(merged_output[0]):
                    all_anno = merged_output[0].copy()
                    for i in range(len(verbal_annotations)):
                        anno = verbal_annotations[i]['annotations']
                        all_anno[i]['annotations'] = anno
                    
                    with open(all_file, "w") as f:
                        print(f"Merged verbal_llama annotations with existing video annotations to {all_file}")
                        json.dump(all_anno, f, indent=4)
                
            else:
                print(f"Already annotated verbal_llama files in {folder}, skipping...")
        
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Annotate and merge team behavior analysis outputs.')
    parser.add_argument('output_dir', help='Path to the directory containing output folders')
    
    args = parser.parse_args()
    
    # Initialize LLM and codebook
    llm, codebook = init()
    
    # Process the output directory
    print(f"Processing outputs in: {args.output_dir}")
    annotate_and_merge_outputs(llm, args.output_dir, codebook)
    print("\nAnnotation and merging complete!")

if __name__ == '__main__':
    main()