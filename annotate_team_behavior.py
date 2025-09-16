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
        (15) express frustration: verbalizes irritation, impatience, or dissatisfaction. "We have been going in circles on this issue."
        (16) acknowledge contribution: verbally recognizes another person's input, but not agreeing or expanding. "That is a great point."
        (17) encourage participation: invites someone else to contribute. "Alex, what do you think?"
        (18) express enthusiasm: expresses excitement, optimism, or encouragement. "This is an exciting direction!"
        (19) express humor: makes a joke or laughs. example: "well, at least this model isn't as bad as our last one!"
    """

    code_book_v2 = """
    (1) propose new idea: introducing NEW ideas, suggestions, solutions, or approaches that have not been previously discussed in the meeting. Key Distinguisher: NEWNESS - the content has not been mentioned before. Example: "I think we should try a completely different approach..."; "Here's a new idea - what about..." ;
    (2) develop idea: expanding, building upon, or elaborating existing ideas through reasoning, examples, clarification, and evidence.  Example: "This appraoch would work for our problem because... "; "Let me give you a concrete example of how that would look..."; "A Nature paper showed this method outperforms others.";
    (3) ask question: Request information, clarification, or expertise from other team members on a prior statement or idea proposed by another group member. Example:"Can you explain what you mean by 'latent variable modeling'?"; "do we have data that is needed to train this model?"; "what is your thought on this approach?"; NOT this code if the speaker is identifying a gap or offering critiques.
    (4) signal expertise: Explicitly stating one's own or others' expertise or qualifications related to the task. Example: "Emily is our expert in bio-engineering, she could take the lead here"; "I came from a bio-chemistry background and have worked on.. ";
    (5) identify gap: explicitly recognizing one's own or the group's lack of knowledge, skill, resource, or familiarity in a particular domain, approach or topic. Examples: "I'm not very familiar with this topic"; "This isn't really my area of expertise"; "we don't have the data for that".
    (6) acknowledge contribution: verbally recognizes another group member's input, but not agreeing or expanding. Example: "Lisa has previously brought up the idea of using oxygen as the core material"; Not this code if the utterance only has a few words such as "okay", "mhm", "I see" or acknowledge one's own input, such as "I wrote down some ideas".
    (7) supportive response: Expressing agreement, validation, or positive evaluation for other group members' contributions without adding new content. NOT this code if the speaker simply says "yep" or "yes" to acknowledge a prior statement. Example: "I agree with your approach"; "You made agreat point".
    (8) critical response: Questioning, challenging, disagreeing with, or providing negative evaluation of ideas, approaches, or information provided by other group members.  Example: "I'm concerned about the feasibility of..."; "Have we considered the risks of...?"; "You might be missing a very important limitation of this approach". 
    (9) offer feedback: Provide specific suggestions for improvement, modification of existing ideas or approaches proposed by other group members. NOT This Code If: Pure criticism without suggestions or simple agreement. Examples: "here's how we could strengthen the idea..."; 
    (10) summarize conversation: Summarize what has been previously discussed by the group. For example: "So far we have talked about the possibility of training an AI model and limitations of data for traininig."
    (11) express humor: makes a joke. example: "well, at least this model isn't as bad as our last one!" Not this code if the speaker is simply laughing, such as "haha" or if only the tone sounds humorous but not explicitly making a joke.
    (12) encourage participation: invites someone else in the group to contribute their expertise, opinions or ideas. "Alex, what do you think?"; "Anyone has any thoughts?";
    (13) process management: Managing meeting flow, time, structure, or organizing group activities. Examples: "Let's keep to the agenda";"We have 5 minutes left." NOT This Code If: Clarifying goals or assigning tasks, or simply greeting the group.
    (14) assign task: assigns responsibility, roles, deadlines, or action items to the group or a group member. Example: "Alex, can you handle data processing?"; NOT This Code If: Defining what needs to be accomplished (goals);
    (15) clarify goal:  Defining, clarifying, or seeking clarity on objectives, outcomes, expectations, or success criteria for the group to achieve. Key Distinguisher: DEFINING what needs to be accomplished by the group. Examples: "Let's be clear about what we're trying to achieve here..."; "our goal is to ...";
    (16) confirm decision: Explicitly committing to, choosing, or confirming a single idea, approach, goal, or course of action as a group outcome. NOT This Code If: The group is still brainstorming or discussing multiple options without clear closure or commitment. Examples: "It sounds like everyone agrees we'll proceed with the third option."

    """

    code_book_v3 = """
    (1) Idea Generation & Development: Introducing a new idea OR expanding on an existing idea with reasoning, examples, or evidence. Do not apply if: Utterance only affirms, acknowledges, or critiques without adding content. Examples: "What if we run a pilot with undergraduates?"; "Building on Alex's point, we could also test this in a different context."

    (2) Information Seeking & Gap Identification: Asking a direct question OR highlighting missing knowledge/resources, either for the group or oneself. Do not apply if: The utterance proposes a solution instead of identifying a gap. Examples: "Do we have the dataset needed for this?"; "I'm not familiar with this method, can someone explain it?"

    (3) Knowledge Contribution & Expertise Signal: Providing factual information, sharing prior experience, or explicitly stating one's own/others' expertise, role, or goals. Examples: "My group has been working on this problem…"; "Last year someone published a paper to confirm the possibility of this method."

    (4) Evaluation & Feedback: Expressing judgment of an idea's content — agreement, disagreement, critique, or suggestion for improvement. Do not apply if: Utterance only thanks or praises the person (→ Code 5). Examples: "Maybe more data isn't exactly the problem that we're looking at."; "That hesitation by itself says our models aren't quite there yet."

    (5) Acknowledgment, Support, & Interest: Recognizing or affirming another's contribution/effort OR expressing enthusiasm, curiosity, or positive affect toward an idea or person. Includes appreciation and positive humor. Do not apply if: Utterance evaluates the content of an idea (→ Code 4). Examples: "Thanks for pulling those numbers together."; "I was really interested in what Lisa said."

    (6) Participation & Inclusion: Explicitly inviting or encouraging others to contribute. Examples: "Anyone want to chime in?"; "Alex, what do you think?"

    (7) Process & Task Management: Managing meeting flow (time, agenda, topic), assigning tasks, OR clarifying team goals/expectations. Examples: "We have 10 minutes left, let's wrap up."; "Lisa, can you draft the methods section?"

    (8) Decision & Confirmation: Explicitly confirming consensus or commitment to a course of action for the group. Do not apply if: Group is still exploring options. Examples: "So the decision is to submit jointly."; "It sounds like we're all going with option B."

    (9) Summarization & Integration: Synthesizing or restating multiple prior contributions into a coherent summary. Do not apply if: New ideas are introduced (→ Code 1). Examples: "I'm just making notes here. So I guess we talked about…"; "So far, we've discussed two main approaches and their limitations."

    """

    return client, code_book_v3

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
def annotate_utterances(client, merged_list, codebook, type='deductive'):
    """
    Iterates over each utterance, using the full list as context.
    Returns structured annotations in JSON format.
    """
    # print(f"Annotating {len(merged_list)} utterances")
    annotations = []  # Store annotations for all utterances

    for i, utterance in enumerate(merged_list):
        # Prepare the prompt
        if type == 'deductive':
            comm_prompt = f"""
            You are an expert in deductive qualitative coding, team science, and qualitative coding. Your task is to analyze an utterance from a scientific collaboration meeting and annotate it using the codes in the provided codebook, then score each code based on quality criteria. 
            Apply code based on what is explicitly observed from the utterance, not inferred intent or motivation. Do not force a code that is not explicitly observed in the utterance.
            Use the full conversation context to understand what has been previously discussed when deciding the most accurate code that applies to the utterance.

            **Annotation Guidelines:** 
            - If no code applies, write 'None' as the code name.
            - Only choose multiple codes (no more than 3) if they are all explicitly observed in the utterance.
            - If the utterance only has a few words such as "yep", "umm", "I see", always choose the code "None".
            - For each code you choose, provide a json object with the following fields:
                code_name: the name of the code from the codebook that applies to the utterance;
                explanation: Justify your reasoning on why this code applies in 1 sentence, using evidence from the utterance and context, and definitions from the codebook;
                score: a numerical score (0, 1, 2, or 3) based on the quality criteria below;
                score_justification: explain why you gave this score in 1 sentence;

            ** Codebook for Annotation:**
            {codebook}

            ** Scoring Criteria (0-3 scale for each code):**
            0 = Negative/dysfunctional (hurts the process)
            1 = Minimal/weak (barely functional)
            2 = Adequate/average (functional, but not special)
            3 = High-quality/exceptional (moves team forward and makes team more effective, very concrete and specific)

            Idea Generation & Development:
            0 = Negative/dysfunctional. Example: "That's a stupid idea." (dismissive, hurts collaboration)
            1 = Minimal/weak. Example: "We should try something different." (vague, no detail, not actionable)
            2 = Adequate/average. Example: "Let's use more data for this." (clear direction but undeveloped)
            3 = High-quality/exceptional. Example: "Building on Alex's idea, we could run a small pilot study with undergraduates to check feasibility." (novel, relevant, elaborated)

            Information Seeking & Gap Identification:
            0 = Negative/dysfunctional. Example: "I don't care about that." (dismissive, blocks information flow)
            1 = Minimal/weak. Example: "I don't know." (stated, no follow-up)
            2 = Adequate/average. Example: "Do we have the dataset?" (clear but general)
            3 = High-quality/exceptional. Example: "Do we have the labeled MRI dataset from 2020?" (clear + specific)

            Knowledge Contribution & Expertise Signal:
            0 = Negative/dysfunctional. Example: "That's not how it works in my field." (dismissive, creates barriers)
            1 = Minimal/weak. Example: "My favorite class was statistics." (off-topic, not relevant to task)
            2 = Adequate/average. Example: "I used clustering in my lab" (relevant fact/expertise)
            3 = High-quality/exceptional. Example: "In my lab we used hierarchical clustering with 500 samples and got 85% accuracy" (relevant + concrete detail)

            Evaluation & Feedback:
            0 = Negative/dysfunctional. Example: "That's completely wrong." (dismissive, harsh, no constructive value)
            1 = Minimal/weak. Example: "That's wrong." (dismissive, no reason, harsh)
            2 = Adequate/average. Example: "That could work because it's efficient." (judgment with minimal reason)
            3 = High-quality/exceptional. Example: "I don't think this will scale because it is very expensive. Instead, we could test with a smaller subset first." (constructive critique with reasoning + suggestions)

            Acknowledgment, Support, & Interest:
            0 = Negative/dysfunctional. Example: "Whatever." (dismissive, undermines others)
            1 = Minimal/weak. Example: "Yeah." / "Okay." (token acknowledgment)
            2 = Adequate/average. Example: "That's a good point, Alex." / "Interesting idea." (explicit thanks, praise, or mild curiosity)
            3 = High-quality/exceptional. Example: "I really appreciate how you explained that—it cleared things up for me." (strong acknowledgment/enthusiasm)

            Participation & Inclusion:
            0 = Negative/dysfunctional. Example: "I don't want to hear from anyone else." (excludes others, blocks participation)
            1 = Minimal/weak. Example: "Any thoughts?" (generic invite)
            2 = Adequate/average. Example: "Alex, what do you think?" (direct invite, general)
            3 = High-quality/exceptional. Example: "Shannon, since you worked on the dataset, what's your view?" (direct invite + topic/expertise specified)

            Process & Task Management:
            0 = Negative/dysfunctional. Example: "This meeting is a waste of time." (undermines process)
            1 = Minimal/weak. Example: "Let's stay on track" (vague)
            2 = Adequate/average. Example: "Let's move to the next agenda item" (clear structuring)
            3 = High-quality/exceptional. Example: "Our goal is to finish methods by Friday, and Lisa will draft it" (clear structuring + detail)

            Decision & Confirmation:
            0 = Negative/dysfunctional. Example: "I don't care what we decide." (undermines decision-making)
            1 = Minimal/weak. Example: "I guess that's fine." (ambiguous closure, unclear if all agree)
            2 = Adequate/average. Example: "So maybe we'll go with option B?" (suggested decision, unclear consensus)
            3 = High-quality/exceptional. Example: "Okay, we all agree on option B. That's the plan." (explicit confirmation of consensus)

            Summarization & Integration:
            0 = Negative/dysfunctional. Example: "This whole discussion was pointless." (dismissive, undermines progress)
            1 = Minimal/weak. Example: "So yeah, we talked about some things." (incomplete/inaccurate summary)
            2 = Adequate/average. Example: "We've talked about data collection, but not analysis yet." (accurate but partial)
            3 = High-quality/exceptional. Example: "So far, we've considered A and B approaches. Alex raised feasibility, Lisa added cost concerns." (accurate, comprehensive, integrating multiple inputs)

            ** Prior Conversation Context (previous 5 utterances):**
            {json.dumps(merged_list[max(0, i-4):i+1], indent=2)}  # Include previous 5 utterances up to the current one

            ** Utterance to Annotate:**
            "{utterance}"

            **Expected Output:**
            Output ONLY a single JSON object where each coded category is a key and contains an object with explanation, score, and score_justification.
            If multiple codes apply, include the most relevant codes in the same JSON object.
            If no code applies, use {{"None": {{"explanation": "No relevant code applies to this utterance", "score": 0, "score_justification": "No code to score"}}}}.
            Do not include any explanatory text or reasoning outside the JSON. Do not provide multiple JSON blocks.
            Example format: {{"code_name": {{"explanation": "explanation", "score": 2, "score_justification": "score reason"}}, "another_code": {{"explanation": "another explanation", "score": 1, "score_justification": "another score reason"}}}}
            """
        else:
            
            comm_prompt = f"""
                You are assisting with inductive qualitative coding of a long multi-speaker meeting transcript. These transcripts come from recorded meetings between groups of scientists 
                who are meeting for the first time. They have been assigned to discuss ideas within pre-determined scientific problem domains. After these meetings, 
                some subset of participants may choose to form teams and submit grant proposals together. 
                Your job is to analyze the transcripts to understand communication behaviors that may be detrimental to possible team formation, and estimate funding likelihood.
                
                You will analyze ONLY the provided utterance (not the context) and output structured JSON.

                GOAL
                - Identify communication strategies/behaviors relevant to collaboration, team formation, and likelihood of grant success.
                - Multiple strategies can appear in one utterance; code each distinctly.
                - If an utterance has no relevant behavior, record "none" for it.

                INPUTS
                - Utterance to annotate: {utterance}.
                - Context (The 4 previous and following utterances): {json.dumps(merged_list[max(0, i-4):min(len(merged_list), i+5)], indent=2)} 
                
                TASKS
                (1) INDUCTIVE CODING
                For each utterance:
                - Identify zero or more behaviors or strategies(codes) that appears in this utterance. Apply code based on what is explicitly observed from the utterance, not inferred intent or motivation.
                Use the provided context to understand what has been previously or later discussed but do not code the behaviors in the context.
                - For each code:
                * code_name: short (e.g., "ask question", "offer criticism", "propose idea" .... etc).
                * definition: 1–2 sentence definition (your words).
                * justification: justify your chosen code with rationales and exact quote (verbatim substring from the utterance) as evidence.
                - If no relevant behavior for an utterance, include a single item: {{"code_name":"none"}} for that utterance.

                CONSTRAINTS
                - Do NOT invent content beyond the utterance text.
                - Keep code_name controlled and short; prefer reusing existing names if possible.
                - You can choose more than 1 behavior (maximum 3) but only the additional ones are all explicitely observed in the utterance.
                - If the utterance only has a few words such as "yep", "umm", "I see", always choose the code "none".
                - Use provided char offsets as-is; include them on each coded behavior.

                OUTPUT JSON SCHEMA (STRICT):
                {{
                    "codes": [
                        {{
                        "code_name": "<short>",
                        "definition": "<1-2 sentences>",
                        "justification": "<1-2 sentences with verbatim quote>",
                        }}, 
                        {{"code_name":"none"}} (if no code applies, no quality code is needed)
                    ]
                }}
        """


        # Call Gemini API with retry logic for server errors
        max_retries = 3
        retry_delay = 5  # seconds
        final_wait = 300  # 5 minutes
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[comm_prompt],
                    config={
                        'response_mime_type':'application/json',
                        'temperature':0     
                    },)
                
                annotation = json.loads(response.text)  # Parse response as JSON
                print("utterance: ", utterance)
                print("annotation: ", annotation)
                break  # Success, exit retry loop
                
            except ServerError as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    if attempt < max_retries - 1:  # Not the last attempt
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Server overloaded (503 error). Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final attempt failed, wait 5 minutes and try one more time
                        print(f"Max retries reached. Server still overloaded. Waiting {final_wait} seconds (5 minutes) for final attempt...")
                        time.sleep(final_wait)
                        
                        try:
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[comm_prompt],
                                config={
                                    'response_mime_type':'application/json',
                                    'temperature':0     
                                },)
                            
                            annotation = json.loads(response.text)  # Parse response as JSON
                            print("utterance: ", utterance)
                            print("annotation: ", annotation)
                            break  # Success after final wait
                            
                        except ServerError as final_e:
                            if "503" in str(final_e) or "overloaded" in str(final_e).lower():
                                print(f"Server still overloaded after 5-minute wait. Exiting program.")
                                print(f"Failed utterance: {utterance}")
                                exit(1)
                            else:
                                print(f"Server error after final wait: {final_e}. Exiting program.")
                                print(f"Failed utterance: {utterance}")
                                exit(1)
                        except Exception as final_e:
                            print(f"Error after final wait: {final_e}. Exiting program.")
                            print(f"Failed utterance: {utterance}")
                            exit(1)
                else:
                    # Other server errors
                    print(f"Server error: {e}. Exiting program.")
                    print(f"Failed utterance: {utterance}")
                    exit(1)
                    
            except Exception as e:
                print(f"Error parsing response or other error: {e}. Exiting program.")
                print(f"Failed utterance: {utterance}")
                exit(1)

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

def annotate_and_merge_outputs(client, output_dir, codebook, fileName, type='deductive'):
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
            verbal_file = os.path.join(folder, f"verbal_{fileName}_{json_dir_name}.json")
            all_file = os.path.join(folder, f"all_{fileName}_{json_dir_name}.json")
            
            if not is_valid_json_file(verbal_file):
                print(f"No existing/valid verbal file in {folder}, annotating now...")
                try:
                    merged_output = merge_output_json(folder)
                except InvalidJsonContentError as e:
                    print(f"Skipping {json_dir} due to invalid JSON content: {str(e)}")
                    continue
                    
                verbal_annotations = []
                
                # print(f"Annotating verbal behaviors for {json_dir_name}")
                annotations = annotate_utterances(client, merged_output[1], codebook, type=type)
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
    annotate_and_merge_outputs(client, args.output_dir, codebook, 'gm_v3', type='deductive')
    print("\nAnnotation and merging complete!")

if __name__ == '__main__':
    main()