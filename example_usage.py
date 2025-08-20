#!/usr/bin/env python3
"""
Example usage of the modified annotate_team_behavior_othermodels.py script
with AI-VERDE API via LangChain.

Make sure to set up your .env file with:
NCEMS_API_KEY=your_api_key_here
NCEMS_API_URL=https://llm-api.cyverse.ai
"""

import os
from dotenv import load_dotenv
from annotate_team_behavior_othermodels import init, annotate_utterances, extract_json_from_response

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are set
    if not os.environ.get("NCEMS_API_KEY"):
        print("Error: NCEMS_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return
    
    if not os.environ.get("NCEMS_API_URL"):
        print("Error: NCEMS_API_URL not found in environment variables")
        print("Please set it in your .env file")
        return
    
    # Initialize the LLM and codebook
    print("Initializing AI-VERDE API connection...")
    llm, codebook = init()
    
    # Example utterances to annotate
    test_utterances = [
        "I think we should try a completely different approach using reinforcement learning.",
        "Can you explain what you mean by 'latent variable modeling'?",
        "I agree with your approach, that makes a lot of sense.",
        "Alex, can you handle the data processing for this project?"
    ]
    
    print(f"Testing annotation with {len(test_utterances)} sample utterances...")
    
    # Annotate the test utterances
    annotations = annotate_utterances(llm, test_utterances, codebook)
    
    # Print results
    for i, annotation in enumerate(annotations):
        print(f"\nUtterance {i+1}: {annotation['utterance']}")
        print("Annotations:")
        for code_name, explanation in annotation['annotations'].items():
            print(f"  - {code_name}: {explanation}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
