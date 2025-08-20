# AI-VERDE API Integration for Team Behavior Annotation

This code has been modified to use the AI-VERDE API via LangChain instead of the Google Gemini API for annotating team behavior in scientific collaboration meetings.

## Changes Made

### 1. Dependencies Updated
- Replaced `google-genai` with `langchain-community`
- Updated imports to use LangChain's `ChatLiteLLM`

### 2. API Configuration
- Changed from Google Gemini API to AI-VERDE API
- Updated environment variable names:
  - `GOOGLE_API_KEY` → `NCEMS_API_KEY`
  - Added `NCEMS_API_URL` for the AI-VERDE endpoint

### 3. Function Updates
- Modified `init()` function to initialize LangChain with AI-VERDE API
- Updated `annotate_utterances()` to use LangChain's `invoke()` method
- Changed parameter names from `client` to `llm` throughout the code

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in your project directory with:
```
NCEMS_API_KEY=your_ai_verde_api_key_here
NCEMS_API_URL=https://llm-api.cyverse.ai
```

### 3. Get Your AI-VERDE API Key
Follow the instructions at [AI-VERDE Documentation](https://aiverde-docs.cyverse.ai/api/api-token-langchain/#3-create-python-scripts) to obtain your API key.

### 4. Get Available Models
You can check which models are available with:
```bash
curl -s -L "https://llm-api.cyverse.ai/v1/models" -H "Authorization: Bearer [YOUR_API_KEY]" -H 'Content-Type: application/json' | jq
```

## Usage

### Basic Usage
```python
from annotate_team_behavior_othermodels import init, annotate_utterances

# Initialize the LLM and codebook
llm, codebook = init()

# Annotate utterances
utterances = ["Your utterance here"]
annotations = annotate_utterances(llm, utterances, codebook)
```

### Command Line Usage
```bash
python annotate_team_behavior_othermodels.py /path/to/output/directory
```

### Test the Integration
Run the example script to test your setup:
```bash
python example_usage.py
```

## Model Configuration

The default model is set to `litellm_proxy/gemini-2.5-flash`. You can change this in the `init()` function:

```python
llm = ChatLiteLLM(
    model="litellm_proxy/your-preferred-model",  # Change this line
    api_key=NCEMS_API_KEY,
    api_base=NCEMS_API_URL
)
```

## Key Differences from Original Code

1. **API Client**: Uses LangChain's `ChatLiteLLM` instead of Google's `genai.Client`
2. **API Call**: Uses `llm.invoke(prompt)` instead of `client.models.generate_content()`
3. **Response Parsing**: Uses `response.content` instead of `response.text`
4. **Environment Variables**: Uses `NCEMS_API_KEY` and `NCEMS_API_URL` instead of `GOOGLE_API_KEY`

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Make sure your `.env` file contains the correct environment variables
2. **Connection Errors**: Verify your `NCEMS_API_URL` is correct
3. **Model Not Available**: Check available models using the curl command above
4. **JSON Parsing Errors**: The AI model should return valid JSON. If not, check the prompt format

### Error Messages
- `NCEMS_API_KEY not found`: Set your API key in the `.env` file
- `NCEMS_API_URL not found`: Set your API URL in the `.env` file
- `JSONDecodeError`: The model response wasn't valid JSON

## References

- [AI-VERDE Documentation](https://aiverde-docs.cyverse.ai/api/api-token-langchain/#3-create-python-scripts)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
