# Audio Transcription Script

This script (`transcribe_audio.py`) is a modified version of `analyze_video.py` that extracts audio from videos and uses the Deepgram API for transcription instead of using Gemini for video analysis.

## Features

- **Video Processing**: Maintains the same video splitting and chunking functionality as the original script
- **Audio Extraction**: Extracts audio from video files using ffmpeg
- **Deepgram Transcription**: Uses Deepgram API for high-quality speech-to-text transcription
- **Speaker Diarization**: Identifies different speakers in the audio
- **Structured Output**: Saves transcriptions in JSON format with timestamps and speaker information

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_transcribe.txt
```

## Environment Setup

1. Create a `.env` file in the same directory as the script
2. Add your Deepgram API key:

```
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

You can get a Deepgram API key by signing up at [https://deepgram.com/](https://deepgram.com/)

## Usage

### Command Line Interface

```bash
python transcribe_audio.py --dir /path/to/video/directory --process-video yes --transcribe-audio yes
```

### Arguments

- `--dir`: Full path to the directory where videos are stored (required)
- `--process-video`: Whether to process videos (split into chunks if >10 minutes) - 'yes' or 'no' (required)
- `--transcribe-audio`: Whether to transcribe audio files - 'yes' or 'no' (required)

### Example

```bash
python transcribe_audio.py --dir /Users/username/videos/conference_2023 --process-video yes --transcribe-audio yes
```

## Workflow

1. **Video Processing**: 
   - Scans the directory for video files (supports .mp4, .mkv, .avi, .mov, .flv, .wmv)
   - Converts .mkv files to .mp4 if needed
   - Splits videos longer than 10 minutes into 10-minute chunks

2. **Audio Extraction**:
   - Extracts audio from each video chunk using ffmpeg
   - Saves audio as 16kHz mono WAV files for optimal transcription quality

3. **Transcription**:
   - Uses Deepgram's Nova-2 model for transcription
   - Enables speaker diarization to identify different speakers
   - Includes punctuation and smart formatting
   - Saves results in structured JSON format

## Output Structure

The script creates the following directory structure:

```
outputs/
└── NEW_[folder_name]/
    └── output_[video_name]/
        ├── chunk1.json
        ├── chunk2.json
        └── ...
```

Each JSON file contains:

```json
{
    "metadata": {
        "duration": 600.0,
        "channels": 1,
        "created": "2024-01-01T12:00:00.000Z",
        "model_info": {
            "name": "nova-2",
            "version": "2024-01-01",
            "arch": "nova-2"
        }
    },
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "Full transcript text here...",
                        "confidence": 0.95,
                        "words": [
                            {
                                "word": "Hello",
                                "start": 0.5,
                                "end": 0.8,
                                "confidence": 0.99,
                                "speaker": 0
                            }
                        ]
                    }
                ]
            }
        ]
    }
}
```

## Key Differences from Original Script

1. **API Integration**: Uses Deepgram instead of Gemini
2. **Audio Focus**: Extracts and processes audio instead of analyzing video content
3. **Transcription Output**: Produces structured transcriptions with speaker identification
4. **Simplified Workflow**: Removes video analysis prompts and focuses on speech-to-text

## Error Handling

- Retries transcription up to 3 times if API calls fail
- Handles various video formats and conversions
- Preserves processing status to avoid re-processing completed files
- Provides detailed error messages for troubleshooting

## Notes

- The script maintains the same path dictionary structure as the original for compatibility
- Audio files are stored alongside video chunks for reference
- Processing status is tracked to allow resuming interrupted operations
- Supports the same directory structure and naming conventions as the original script
