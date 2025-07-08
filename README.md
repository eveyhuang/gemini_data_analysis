# gemini_data_analysis

In terminal:
1. Create a virtual environment by `virtualenv venv --python=python3.12.3`; or if on Quest, create a virtual environment through mamba
2. Run `pip install -r requirements.txt`
3. Export gemini api key as environment variable by running "export GOOGLE_API_KEY=‘yourAPIkey’"
4. Run `python analyze_video.py --dir "/path/to/video/directory" --process-video yes --annotate-video yes' (modify path and whether or not to process video and annotate)
