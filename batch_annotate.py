#!/usr/bin/env python3
"""
Advanced batch annotation script with more features:
- Resume from where it left off
- Skip already processed folders
- Better logging and progress tracking
- Configurable options
"""

import os
import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

class BatchAnnotator:
    def __init__(self, base_dir="outputs", script_path="annotate_team_behavior.py", 
                 log_file="batch_annotation.log", state_file="batch_state.json",
                 venv_python="/Users/eveyhuang/Documents/NICO/.venv/bin/python"):
        self.base_dir = base_dir
        self.script_path = script_path
        self.log_file = log_file
        self.state_file = state_file
        self.venv_python = venv_python
        self.state = self.load_state()
        
    def load_state(self):
        """Load processing state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"completed": [], "failed": [], "skipped": []}
    
    def save_state(self):
        """Save processing state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def log(self, message):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def get_output_folders(self):
        """Get all output folders that contain JSON files, sorted by name."""
        output_folders = []
        try:
            # First, get all conference directories
            conference_dirs = []
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    conference_dirs.append(item_path)
            
            # Then, find output folders within each conference directory
            for conf_dir in conference_dirs:
                for item in os.listdir(conf_dir):
                    if item.startswith('output_') or item.startswith('output-'):
                        full_path = os.path.join(conf_dir, item)
                        if os.path.isdir(full_path):
                            # Check if this folder contains JSON files directly or in subdirectories
                            if self.has_json_files(full_path):
                                output_folders.append(full_path)
                            else:
                                # Look for subdirectories with JSON files
                                subdirs_with_json = self.find_json_subdirs(full_path)
                                output_folders.extend(subdirs_with_json)
            
            output_folders.sort()
            return output_folders
        except Exception as e:
            self.log(f"Error accessing directory {self.base_dir}: {e}")
            return []
    
    def has_json_files(self, folder_path):
        """Check if folder contains JSON files directly."""
        try:
            for item in os.listdir(folder_path):
                if item.endswith('.json') and not (item.startswith('all') or item.startswith('verbal')):
                    return True
        except:
            pass
        return False
    
    def find_json_subdirs(self, folder_path):
        """Find subdirectories that contain JSON files."""
        json_subdirs = []
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    # Check if this subdirectory contains JSON files
                    if self.has_json_files(item_path):
                        json_subdirs.append(item_path)
        except:
            pass
        return json_subdirs
    
    def is_already_processed(self, folder_path):
        """Check if folder has already been processed."""
        folder_name = os.path.basename(folder_path)
        return folder_name in self.state["completed"]
    
    def check_annotation_files_exist(self, folder_path):
        """Check if annotation files already exist in the folder."""
        # Look for files with 'gm_v4' in the name
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if 'gm_v4' in file and file.endswith('.json'):
                        return True
        except:
            pass
        return False
    
    def verify_annotation_files_created(self, folder_path):
        """Verify that annotation files were actually created after processing."""
        # Look for files with 'gm_v4' in the name
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if 'gm_v4' in file and file.endswith('.json'):
                        # Check if the file was created recently (within last 5 minutes)
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        current_time = time.time()
                        if current_time - file_time < 300:  # 5 minutes
                            return True
        except:
            pass
        return False
    
    def run_annotation_for_folder(self, folder_path, timeout=3600):
        """Run the annotation script for a specific folder."""
        folder_name = os.path.basename(folder_path)
        self.log(f"Starting annotation for: {folder_name}")
        
        # Check if the virtual environment exists
        if not os.path.exists(self.venv_python):
            self.log(f"ERROR: Virtual environment not found at {self.venv_python}")
            self.state["failed"].append(folder_name)
            self.save_state()
            return False
        
        try:
            result = subprocess.run([
                self.venv_python, self.script_path, folder_path
            ], capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                # Check if there are any error messages in stderr that indicate failure
                error_indicators = [
                    "ModuleNotFoundError", "ImportError", "Error", "Traceback", 
                    "Exception", "Failed", "Timeout", "Connection", "API", 
                    "No JSON files found", "skipping", "Skipping"
                ]
                
                has_error = False
                error_message = ""
                
                # Check stderr for errors
                if result.stderr:
                    for indicator in error_indicators:
                        if indicator in result.stderr:
                            has_error = True
                            error_message = result.stderr
                            break
                
                # Check stdout for errors (some errors go to stdout)
                if not has_error and result.stdout:
                    for indicator in error_indicators:
                        if indicator in result.stdout:
                            has_error = True
                            error_message = result.stdout
                            break
                
                if has_error:
                    self.log(f"ERROR: Failed to process {folder_name} due to error")
                    self.log(f"Error output: {error_message}")
                    self.state["failed"].append(folder_name)
                    self.save_state()
                    return False
                else:
                    # Verify that annotation files were actually created
                    if self.verify_annotation_files_created(folder_path):
                        self.log(f"SUCCESS: Completed annotation for {folder_name}")
                        self.state["completed"].append(folder_name)
                        self.save_state()
                        return True
                    else:
                        self.log(f"ERROR: No annotation files created for {folder_name}")
                        self.state["failed"].append(folder_name)
                        self.save_state()
                        return False
            else:
                self.log(f"ERROR: Failed to process {folder_name} (return code: {result.returncode})")
                if result.stderr:
                    self.log(f"Error output: {result.stderr}")
                self.state["failed"].append(folder_name)
                self.save_state()
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: Annotation for {folder_name} took too long")
            self.state["failed"].append(folder_name)
            self.save_state()
            return False
        except Exception as e:
            self.log(f"EXCEPTION: Error processing {folder_name}: {e}")
            self.state["failed"].append(folder_name)
            self.save_state()
            return False
    
    def run_batch(self, skip_processed=True, continue_on_error=True, delay=5):
        """Run batch processing with options."""
        self.log("="*80)
        self.log("Starting batch annotation process")
        self.log(f"Base directory: {os.path.abspath(self.base_dir)}")
        self.log(f"Script: {os.path.abspath(self.script_path)}")
        self.log(f"Python executable: {self.venv_python}")
        self.log(f"Skip processed: {skip_processed}")
        self.log(f"Continue on error: {continue_on_error}")
        self.log("="*80)
        
        # Check if base directory exists
        if not os.path.exists(self.base_dir):
            self.log(f"ERROR: Base directory '{self.base_dir}' does not exist!")
            return
        
        # Check if script exists
        if not os.path.exists(self.script_path):
            self.log(f"ERROR: Script '{self.script_path}' does not exist!")
            return
        
        # Get all output folders
        output_folders = self.get_output_folders()
        
        if not output_folders:
            self.log(f"ERROR: No output folders found in '{self.base_dir}'")
            return
        
        # Filter out already processed folders if requested
        if skip_processed:
            original_count = len(output_folders)
            output_folders = [f for f in output_folders if not self.is_already_processed(f)]
            skipped_count = original_count - len(output_folders)
            if skipped_count > 0:
                self.log(f"SKIPPED: {skipped_count} already processed folders")
        
        self.log(f"Found {len(output_folders)} folders to process")
        
        # Process each folder
        successful = 0
        failed = 0
        skipped = 0
        start_time = time.time()
        
        for i, folder_path in enumerate(output_folders, 1):
            folder_name = os.path.basename(folder_path)
            
            # Check if already has annotation files
            if self.check_annotation_files_exist(folder_path):
                self.log(f"SKIP: {folder_name} already has annotation files")
                self.state["skipped"].append(folder_name)
                skipped += 1
                continue
            
            self.log(f"Processing folder {i}/{len(output_folders)}: {folder_name}")
            
            if self.run_annotation_for_folder(folder_path):
                successful += 1
            else:
                failed += 1
                if not continue_on_error:
                    response = input(f"Folder {folder_name} failed. Continue? (y/n): ").lower().strip()
                    if response != 'y' and response != 'yes':
                        self.log("Stopping batch processing as requested.")
                        break
            
            # Delay between folders
            if i < len(output_folders):
                self.log(f"Waiting {delay} seconds before next folder...")
                time.sleep(delay)
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        self.log("="*80)
        self.log("BATCH PROCESSING SUMMARY")
        self.log("="*80)
        self.log(f"Total folders: {successful + failed + skipped}")
        self.log(f"Successful: {successful}")
        self.log(f"Failed: {failed}")
        self.log(f"Skipped: {skipped}")
        self.log(f"Total time: {total_time/60:.1f} minutes")
        if successful + failed > 0:
            self.log(f"Average time per folder: {total_time/(successful + failed):.1f} seconds")
        
        if failed == 0:
            self.log("All folders processed successfully!")
        else:
            self.log(f"{failed} folder(s) failed. Check the log for details.")

def main():
    parser = argparse.ArgumentParser(description='Advanced batch annotation script')
    parser.add_argument('--base-dir', default='outputs', help='Base directory containing output folders')
    parser.add_argument('--script', default='annotate_team_behavior.py', help='Path to annotation script')
    parser.add_argument('--venv-python', default='/Users/eveyhuang/Documents/NICO/.venv/bin/python', 
                       help='Path to Python executable in virtual environment')
    parser.add_argument('--no-skip', action='store_true', help='Do not skip already processed folders')
    parser.add_argument('--stop-on-error', action='store_true', help='Stop processing on first error')
    parser.add_argument('--delay', type=int, default=5, help='Delay between folders (seconds)')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout per folder (seconds)')
    
    args = parser.parse_args()
    
    annotator = BatchAnnotator(
        base_dir=args.base_dir,
        script_path=args.script,
        venv_python=args.venv_python
    )
    
    annotator.run_batch(
        skip_processed=not args.no_skip,
        continue_on_error=not args.stop_on_error,
        delay=args.delay
    )

if __name__ == "__main__":
    main()
