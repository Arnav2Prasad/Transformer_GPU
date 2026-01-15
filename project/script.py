





import os
import time
import subprocess
import shutil
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path to train.py
train_file = "train.py"

# GitHub Configuration from .env file
GITHUB_REPO_URL = "https://github.com/Arnav2Prasad/Transformer_GPU.git"
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "Arnav2Prasad")  # From .env, default if not set
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # From .env file
REPO_NAME = "Transformer_GPU"
LOCAL_REPO_PATH = "./Transformer_GPU"
MONITOR_LOGS_DIR = os.path.join(LOCAL_REPO_PATH, "monitor_logs")

GIT_USER_NAME = os.getenv("GIT_USER_NAME", "Arnav2Prasad")
GIT_USER_EMAIL = os.getenv("GIT_USER_EMAIL", "arnav.pr")





import sys
from contextlib import redirect_stdout, redirect_stderr
import io




class OutputCapture:
    """Capture all screen output to file and console."""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file = None
        
    def __enter__(self):
        # Open log file in append mode
        self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
        
        # Create a custom stream that writes to both console and file
        class TeeOutput:
            def __init__(self, console, file):
                self.console = console
                self.file = file
                
            def write(self, data):
                # Write to console
                self.console.write(data)
                # Write to file
                self.file.write(data)
                self.file.flush()
                
            def flush(self):
                self.console.flush()
                self.file.flush()
        
        # Replace stdout and stderr
        sys.stdout = TeeOutput(self.original_stdout, self.log_file)
        sys.stderr = TeeOutput(self.original_stderr, self.log_file)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Close log file
        if self.log_file:
            self.log_file.close()






def create_env_file_if_not_exists():
    """Create .env file with template if it doesn't exist."""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print(f"📝 Creating {env_file} file with template...")
        with open(env_file, "w") as f:
            f.write("# GitHub Configuration\n")
            f.write(f"GITHUB_USERNAME={GITHUB_USERNAME}\n")
            f.write("GITHUB_TOKEN=your_github_personal_access_token_here\n")
            f.write("\n# Add other environment variables below\n")
            f.write("# WANDB_API_KEY=\n")
            f.write("# OPENAI_API_KEY=\n")
        
        print(f"✅ Created {env_file} file")
        print(f"⚠ Please edit {env_file} and add your GitHub token")
        return False
    return True




def overwrite_train(parallel_value):
    """Overwrite train.py with the given parallel_flag value."""
    with open(train_file, "w") as f:
        f.write(f"parallel_flag = {parallel_value}\n")
        f.write(f"print('parallel_flag : ', parallel_flag)\n")
    print(f"train.py updated with parallel_flag = {parallel_value}")








def run_torchrun_and_capture(i, capture_all=False):
    """Run the torchrun command and capture all output."""
    cmd = (
        "torchrun --standalone --nproc_per_node=2 main.py "
        "--moe --aux_free --eval --max_iters=250 --eval_interval=50 --attn gqa"
    )
    
    print(f"Running torchrun command for i={i}...")
    
    # Create log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    console_log_file = f"run_{i}_console_{timestamp}.log"
    error_log_file = f"run_{i}_error_{timestamp}.log"
    
    # Ensure directory exists
    os.makedirs(f"run_{i}_logs", exist_ok=True)
    
    # Open log files
    with open(console_log_file, 'w', encoding='utf-8') as console_f, \
         open(error_log_file, 'w', encoding='utf-8') as error_f:
        
        # Write headers
        console_f.write(f"TorchRun Console Output - i={i}\n")
        console_f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        console_f.write("="*80 + "\n\n")
        
        error_f.write(f"TorchRun Error Output - i={i}\n")
        error_f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        error_f.write("="*80 + "\n\n")
        
        # Run command
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Capture stdout and stderr separately
        import threading
        
        def capture_output(stream, output_type="stdout"):
            """Capture output from a stream."""
            log_file = console_f if output_type == "stdout" else error_f
            for line in stream:
                # Print to console
                print(line, end='')
                # Write to appropriate log file
                log_file.write(line)
                log_file.flush()
        
        # Start threads for capturing
        stdout_thread = threading.Thread(target=capture_output, args=(process.stdout, "stdout"))
        stderr_thread = threading.Thread(target=capture_output, args=(process.stderr, "stderr"))
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for threads to complete
        stdout_thread.join()
        stderr_thread.join()
        
        process.wait()
        
        # Write footers
        console_f.write(f"\n\n{'='*80}\n")
        console_f.write(f"Exit Code: {process.returncode}\n")
        console_f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        error_f.write(f"\n\n{'='*80}\n")
        error_f.write(f"Exit Code: {process.returncode}\n")
        error_f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nLogs saved to: {console_log_file} and {error_log_file}")
    return console_log_file, error_log_file



def setup_git_repo():
    """Clone or setup the GitHub repository."""
    print(f"\n{'='*80}")
    print("Setting up GitHub repository...")
    
    # Check if token is available
    if not GITHUB_TOKEN or GITHUB_TOKEN == "your_github_personal_access_token_here":
        print("❌ GitHub token not found in .env file!")
        print("   Please add your token to the .env file:")
        print("   GITHUB_TOKEN=your_actual_token_here")
        return False
    
    # Check if repo already exists
    if os.path.exists(LOCAL_REPO_PATH):
        print(f"Repository already exists at {LOCAL_REPO_PATH}")
        
        # Pull latest changes
        try:
            os.chdir(LOCAL_REPO_PATH)
            # Use token for authentication
            auth_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
            subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
            subprocess.run(["git", "pull", "origin", "main"], check=True)
            print("✓ Pulled latest changes from GitHub")
            os.chdir("..")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Could not pull changes: {e}")
    else:
        # Clone the repository with token authentication
        print(f"Cloning repository with authentication...")
        try:
            auth_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
            subprocess.run(["git", "clone", auth_url], check=True)
            print("✓ Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository: {e}")
            return False
    
    # Ensure monitor_logs directory exists
    os.makedirs(MONITOR_LOGS_DIR, exist_ok=True)
    
    return True

def copy_output_files_to_repo(run_number, output_dir="."):
    """
    Copy all output files from Kaggle's output directory to GitHub repo.
    Modify output_dir based on where Kaggle stores outputs.
    """
    print(f"\nCopying output files for run {run_number}...")
    
    # Create run-specific directory in monitor_logs
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder_name = f"run_{run_number}_{run_timestamp}"
    run_folder_path = os.path.join(MONITOR_LOGS_DIR, run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)
    
    # Common Kaggle output locations (modify as needed)
    kaggle_output_dirs = [
        "/kaggle/working",  # Kaggle's working directory
        "/kaggle/output",   # Kaggle's output directory
        ".",                # Current directory
        "./output",         # Local output directory
        "./logs",           # Logs directory
        f"./run_{run_number}_logs",  # Run-specific logs
    ]
    
    files_copied = []
    
    for output_dir in kaggle_output_dirs:
        if os.path.exists(output_dir):
            print(f"  Scanning: {output_dir}")
            files_to_copy = []
            patterns_to_copy = [
                "wandb/*",                    # All wandb files
                "manifest.json",              # Manifest file
                "*.log",                      # Log files
                "run_*_logs/*",               # Run-specific logs
                "complete_execution_log.txt", # Complete log if exists
            ]

            # Collect files matching patterns
            for pattern in patterns_to_copy:
                import glob
                for file_path in glob.glob(pattern, recursive=True):
                    if os.path.isfile(file_path):  # Only files, not directories
                        files_to_copy.append(file_path)
    
    # Create a manifest file
    manifest = {
        "run_number": run_number,
        "timestamp": run_timestamp,
        "total_files": len(files_copied),
        "files": files_copied,
        "copy_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    manifest_path = os.path.join(run_folder_path, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Copied {len(files_copied)} files to {run_folder_path}")
    return run_folder_path, len(files_copied)








def commit_and_push_to_github(run_number, files_count):
    """Commit changes and push to GitHub."""
    print(f"\n{'='*80}")
    print(f"Committing and pushing to GitHub for run {run_number}...")
    
    try:
        os.chdir(LOCAL_REPO_PATH)
        
        # Add all files
        # ✅ ADD THESE 2 LINES HERE - FORCE GIT CONFIG
        print("Configuring git identity...")
        subprocess.run(["git", "config", "user.email", "arnav2prasad@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Arnav2Prasad"], check=True)
        

        subprocess.run(["git", "add", "."], check=True)
        print("✓ Files added to git")
        
        # Commit with descriptive message
        commit_message = f"Add monitor logs for run {run_number} - {files_count} files - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"✓ Committed: {commit_message}")
        
        # Configure remote with token
        auth_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
        subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
        
        # Push to GitHub
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✓ Pushed to GitHub successfully!")
        
        os.chdir("..")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
        os.chdir("..")
        return False
    except Exception as e:
        print(f"❌ Error during git operations: {e}")
        os.chdir("..")
        return False

def create_summary_file(total_runs, success_runs):
    """Create a summary file of all executions."""
    summary = {
        "total_runs": total_runs,
        "successful_runs": success_runs,
        "execution_date": datetime.now().strftime('%Y-%m-%d'),
        "start_time": start_time,
        "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "runs": []
    }
    
    # Find all run folders
    if os.path.exists(MONITOR_LOGS_DIR):
        for run_folder in os.listdir(MONITOR_LOGS_DIR):
            run_path = os.path.join(MONITOR_LOGS_DIR, run_folder)
            if os.path.isdir(run_path) and run_folder.startswith("run_"):
                manifest_path = os.path.join(run_path, "manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        summary["runs"].append(manifest)
    
    summary_path = os.path.join(MONITOR_LOGS_DIR, "execution_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Execution summary saved to: {summary_path}")
    return summary_path






def main():
    global start_time
    
    # Create a master log file for ALL screen output
    master_log_file = "complete_execution_log.txt"
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Capture ALL output from this point onward
    with OutputCapture(master_log_file) as capture:
        print(f"{'='*80}")
        print("TORCHRUN EXECUTION WITH GITHUB BACKUP")
        print(f"Start Time: {start_time}")
        print(f"GitHub Repo: {GITHUB_REPO_URL}")
        print(f"Complete log will be saved to: {master_log_file}")
        print(f"{'='*80}\n")
        
        # Check and create .env file if needed
        env_exists = create_env_file_if_not_exists()
        
        if not env_exists:
            print("\n⚠ Please configure your .env file before continuing.")
            print("   Required: GITHUB_TOKEN")
            print("\nPress Enter to continue with local execution only...")
            input()
        
        # Configure git globally
        print("Configuring git identity globally...")
        git_email = os.getenv("GIT_USER_EMAIL", "arnav2prasad@example.com")
        git_name = os.getenv("GIT_USER_NAME", "Arnav2Prasad")
        
        subprocess.run(["git", "config", "--global", "user.email", git_email], check=False)
        subprocess.run(["git", "config", "--global", "user.name", git_name], check=False)
        
        # Setup GitHub repository
        github_enabled = False
        if GITHUB_TOKEN and GITHUB_TOKEN != "your_github_personal_access_token_here":
            github_enabled = setup_git_repo()
        else:
            print("\n⚠ GitHub token not configured.")
            print("   Continuing with local execution only...")
        
        success_count = 0
        total_runs = 8
        
        for i in range(1, total_runs + 1):
            print(f"\n{'#'*80}")
            print(f"STARTING RUN {i}/{total_runs}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'#'*80}\n")
            
            try:
                # Overwrite train.py
                overwrite_train(i)
                
                # Run torchrun and capture output
                log_file = run_torchrun_and_capture(i)
                
                # Copy output files to GitHub repo
                run_folder, files_count = copy_output_files_to_repo(i)
                
                # Commit and push to GitHub
                if github_enabled:
                    if commit_and_push_to_github(i, files_count):
                        success_count += 1
                else:
                    print(f"⚠ GitHub not enabled. Files saved locally to: {run_folder}")
                    success_count += 1
                
                print(f"\n{'✓'*40}")
                print(f"RUN {i} COMPLETED SUCCESSFULLY")
                print(f"Files saved: {files_count}")
                if github_enabled:
                    print(f"Pushed to GitHub: ✓")
                print(f"{'✓'*40}")
                
            except Exception as e:
                print(f"\n{'✗'*40}")
                print(f"RUN {i} FAILED: {e}")
                import traceback
                traceback.print_exc()  # This will also be captured!
                print(f"{'✗'*40}")
            
            # Add separation between runs
            if i < total_runs:
                print(f"\n{'='*80}")
                print("WAITING 10 SECONDS BEFORE NEXT RUN...")
                print(f"{'='*80}")
                time.sleep(10)
        
        # Create and push final summary
        if github_enabled and success_count > 0:
            print(f"\n{'='*80}")
            print("CREATING FINAL EXECUTION SUMMARY...")
            summary_file = create_summary_file(total_runs, success_count)
            
            # Also copy the master log file to GitHub
            if os.path.exists(master_log_file):
                print(f"Copying complete execution log to GitHub...")
                shutil.copy2(master_log_file, os.path.join(MONITOR_LOGS_DIR, "complete_execution_log.txt"))
            
            # Push everything to GitHub
            os.chdir(LOCAL_REPO_PATH)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"Add execution summary - {success_count}/{total_runs} runs completed"], check=True)
            
            if GITHUB_TOKEN:
                auth_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
                subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
            
            subprocess.run(["git", "push", "origin", "main"], check=True)
            os.chdir("..")
        
        # Final output
        print(f"\n{'='*80}")
        print("EXECUTION COMPLETE!")
        print(f"{'='*80}")
        print(f"Total runs: {total_runs}")
        print(f"Successful runs: {success_count}")
        print(f"Start time: {start_time}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if github_enabled:
            print(f"\n✅ All logs have been saved to GitHub repository:")
            print(f"  Repository: {GITHUB_REPO_URL}")
            print(f"  Local path: {os.path.abspath(LOCAL_REPO_PATH)}")
            print(f"  Monitor logs: {os.path.abspath(MONITOR_LOGS_DIR)}")
        else:
            print(f"\n⚠ GitHub backup not enabled.")
            print(f"  Local files saved to: {os.path.abspath('.')}")
        
        print(f"\n📋 Complete screen log saved to: {os.path.abspath(master_log_file)}")
        print(f"{'='*80}")
    
    # After context manager exits, print final message
    print(f"\n🎉 Execution finished! Check {master_log_file} for complete logs.")




if __name__ == "__main__":
    # Install required package if not already installed
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv
    
    main()