# import os
# import time

# # Path to train.py
# train_file = "train.py"

# def overwrite_train(parallel_value):
#     """Overwrite train.py with the given parallel_flag value."""
#     with open(train_file, "w") as f:
#         f.write(f"parallel_flag = {parallel_value}\n")
#         f.write(f"print('parallel_flag : ', parallel_flag)\n")
#     print(f"train.py updated with parallel_flag = {parallel_value}")

# def run_torchrun():
#     """Run the torchrun command."""
#     cmd = (
#         "torchrun --standalone --nproc_per_node=2 main.py "
#         "--moe --aux_free --eval --max_iters=50 --eval_interval=50 --attn gqa"
#     )
#     print("Running torchrun command...")
#     os.system(cmd)

# def main():

#     for i in range(1,9):
#         print('i ->',i)
#         print('i ->',i)
#         print('i ->',i)
#         print('i ->',i)
#         overwrite_train(i)
#         run_torchrun()

#         print("=======\n" * 100)



# if __name__ == "__main__":
#     main()



# # # will the below code work???????
# # # will it send the txt file to the receiver email??????

# # # import os
# # # import time
# # # import subprocess
# # # import smtplib
# # # from email.mime.multipart import MIMEMultipart
# # # from email.mime.text import MIMEText
# # # from email.mime.application import MIMEApplication

# # # # Path to train.py
# # # train_file = "train.py"
# # # log_file = "finally_we_logged_it.txt"

# # # def overwrite_train(parallel_value):
# # #     """Overwrite train.py with the given parallel_flag value."""
# # #     with open(train_file, "w") as f:
# # #         f.write(f"parallel_flag = {parallel_value}\n")
# # #         f.write(f"print('parallel_flag : ', parallel_flag)\n")
# # #     print(f"train.py updated with parallel_flag = {parallel_value}")

# # # def run_torchrun_and_log():
# # #     """Run the torchrun command and capture all output."""
# # #     cmd = (
# # #         "torchrun --standalone --nproc_per_node=2 main.py "
# # #         "--moe --aux_free --eval --max_iters=250 --eval_interval=50 --attn gqa"
# # #     )
    
# # #     print("Running torchrun command...")
# # #     print("-" * 50)
    
# # #     # Open log file in append mode
# # #     with open(log_file, 'a', encoding='utf-8') as log_f:
# # #         # Write separator
# # #         separator = f"\n{'='*80}\nTorchRun Execution - {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
# # #         log_f.write(separator)
# # #         print(separator.strip())
        
# # #         # Run command and capture output in real-time
# # #         process = subprocess.Popen(
# # #             cmd, 
# # #             shell=True, 
# # #             stdout=subprocess.PIPE, 
# # #             stderr=subprocess.STDOUT,  # Combine stderr with stdout
# # #             universal_newlines=True,
# # #             bufsize=1
# # #         )
        
# # #         # Read output line by line and write to both console and file
# # #         for line in process.stdout:
# # #             # Print to console
# # #             print(line, end='')
# # #             # Write to log file
# # #             log_f.write(line)
# # #             log_f.flush()  # Ensure immediate writing
        
# # #         # Wait for process to complete
# # #         process.wait()
        
# # #         # Write exit code
# # #         exit_msg = f"\nProcess completed with exit code: {process.returncode}\n"
# # #         log_f.write(exit_msg)
# # #         print(exit_msg.strip())



# # # '''
# # # def send_email_with_log(receiver_email, sender_email, sender_password, 
# # #                         smtp_server="smtp.office365.com", smtp_port=587):
# # #     """
# # #     Send the log file as a draft email attachment.
    
# # #     Note: For Outlook draft emails, you might need to use Microsoft Graph API.
# # #     This function sends it as a regular email, but you can modify it to save as draft
# # #     if you have Office 365 API credentials.
# # #     """
    
# # #     # Create message
# # #     msg = MIMEMultipart()
# # #     msg['From'] = sender_email
# # #     msg['To'] = receiver_email
# # #     msg['Subject'] = f"TorchRun Execution Log - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
# # #     # Email body
# # #     body = f"""
# # #     <html>
# # #     <body>
# # #         <h2>TorchRun Execution Complete</h2>
# # #         <p>TorchRun execution has completed. Please find the detailed logs attached.</p>
# # #         <p>Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
# # #         <hr>
# # #         <p><i>This is an automated email.</i></p>
# # #     </body>
# # #     </html>
# # #     """
# # #     msg.attach(MIMEText(body, 'html'))
    
# # #     # Attach log file
# # #     try:
# # #         with open(log_file, 'rb') as f:
# # #             attachment = MIMEApplication(f.read(), _subtype="txt")
# # #             attachment.add_header('Content-Disposition', 'attachment', 
# # #                                  filename=f"torchrun_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
# # #             msg.attach(attachment)
# # #     except FileNotFoundError:
# # #         error_msg = "Log file not found. No attachment included."
# # #         msg.attach(MIMEText(error_msg, 'plain'))
# # #         print(error_msg)
    
# # #     # Send email
# # #     try:
# # #         with smtplib.SMTP(smtp_server, smtp_port) as server:
# # #             server.starttls()
# # #             server.login(sender_email, sender_password)
# # #             server.send_message(msg)
# # #         print(f"\nEmail sent successfully to {receiver_email}")
# # #     except Exception as e:
# # #         print(f"\nFailed to send email: {e}")
# # # '''


# # # def send_email_with_log(receiver_email, sender_email, sender_password, 
# # #                         smtp_server="smtp.office365.com", smtp_port=587):
# # #     """
# # #     Send the log file as email attachment with better error handling.
# # #     """
    
# # #     # Validate log file exists and has content
# # #     if not os.path.exists(log_file):
# # #         print(f"✗ Error: Log file '{log_file}' not found!")
# # #         return False
    
# # #     file_size = os.path.getsize(log_file)
# # #     if file_size == 0:
# # #         print(f"⚠ Warning: Log file '{log_file}' is empty!")
    
# # #     print(f"📄 Log file size: {file_size} bytes")
    
# # #     # Create message
# # #     msg = MIMEMultipart()
# # #     msg['From'] = sender_email
# # #     msg['To'] = receiver_email
# # #     msg['Subject'] = f"TorchRun Execution Log - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
# # #     # Email body
# # #     body = f"""
# # #     <html>
# # #     <body>
# # #         <h2>TorchRun Execution Complete</h2>
# # #         <p>TorchRun execution has completed. Please find the detailed logs attached.</p>
# # #         <p>Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
# # #         <p>Log File: {log_file} ({file_size} bytes)</p>
# # #         <hr>
# # #         <p><i>This is an automated email.</i></p>
# # #     </body>
# # #     </html>
# # #     """
# # #     msg.attach(MIMEText(body, 'html'))
    
# # #     # Attach log file
# # #     try:
# # #         with open(log_file, 'rb') as f:
# # #             attachment = MIMEApplication(f.read(), _subtype="txt")
# # #             filename = f"torchrun_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
# # #             attachment.add_header('Content-Disposition', 'attachment', filename=filename)
# # #             msg.attach(attachment)
# # #         print(f"✓ Log file attached: {filename}")
# # #     except Exception as e:
# # #         print(f"✗ Failed to attach log file: {e}")
# # #         return False
    
# # #     # Send email
# # #     try:
# # #         print(f"📧 Connecting to {smtp_server}:{smtp_port}...")
# # #         with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
# # #             print("🔐 Starting TLS...")
# # #             server.starttls()
# # #             print(f"👤 Logging in as {sender_email}...")
# # #             server.login(sender_email, sender_password)
# # #             print("✉️ Sending email...")
# # #             server.send_message(msg)
# # #         print(f"✅ Email sent successfully to {receiver_email}")
# # #         return True
        
# # #     except smtplib.SMTPAuthenticationError:
# # #         print("❌ Authentication failed. Check your email/password.")
# # #         print("   Tip: Use App Password if 2FA is enabled.")
# # #         return False
# # #     except smtplib.SMTPException as e:
# # #         print(f"❌ SMTP Error: {e}")
# # #         return False
# # #     except Exception as e:
# # #         print(f"❌ Unexpected error: {e}")
# # #         return False







# # # def main():
# # #     # Clear or create log file at the beginning
# # #     with open(log_file, 'w', encoding='utf-8') as f:
# # #         f.write(f"TorchRun Execution Log\n")
# # #         f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# # #         f.write(f"{'='*80}\n\n")
    
# # #     print(f"Logging all output to: {log_file}")
    
# # #     # First run
# # #     print("\n" + "="*80)
# # #     print("FIRST EXECUTION: parallel_flag = 8")
# # #     print("="*80)
# # #     overwrite_train(8)
# # #     run_torchrun_and_log()
    
# # #     # Print separator for clarity
# # #     print("\n" + "="*80)
# # #     print("WAITING 5 SECONDS BEFORE NEXT EXECUTION...")
# # #     print("="*80)
# # #     time.sleep(5)
    
# # #     # Second run
# # #     print("\n" + "="*80)
# # #     print("SECOND EXECUTION: parallel_flag = 5")
# # #     print("="*80)
# # #     overwrite_train(5)
# # #     run_torchrun_and_log()
    
# # #     # Send email with log file
# # #     print("\n" + "="*80)
# # #     print("SENDING EMAIL WITH LOG FILE...")
# # #     print("="*80)
    
# # #     # Email configuration - UPDATE THESE VALUES
# # #     receiver_email = "your_email@outlook.com"  # Change this
# # #     sender_email = "your_outlook_email@outlook.com"  # Change this
# # #     sender_password = "your_outlook_password"  # Change this
# # #     smtp_server = "smtp.office365.com"
# # #     smtp_port = 587
    
# # #     send_email_with_log(receiver_email, sender_email, sender_password, 
# # #                         smtp_server, smtp_port)
    
# # #     print("\n" + "="*80)
# # #     print("EXECUTION COMPLETE")
# # #     print(f"All logs saved to: {os.path.abspath(log_file)}")
# # #     print("="*80)





# # # if __name__ == "__main__":
# # #     main()

















# # import os
# # import time
# # import subprocess
# # import smtplib
# # from email.mime.multipart import MIMEMultipart
# # from email.mime.text import MIMEText
# # from email.mime.application import MIMEApplication

# # # Path to train.py
# # train_file = "train.py"
# # log_file = "finally_we_logged_it.txt"



# # def overwrite_train(parallel_value):
# #     """Overwrite train.py with the given parallel_flag value."""
# #     with open(train_file, "w") as f:
# #         f.write(f"parallel_flag = {parallel_value}\n")
# #         f.write(f"print('parallel_flag : ', parallel_flag)\n")
# #     print(f"train.py updated with parallel_flag = {parallel_value}")



# # def run_torchrun_and_log():
# #     """Run the torchrun command and capture all output."""
# #     cmd = (
# #         "torchrun --standalone --nproc_per_node=2 main.py "
# #         "--moe --aux_free --eval --max_iters=250 --eval_interval=50 --attn gqa"
# #     )
    
# #     print("Running torchrun command...")
# #     print("-" * 50)
    
# #     # Open log file in append mode
# #     with open(log_file, 'a', encoding='utf-8') as log_f:
# #         # Write separator
# #         separator = f"\n{'='*80}\nTorchRun Execution - {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
# #         log_f.write(separator)
# #         print(separator.strip())
        
# #         # Run command and capture output in real-time
# #         process = subprocess.Popen(
# #             cmd, 
# #             shell=True, 
# #             stdout=subprocess.PIPE, 
# #             stderr=subprocess.STDOUT,  # Combine stderr with stdout
# #             universal_newlines=True,
# #             bufsize=1
# #         )
        
# #         # Read output line by line and write to both console and file
# #         for line in process.stdout:
# #             # Print to console
# #             print(line, end='')
# #             # Write to log file
# #             log_f.write(line)
# #             log_f.flush()  # Ensure immediate writing
        
# #         # Wait for process to complete
# #         process.wait()
        
# #         # Write exit code
# #         exit_msg = f"\nProcess completed with exit code: {process.returncode}\n"
# #         log_f.write(exit_msg)
# #         print(exit_msg.strip())

# # def send_email_with_log(receiver_email, sender_email, sender_password, 
# #                         smtp_server="smtp.office365.com", smtp_port=587):
# #     """
# #     Send the log file as a draft email attachment.
# #     """
    
# #     # Create message
# #     msg = MIMEMultipart()
# #     msg['From'] = sender_email
# #     msg['To'] = receiver_email
# #     msg['Subject'] = f"TorchRun Execution Log - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
# #     # Email body
# #     body = f"""
# #     <html>
# #     <body>
# #         <h2>TorchRun Execution Complete</h2>
# #         <p>TorchRun execution has completed for parallel_flag values 1 through 8.</p>
# #         <p>Total executions: 8</p>
# #         <p>Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
# #         <p>Please find the detailed logs attached.</p>
# #         <hr>
# #         <p><i>This is an automated email.</i></p>
# #     </body>
# #     </html>
# #     """
# #     msg.attach(MIMEText(body, 'html'))
    
# #     # Attach log file
# #     try:
# #         with open(log_file, 'rb') as f:
# #             attachment = MIMEApplication(f.read(), _subtype="txt")
# #             attachment.add_header('Content-Disposition', 'attachment', 
# #                                  filename=f"torchrun_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
# #             msg.attach(attachment)
# #     except FileNotFoundError:
# #         error_msg = "Log file not found. No attachment included."
# #         msg.attach(MIMEText(error_msg, 'plain'))
# #         print(error_msg)
    
# #     # Send email
# #     try:
# #         with smtplib.SMTP(smtp_server, smtp_port) as server:
# #             server.starttls()
# #             server.login(sender_email, sender_password)
# #             server.send_message(msg)
# #         print(f"\nEmail sent successfully to {receiver_email}")
# #     except Exception as e:
# #         print(f"\nFailed to send email: {e}")

# # def create_significant_demarcation(run_number, total_runs, parallel_value, wait_time=5):
# #     """
# #     Create a VERY SIGNIFICANT demarcation between runs.
# #     """
# #     demarcation = f"""
    
# # {'#'*100}
# # {'#'*100}
# # {'#'*100}
# #                             RUN {run_number}/{total_runs} COMPLETED
# #                             parallel_flag = {parallel_value}
# #                             Waiting {wait_time} seconds...
# # {'#'*100}
# # {'#'*100}
# # {'#'*100}

# # """
# #     print(demarcation)
    
# #     # Also write to log file
# #     with open(log_file, 'a', encoding='utf-8') as log_f:
# #         log_f.write(demarcation)
    
# #     # Wait before next run
# #     if run_number < total_runs:  # Don't wait after the last run
# #         print(f"⏳ Waiting {wait_time} seconds before next execution...")
# #         for i in range(wait_time, 0, -1):
# #             print(f"   {i}...", end=' ', flush=True)
# #             time.sleep(1)
# #         print("\n")

# # def main():
# #     # Clear or create log file at the beginning
# #     with open(log_file, 'w', encoding='utf-8') as f:
# #         f.write(f"TORCHRUN EXECUTION LOG - ALL RUNS\n")
# #         f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
# #         f.write(f"Testing parallel_flag values from 1 to 8\n")
# #         f.write(f"{'='*100}\n\n")
    
# #     print(f"Logging all output to: {log_file}")
    
# #     # Define parallel_flag values to test (1 to 8)
# #     parallel_values = list(range(1, 9))  # [1, 2, 3, 4, 5, 6, 7, 8]
# #     total_runs = len(parallel_values)
    
# #     print("\n" + "="*100)
# #     print(f"STARTING TORCHRUN EXECUTION FOR {total_runs} DIFFERENT parallel_flag VALUES")
# #     print(f"Values to test: {parallel_values}")
# #     print("="*100 + "\n")
    
# #     # Run for each parallel_flag value
# #     for run_number, parallel_value in enumerate(parallel_values, 1):
# #         # VERY SIGNIFICANT START DEMARCATION
# #         start_demarcation = f"""
# # {'*'*100}
# # {'*'*100}
# #                     STARTING RUN {run_number}/{total_runs}
# #                     parallel_flag = {parallel_value}
# #                     {time.strftime('%Y-%m-%d %H:%M:%S')}
# # {'*'*100}
# # {'*'*100}
# # """
# #         print(start_demarcation)
        
# #         # Write start demarcation to log file
# #         with open(log_file, 'a', encoding='utf-8') as log_f:
# #             log_f.write(start_demarcation)
        
# #         # Update train.py and run torchrun
# #         overwrite_train(parallel_value)
# #         run_torchrun_and_log()
        
# #         # VERY SIGNIFICANT END DEMARCATION (except after last run)
# #         if run_number < total_runs:
# #             create_significant_demarcation(run_number, total_runs, parallel_value, wait_time=5)
# #         else:
# #             # Final completion demarcation
# #             final_demarcation = f"""
            
# # {'!'*100}
# # {'!'*100}
# # {'!'*100}
# #                     ALL {total_runs} RUNS COMPLETED!
# #                     Final parallel_flag = {parallel_value}
# #                     {time.strftime('%Y-%m-%d %H:%M:%S')}
# # {'!'*100}
# # {'!'*100}
# # {'!'*100}
# # """
# #             print(final_demarcation)
# #             with open(log_file, 'a', encoding='utf-8') as log_f:
# #                 log_f.write(final_demarcation)
    
# #     # Send email with log file
# #     print("\n" + "="*100)
# #     print("SENDING EMAIL WITH LOG FILE...")
# #     print("="*100)
    
# #     # Email configuration - UPDATE THESE VALUES
# #     receiver_email = "arnavp22.comp@coeptech.ac.in"  # Change this
# #     sender_email = "arnavp22.comp@coeptech.ac.in"  # Change this
# #     sender_password = "your_outlook_password"  # Change this
# #     smtp_server = "smtp.office365.com"
# #     smtp_port = 587
    
# #     send_email_with_log(receiver_email, sender_email, sender_password, 
# #                         smtp_server, smtp_port)
    
# #     # Final summary
# #     print("\n" + "="*100)
# #     print("EXECUTION SUMMARY")
# #     print("="*100)
# #     print(f"✓ Total runs completed: {total_runs}")
# #     print(f"✓ parallel_flag values tested: {parallel_values}")
# #     print(f"✓ All logs saved to: {os.path.abspath(log_file)}")
    
# #     # Show file size
# #     if os.path.exists(log_file):
# #         file_size = os.path.getsize(log_file)
# #         print(f"✓ Log file size: {file_size} bytes ({file_size/1024:.2f} KB)")
    
# #     print(f"✓ Email sent to: {receiver_email}")
# #     print("="*100)

# # if __name__ == "__main__":
# #     main()













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




def run_torchrun_and_capture(i):
    """Run the torchrun command and capture all output."""
    cmd = (
        "torchrun --standalone --nproc_per_node=2 main.py "
        "--moe --aux_free --eval --max_iters=50 --eval_interval=50 --attn gqa"
    )
    
    print(f"Running torchrun command for i={i}...")
    
    # Create a unique log file for this run
    log_filename = f"torchrun_i{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(f"run_{i}_logs", log_filename)
    
    # Ensure directory exists
    os.makedirs(f"run_{i}_logs", exist_ok=True)
    
    # Run command and capture output
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    with open(log_filepath, 'w', encoding='utf-8') as log_f:
        # Write header
        log_f.write(f"TorchRun Execution - i={i}\n")
        log_f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("="*80 + "\n\n")
        
        # Capture output in real-time
        for line in process.stdout:
            # Print to console
            print(line, end='')
            # Write to log file
            log_f.write(line)
            log_f.flush()
    
    process.wait()
    
    # Add footer
    with open(log_filepath, 'a', encoding='utf-8') as log_f:
        log_f.write(f"\n\n{'='*80}\n")
        log_f.write(f"Exit Code: {process.returncode}\n")
        log_f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nLog saved to: {log_filepath}")
    return log_filepath

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
            for root, dirs, files in os.walk(output_dir):
                # Skip .git directories
                if '.git' in root:
                    continue
                    
                for file in files:
                    # Skip certain file types
                    if any(file.endswith(ext) for ext in ['.pyc', '.gitignore', '.git']):
                        continue
                    
                    source_path = os.path.join(root, file)
                    # Create relative path
                    rel_path = os.path.relpath(source_path, output_dir)
                    dest_path = os.path.join(run_folder_path, rel_path)
                    
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        files_copied.append(rel_path)
                        # print(f"    ✓ Copied: {rel_path}")
                    except Exception as e:
                        print(f"    ✗ Failed to copy {rel_path}: {e}")

                    print('all files have been copied.....')
    
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
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"{'='*80}")
    print("TORCHRUN EXECUTION WITH GITHUB BACKUP")
    print(f"Start Time: {start_time}")
    print(f"GitHub Repo: {GITHUB_REPO_URL}")
    print(f"{'='*80}\n")
    
    # Check and create .env file if needed
    env_exists = create_env_file_if_not_exists()
    
    if not env_exists:
        print("\n⚠ Please configure your .env file before continuing.")
        print("   Required: GITHUB_TOKEN")
        print("\nPress Enter to continue with local execution only...")
        input()
    
    # Setup GitHub repository (only if token exists)
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
            
            # Copy output files to GitHub repo (if enabled)
            run_folder, files_count = copy_output_files_to_repo(i)
            
            # Commit and push to GitHub (if enabled)
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
            print(f"{'✗'*40}")
        
        # Add separation between runs (except after last run)
        if i < total_runs:
            print(f"\n{'='*80}")
            print("WAITING 10 SECONDS BEFORE NEXT RUN...")
            print(f"{'='*80}")
            time.sleep(10)
    
    # Create and push final summary (if GitHub enabled)
    if github_enabled and success_count > 0:
        print(f"\n{'='*80}")
        print("CREATING FINAL EXECUTION SUMMARY...")
        summary_file = create_summary_file(total_runs, success_count)
        
        # Push the summary to GitHub
        os.chdir(LOCAL_REPO_PATH)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"Add execution summary - {success_count}/{total_runs} runs completed"], check=True)
        
        # Ensure remote URL is set with token
        auth_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
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
        print(f"  To enable GitHub backup, edit .env file and add your token.")
    
    print(f"{'='*80}")





if __name__ == "__main__":
    # Install required package if not already installed
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv
    
    main()