# # test_drive_upload.py
# from google_drive_module import upload_to_drive  # Your upload function
# import os

# # Create a test file
# test_file = "test_upload.txt"
# with open(test_file, 'w') as f:
#     f.write("Test content for Google Drive upload\n")

# # Try uploading
# folder_id = "1uKabfClBuho-YxMJO4Iq7gmMiWd2f3AX"
# link, file_id = upload_to_drive(test_file, folder_id)

# if link:
#     print(f"✅ Upload successful!")
#     print(f"🔗 Link: {link}")
# else:
#     print("❌ Upload failed - check permissions/credentials")

# # Cleanup
# if os.path.exists(test_file):
#     os.remove(test_file)

import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Google Drive API Scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """Authenticate and create Google Drive API service."""
    creds = None
    
    # Token file stores the user's access and refresh tokens
    token_file = 'token.pickle'
    
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
        print(f"✓ Loaded credentials from {token_file}")
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print("🔑 No valid credentials found. Starting OAuth flow...")
            if not os.path.exists('credentials.json'):
                print("❌ ERROR: 'credentials.json' file not found!")
                print("   Please download credentials from Google Cloud Console:")
                print("   1. Go to https://console.cloud.google.com/")
                print("   2. Create credentials → OAuth 2.0 Client ID → Desktop app")
                print("   3. Download as 'credentials.json'")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)
        
        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
        print(f"✓ Saved credentials to {token_file}")
    
    try:
        service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive API authenticated successfully")
        return service
    except Exception as e:
        print(f"❌ Failed to build Google Drive service: {e}")
        return None

def upload_to_drive(file_path, folder_id=None, make_public=True):
    """
    Upload a file to Google Drive.
    
    Args:
        file_path: Path to the file to upload
        folder_id: Google Drive folder ID (optional)
        make_public: Whether to make the file publicly accessible (default: True)
    
    Returns:
        tuple: (shareable_link, file_id) or (None, None) if failed
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None
    
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    print(f"\n📤 Uploading '{file_name}' to Google Drive...")
    print(f"   Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    if folder_id:
        print(f"   Target folder ID: {folder_id}")
    
    service = authenticate_google_drive()
    if not service:
        print("❌ Google Drive authentication failed")
        return None, None
    
    try:
        # Create file metadata
        file_metadata = {
            'name': file_name,
            'description': f'TorchRun execution log uploaded on {os.path.basename(__file__)}',
            'mimeType': 'text/plain'
        }
        
        # Add folder ID if provided
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Upload file with progress tracking
        media = MediaFileUpload(
            file_path,
            mimetype='text/plain',
            resumable=True,
            chunksize=1024*1024  # 1MB chunks
        )
        
        print("   Uploading...", end=' ', flush=True)
        request = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, size, createdTime, webViewLink, webContentLink'
        )
        
        # Execute upload
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"\r   Uploading... {progress}% complete", end='', flush=True)
        
        print(f"\r   Uploading... 100% complete ✓")
        
        file_id = response.get('id')
        file_name = response.get('name')
        created_time = response.get('createdTime')
        web_view_link = response.get('webViewLink')
        
        print(f"✅ File uploaded successfully!")
        print(f"   File ID: {file_id}")
        print(f"   Created: {created_time}")
        
        # Make the file publicly viewable if requested
        if make_public and file_id:
            try:
                print("   Setting public permissions...", end=' ', flush=True)
                permission = {
                    'type': 'anyone',
                    'role': 'reader',
                    'allowFileDiscovery': False
                }
                service.permissions().create(
                    fileId=file_id,
                    body=permission,
                    fields='id'
                ).execute()
                print("✓")
            except Exception as perm_error:
                print(f"\n⚠ Warning: Could not set public permissions: {perm_error}")
        
        # Get the final shareable link
        if make_public:
            shareable_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        else:
            shareable_link = web_view_link
        
        print(f"🔗 Shareable link: {shareable_link}")
        
        return shareable_link, file_id
        
    except HttpError as error:
        print(f"\n❌ Google Drive API error: {error}")
        
        # Provide more specific error messages
        if error.resp.status == 404:
            print("   Folder not found. Please check the folder ID.")
        elif error.resp.status == 403:
            print("   Permission denied. Check if the service account has access to the folder.")
        elif error.resp.status == 400:
            print("   Bad request. Check file path and folder ID format.")
        
        return None, None
        
    except Exception as e:
        print(f"\n❌ Unexpected error during upload: {e}")
        return None, None

def upload_to_drive_simple(file_path, folder_id=None):
    """
    Simplified version of upload_to_drive with minimal dependencies.
    Use this if you don't want to use the resumable upload.
    """
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None
    
    service = authenticate_google_drive()
    if not service:
        return None, None
    
    try:
        file_name = os.path.basename(file_path)
        
        # Create file metadata
        file_metadata = {
            'name': file_name,
            'mimeType': 'text/plain'
        }
        
        # Add folder ID if provided
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Upload file
        media = MediaFileUpload(file_path, mimetype='text/plain')
        
        print(f"Uploading {file_name}...", end=' ', flush=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        print("✓")
        
        file_id = file.get('id')
        
        # Make the file publicly viewable
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(
            fileId=file_id,
            body=permission
        ).execute()
        
        shareable_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        
        return shareable_link, file_id
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

# Test function
def test_drive_upload():
    """Test the Google Drive upload functionality."""
    print("🧪 Testing Google Drive Upload Functionality")
    print("="*60)
    
    # Create a test file
    test_file = "test_drive_upload.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test file for Google Drive upload functionality.\n")
        f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📄 Created test file: {test_file}")
    
    # Test upload to root folder
    print("\n1. Testing upload to root folder (no folder_id):")
    link1, file_id1 = upload_to_drive_simple(test_file)
    
    if link1:
        print(f"✅ Success! Link: {link1}")
    else:
        print("❌ Failed to upload to root folder")
    
    # Test upload to specific folder (optional)
    print("\n2. Testing upload to specific folder (if folder_id provided):")
    folder_id = input("Enter folder ID (or press Enter to skip): ").strip()
    
    if folder_id:
        link2, file_id2 = upload_to_drive_simple(test_file, folder_id)
        if link2:
            print(f"✅ Success! Link: {link2}")
        else:
            print("❌ Failed to upload to specified folder")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n🧹 Cleaned up test file: {test_file}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    import time
    test_drive_upload()