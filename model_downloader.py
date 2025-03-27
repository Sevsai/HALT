import os
import requests
from tqdm import tqdm
import argparse
import sys
import json

def download_model_manually(model_id, download_dir, token=None, force_bin=False):
    """Download a model manually"""
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Downloading model {model_id} to {download_dir}")
    print("This may take a while depending on the model size...")
    
    try:
        # Method 1: Use huggingface_hub if available
        try:
            from huggingface_hub import snapshot_download
            print("Using huggingface_hub for download...")
            
            # Only ignore .bin files if safetensors is available and we don't force .bin format
            has_safetensors = check_safetensors_available(model_id, token)
            ignore_patterns = None
            
            if has_safetensors and not force_bin:
                print("SafeTensors format available - using it for faster loading")
                ignore_patterns = ["*.bin"]
            elif force_bin:
                print("Forcing .bin format download (ignoring SafeTensors)")
                ignore_patterns = ["*.safetensors"]
            
            snapshot_download(
                repo_id=model_id,
                local_dir=download_dir,
                local_dir_use_symlinks=False,
                token=token,
                ignore_patterns=ignore_patterns
            )
            print(f"Model downloaded to {download_dir}")
            return True
        except ImportError:
            print("huggingface_hub not available, trying manual download...")
        except Exception as e:
            print(f"Snapshot download failed: {e}")
            print("Falling back to manual download...")
        
        # Method 2: Manual file-by-file download
        # Get model files list
        print("Getting file list from HuggingFace API...")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        files_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        r = requests.get(files_url, headers=headers)
        if r.status_code != 200:
            print(f"Failed to get file list: {r.status_code}")
            print(f"Response: {r.text}")
            return False
            
        files = r.json()
        
        # Check if safetensors are available
        has_safetensors = any(f["path"].endswith(".safetensors") for f in files if f["type"] == "file")
        has_bin = any(f["path"].endswith(".bin") for f in files if f["type"] == "file")
        
        if not has_safetensors and not has_bin:
            print("Warning: No model weight files (.safetensors or .bin) found in repository")
        
        format_used = "unknown"
        # Filter files based on available formats and user preference
        if has_safetensors and not force_bin:
            print("SafeTensors format available - using it for faster loading")
            # Only filter out .bin files that have a corresponding .safetensors version
            files = [f for f in files if f["type"] != "file" or 
                    not f["path"].endswith(".bin") or 
                    not any(sf["path"].replace(".safetensors", ".bin") == f["path"] 
                           for sf in files if sf["type"] == "file" and sf["path"].endswith(".safetensors"))]
            format_used = "safetensors"
        elif force_bin and has_bin:
            print("Forcing .bin format download (ignoring SafeTensors if available)")
            # Filter out .safetensors files
            files = [f for f in files if f["type"] != "file" or not f["path"].endswith(".safetensors")]
            format_used = "bin"
        else:
            print("Using available model format")
            if has_bin:
                format_used = "bin"
            elif has_safetensors:
                format_used = "safetensors"
        
        total_files = len([f for f in files if f["type"] == "file"])
        print(f"Found {total_files} files to download")
        
        success_count = 0
        
        # Create marker file to indicate this is a manually downloaded model
        with open(os.path.join(download_dir, "offline_download.json"), "w") as f:
            json.dump({
                "model_id": model_id,
                "download_date": str(os.path.getmtime(__file__)),
                "download_tool": "model_downloader.py",
                "format_used": format_used
            }, f, indent=2)
        
        # Create subdirectories if needed
        subdirs = set()
        for file in files:
            if file["type"] == "file":
                path_parts = file["path"].split("/")
                if len(path_parts) > 1:  # Has subdirectories
                    subdir = os.path.join(download_dir, *path_parts[:-1])
                    subdirs.add(subdir)
        
        for subdir in subdirs:
            os.makedirs(subdir, exist_ok=True)
        
        # Download each file
        for file in tqdm(files, desc="Files"):
            if file["type"] == "file":
                # Get file path
                file_path = file["path"]
                file_url = f"https://huggingface.co/{model_id}/resolve/main/{file_path}"
                
                # Determine local path
                if "/" in file_path:
                    path_parts = file_path.split("/")
                    local_dir = os.path.join(download_dir, *path_parts[:-1])
                    local_path = os.path.join(local_dir, path_parts[-1])
                else:
                    local_path = os.path.join(download_dir, file_path)
                
                # Skip if file already exists
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    print(f"Skipping existing file: {file_path}")
                    success_count += 1
                    continue
                
                try:
                    # Download file
                    print(f"Downloading {file_path}")
                    with requests.get(file_url, stream=True, headers=headers) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        
                        # Show progress bar for large files
                        chunk_size = 8192
                        with open(local_path, 'wb') as f:
                            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(file_path)) as pbar:
                                for chunk in r.iter_content(chunk_size=chunk_size):
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                        
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error downloading {file_path}: {e}")
                
        print(f"Downloaded {success_count} of {total_files} files to {download_dir}")
        return success_count > 0
    
    except Exception as e:
        print(f"Manual download failed: {e}")
        return False

def check_safetensors_available(model_id, token=None):
    """Check if safetensors format is available for the model"""
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        files_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
        r = requests.get(files_url, headers=headers)
        if r.status_code == 200:
            files = r.json()
            return any(f["path"].endswith(".safetensors") for f in files if f["type"] == "file")
    except:
        pass
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from HuggingFace")
    parser.add_argument("--model", type=str, help="Model ID (e.g., 'NousResearch/DeepHermes-3-Mistral-24B-Preview')")
    parser.add_argument("--dir", type=str, help="Download directory")
    parser.add_argument("--token", type=str, help="HuggingFace token (for private/gated models)")
    parser.add_argument("--force-bin", action="store_true", help="Force download of .bin format even if safetensors is available")
    args = parser.parse_args()

    if not args.model or not args.dir:
        model_id = input("Enter model ID (e.g., 'NousResearch/DeepHermes-3-Mistral-24B-Preview'): ")
        download_dir = input("Enter download directory: ")
        token = input("Enter HuggingFace token (leave empty for public models): ")
        token = token if token else None
        force_bin = input("Force download of .bin format? (y/n): ").lower() == 'y'
    else:
        model_id = args.model
        download_dir = args.dir
        token = args.token
        force_bin = args.force_bin
    
    success = download_model_manually(model_id, download_dir, token, force_bin)
    if success:
        print(f"Model download complete! Now you can use it by:")
        print(f"1. Enabling 'Offline Mode' in the application")
        print(f"2. Setting the model path to: {os.path.abspath(download_dir)}")
        sys.exit(0)
    else:
        print("Failed to download the model")
        sys.exit(1)
