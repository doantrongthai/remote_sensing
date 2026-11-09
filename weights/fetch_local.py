import os
import sys
import hashlib
import json
import argparse
from huggingface_hub import hf_hub_download

def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def fetch_weights(repo_id, model, run_id, dest_dir, token=None):
    target_dir = os.path.join(dest_dir, model, run_id)
    os.makedirs(target_dir, exist_ok=True)
    
    token = token or os.environ.get('HF_TOKEN')
    
    manifest_remote = f"{model}/{run_id}/manifest.json"
    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename=manifest_remote,
            token=token,
            local_dir=target_dir
        )
    except Exception as e:
        print(f"Could not download manifest: {e}")
        return False
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    for fname, expected_hash in manifest.get('files', {}).items():
        dst_file = os.path.join(target_dir, fname)
        if os.path.exists(dst_file):
            current_hash = calculate_sha256(dst_file)
            if current_hash == expected_hash:
                print(f"File {fname} already exists and hash matches. Skipping.")
                continue
                
        remote_path = f"{model}/{run_id}/{fname}"
        print(f"Downloading {remote_path}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            token=token,
            local_dir=target_dir
        )
        downloaded_hash = calculate_sha256(downloaded)
        if downloaded_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {fname}: expected {expected_hash}, got {downloaded_hash}")
        print(f"Verified {fname}: SHA256 {downloaded_hash[:8]}...")
        
    print(f"Weights fetched and verified successfully in: {target_dir}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, default='doantrongthai/sen1floods11-weights')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, default='./weights')
    parser.add_argument('--token', type=str, default='')
    args = parser.parse_args()
    
    fetch_weights(args.repo_id, args.model, args.run_id, args.dest_dir, args.token)
