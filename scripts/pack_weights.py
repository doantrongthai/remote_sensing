import os
import sys
import hashlib
import json
import zipfile
import argparse
from huggingface_hub import HfApi

def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def pack_and_upload(run_dir, model_name, run_id, repo_id='doantrongthai/sen1floods11-weights', token=None):
    token = token or os.environ.get('HF_TOKEN')
    
    files_to_pack = {}
    for fname in ['best.pt', 'last.pt', 'config.yaml']:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            files_to_pack[fname] = calculate_sha256(fpath)
            
    manifest = {
        'model': model_name,
        'run_id': run_id,
        'files': files_to_pack
    }
    manifest_path = os.path.join(run_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    uploaded = False
    if token:
        try:
            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True, token=token)
            for fname in list(files_to_pack.keys()) + ['manifest.json']:
                fpath = os.path.join(run_dir, fname)
                remote_path = f"{model_name}/{run_id}/{fname}"
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type='model',
                    token=token
                )
            print(f"Successfully uploaded weights to HF private repo: {repo_id}/{model_name}/{run_id}")
            uploaded = True
        except Exception as e:
            print(f"Failed to upload to Hugging Face: {e}")
            
    if not uploaded:
        zip_path = os.path.join(run_dir, f"{model_name}_{run_id}_weights.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in list(files_to_pack.keys()) + ['manifest.json']:
                fpath = os.path.join(run_dir, fname)
                zf.write(fpath, arcname=fname)
        print(f"Fallback: Created local zip bundle at {zip_path}")
        return zip_path
        
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--repo_id', type=str, default='doantrongthai/sen1floods11-weights')
    parser.add_argument('--token', type=str, default='')
    args = parser.parse_args()
    
    pack_and_upload(args.run_dir, args.model, args.run_id, args.repo_id, args.token)
