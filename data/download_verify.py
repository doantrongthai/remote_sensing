import os
import pandas as pd

OFFICIAL_SPLITS = {
    'flood_train_data.csv': 252,
    'flood_valid_data.csv': 89,
    'flood_test_data.csv': 90,
    'flood_bolivia_data.csv': 15
}

def verify_dataset(root_dir):
    splits_dir = os.path.join(root_dir, 'splits')
    results = {}
    if not os.path.isdir(splits_dir):
        splits_dir = root_dir
    
    for split_file, expected_count in OFFICIAL_SPLITS.items():
        csv_path = os.path.join(splits_dir, split_file)
        if not os.path.exists(csv_path):
            results[split_file] = {'status': 'MISSING', 'expected': expected_count, 'found': 0}
            continue
        df = pd.read_csv(csv_path, header=None)
        actual_count = len(df)
        status = 'MATCH' if actual_count == expected_count else 'MISMATCH'
        results[split_file] = {'status': status, 'expected': expected_count, 'found': actual_count}

    for folder in ['S1Hand', 'S2Hand', 'LabelHand']:
        dir_path = os.path.join(root_dir, folder)
        if os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.tif')]
            results[folder] = {'status': 'MATCH' if len(files) == 446 else 'MISMATCH', 'expected': 446, 'found': len(files)}
        else:
            results[folder] = {'status': 'MISSING', 'expected': 446, 'found': 0}
            
    return results

if __name__ == '__main__':
    candidates = [
        '/kaggle/input/datasets/doantrongthai/sen1floods11',
        '/kaggle/input/sen1floods11',
        '/kaggle/working/data',
        r'D:\Downloads\remote_sensing\sen1floods11_hand_labeled',
        r'D:\Downloads\remote_sensing\sen1floods11_256x256'
    ]
    default_path = next((c for c in candidates if os.path.exists(c)), candidates[0])
    res = verify_dataset(default_path)
    for k, v in res.items():
        print(f"{k}: {v['status']} (Expected: {v['expected']}, Found: {v['found']})")
