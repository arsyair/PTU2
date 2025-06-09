import os
import librosa
import numpy as np

DATA_DIR = "data/train"
FEATURE_DIR = "features"
SAMPLE_RATE = 16000

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nan_to_num(pitch)
    return mfcc.T, pitch


def save_features(mfcc, pitch, save_path):
    # Simpan sebagai dictionary numpy
    np.savez(save_path, mfcc=mfcc, pitch=pitch)

def process_all():
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)

    for kata in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, kata)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nMemproses kata: {kata}")
        feature_kata_dir = os.path.join(FEATURE_DIR, kata)
        if not os.path.exists(feature_kata_dir):
            os.makedirs(feature_kata_dir)

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not files:
            print(f"  [!] Tidak ada file WAV di {folder_path}")
            continue

        print(f"  File WAV ditemukan: {files}")

        for file in files:
            audio_path = os.path.join(folder_path, file)
            mfcc, pitch = extract_features(audio_path)

            feature_filename = os.path.splitext(file)[0] + ".npz"
            save_path = os.path.join(feature_kata_dir, feature_filename)

            save_features(mfcc, pitch, save_path)
            print(f"    - Fitur disimpan di {save_path}")

if __name__ == "__main__":
    process_all()
