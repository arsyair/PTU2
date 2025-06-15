import os
import librosa
import numpy as np

DATA_DIR = "data/train"
OUT_DIR = "features"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_features(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db.T  # shape (T, 80)

for word in os.listdir(DATA_DIR):
    word_path = os.path.join(DATA_DIR, word)
    if not os.path.isdir(word_path): continue
    for fname in os.listdir(word_path):
        if fname.endswith('.wav'):
            fpath = os.path.join(word_path, fname)
            feats = extract_features(fpath)
            out_name = f"{word}_{fname.replace('.wav', '')}.npy"
            np.save(os.path.join(OUT_DIR, out_name), feats)

print("✅ Fitur mel spectrogram berhasil diekstrak.")
