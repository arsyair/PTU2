import os
import librosa
from scipy.io.wavfile import write

TARGET_SR = 16000
DATA_PATH = "data"  # ganti jika perlu

def resample_and_save(path):
    for folder_type in ["train", "test"]:
        folder_path = os.path.join(DATA_PATH, folder_type)
        for word in os.listdir(folder_path):
            word_path = os.path.join(folder_path, word)
            if not os.path.isdir(word_path):
                continue
            for fname in os.listdir(word_path):
                if not fname.endswith(".wav"):
                    continue
                file_path = os.path.join(word_path, fname)
                print(f"Resampling: {file_path}")
                y, sr = librosa.load(file_path, sr=TARGET_SR)
                y_int16 = (y * 32767).astype("int16")
                write(file_path, TARGET_SR, y_int16)

resample_and_save(DATA_PATH)
print("✅ Semua file berhasil diresample ke 16000 Hz")
