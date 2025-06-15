import numpy as np
import os
import librosa
from scipy.io.wavfile import write
import random

FEATURE_DIR = "features"
DATA_DIR = "data/train"
SR = 16000

def synthesize_proxy(word):
    # Cari semua fitur kata tersebut
    files = [f for f in os.listdir(FEATURE_DIR) if f.startswith(word + "_")]
    if not files:
        print(f"[!] Tidak ada fitur untuk kata: {word}")
        return None
    
    # Pilih salah satu secara acak
    file = random.choice(files)
    mel = np.load(os.path.join(FEATURE_DIR, file)).T  # shape (n_mels, T)
    
    # Balikkan ke audio
    mel_power = librosa.db_to_power(mel)
    audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=SR, n_fft=1024, hop_length=256, n_iter=100)
    audio = audio / np.max(np.abs(audio))
    return audio

def synthesize_sentence(sentence, out_path="output_spss_proxy.wav"):
    words = sentence.strip().lower().split()
    result_audio = np.array([], dtype=np.float32)
    
    for word in words:
        audio = synthesize_proxy(word)
        if audio is not None:
            result_audio = np.concatenate((result_audio, audio, np.zeros(int(SR * 0.2))))
    
    write(out_path, SR, (result_audio * 32767).astype(np.int16))
    print(f"✅ Audio disimpan di: {out_path}")

if __name__ == "__main__":
    kalimat = input("Masukkan kalimat:\n> ")
    synthesize_sentence(kalimat)
