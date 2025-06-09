import os
import numpy as np
from scipy.io import wavfile
import random

# Konfigurasi
DATA_DIR = "data/train"
KATA_UNIK = ["makan", "tidur", "mandi", "masak", "olahraga", "belajar"]
SAMPLE_RATE = 16000
JEDA_MS = 300

def text_to_speech(input_text, output_file="output.wav"):
    words = input_text.lower().split()
    combined_audio = np.array([], dtype=np.int16)

    for word in words:
        if word not in KATA_UNIK:
            print(f"[!] Kata '{word}' tidak ditemukan dalam daftar kata unik.")
            continue

        folder = os.path.join(DATA_DIR, word)
        samples = [f for f in os.listdir(folder) if f.endswith('.wav')]
        if not samples:
            print(f"[!] Tidak ada file audio untuk kata '{word}'.")
            continue

        # Pilih file secara acak
        chosen = random.choice(samples)
        audio_path = os.path.join(folder, chosen)
        print(f"[✓] Menambahkan: {audio_path}")
        
        sr, audio = wavfile.read(audio_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Sample rate tidak cocok: {sr} != {SAMPLE_RATE}")

        combined_audio = np.concatenate((combined_audio, audio))

        # Tambah jeda
        silence = np.zeros(int(SAMPLE_RATE * JEDA_MS / 1000), dtype=np.int16)
        combined_audio = np.concatenate((combined_audio, silence))

    # Simpan output
    wavfile.write(output_file, SAMPLE_RATE, combined_audio)
    print(f"\n[✔] Output TTS disimpan di: {output_file}")

# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    print("=== TTS Concatenative ===")
    input_text = input("Ketik kalimat (contoh: belajar mandi makan):\n> ")
    text_to_speech(input_text)
