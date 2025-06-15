import os
import numpy as np
from scipy.io.wavfile import read, write
import random

# Konfigurasi
SOURCE_DIR = "data/train"
OUT_WAV_DIR = "data_neural/wavs"
OUT_META = "data_neural/metadata.csv"
SAMPLE_RATE = 16000
KATA_KUNCI = ["belajar", "makan", "mandi", "tidur", "olahraga", "masak"]

os.makedirs(OUT_WAV_DIR, exist_ok=True)

# Daftar kombinasi kata-kata (bisa kamu tambah)
kalimat_list = [
    ["saya", "belajar"],
    ["saya", "makan"],
    ["saya", "mandi"],
    ["saya", "tidur"],
    ["saya", "olahraga"],
    ["saya", "masak"],
    ["saya", "belajar", "mandi"],
    ["belajar", "mandi", "tidur"],
    ["makan", "mandi", "olahraga"],
    ["masak", "makan", "tidur"]
]

def get_random_file(kata):
    folder = os.path.join(SOURCE_DIR, kata)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    return os.path.join(folder, random.choice(files)) if files else None

def gabungkan_audio(kata_list, output_path):
    full_audio = np.array([], dtype=np.int16)
    for kata in kata_list:
        if kata == "saya":
            # dummy: silent 0.3s
            silent = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.int16)
            full_audio = np.concatenate((full_audio, silent))
            continue
        path = get_random_file(kata)
        if not path:
            print(f"[!] Tidak ditemukan file untuk '{kata}'")
            return None
        sr, audio = read(path)
        if sr != SAMPLE_RATE:
            print(f"[!] Sample rate tidak cocok: {sr}")
            return None
        full_audio = np.concatenate((full_audio, audio, np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.int16)))
    write(output_path, SAMPLE_RATE, full_audio)
    return True

# Proses semua kalimat
metadata_lines = []
for i, kata_list in enumerate(kalimat_list, start=1):
    filename = f"{i:03d}.wav"
    output_path = os.path.join(OUT_WAV_DIR, filename)
    success = gabungkan_audio(kata_list, output_path)
    if success:
        kalimat_text = " ".join(kata_list)
        metadata_lines.append(f"{filename.replace('.wav','')}|{kalimat_text}")
        print(f"[✓] {filename} ← {kalimat_text}")

# Simpan metadata.csv
with open(OUT_META, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_lines))

print("\n✅ Dataset Neural TTS selesai dibuat!")
