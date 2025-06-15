import os
import numpy as np
import pickle

FEATURE_DIR = "features"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_hmm(data_list, n_states=3):
    D = data_list[0].shape[1]  # jumlah dimensi fitur (mel = 80)
    state_feats = [[] for _ in range(n_states)]

    # Bagi setiap sample ke beberapa state berdasarkan posisi
    for d in data_list:
        l = len(d)
        for i in range(n_states):
            start = int(i * l / n_states)
            end = int((i + 1) * l / n_states)
            state_feats[i].append(d[start:end])

    # Gabungkan semua frame per state
    state_feats = [np.vstack(f) for f in state_feats]

    # Hitung mean dan cov untuk tiap state
    means = [np.mean(s, axis=0) for s in state_feats]
    covs = []
    for s in state_feats:
        try:
            c = np.cov(s.T) + 1e-3 * np.eye(D)
            np.linalg.cholesky(c)  # cek valid
        except np.linalg.LinAlgError:
            print("[!] Covariance tidak stabil — fallback ke diagonal.")
            c = np.diag(np.var(s, axis=0) + 1e-3)
        covs.append(c)

    # Transition probabilities sederhana
    trans = np.full((n_states, n_states), 1e-5)
    for i in range(n_states - 1):
        trans[i, i] = 0.5
        trans[i, i + 1] = 0.5
    trans[-1, -1] = 1.0

    return {"means": means, "covs": covs, "trans": trans}

# Mulai proses training
grouped_data = {}

# Kumpulkan semua file fitur per kata
for f in os.listdir(FEATURE_DIR):
    if not f.endswith(".npy"):
        continue
    word = f.split("_")[0]
    if word not in grouped_data:
        grouped_data[word] = []
    feats = np.load(os.path.join(FEATURE_DIR, f))
    grouped_data[word].append(feats)

# Latih model per kata
for word, feats in grouped_data.items():
    model = train_hmm(feats, n_states=3)
    with open(os.path.join(MODEL_DIR, f"{word}.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model HMM untuk kata '{word}' disimpan.")
