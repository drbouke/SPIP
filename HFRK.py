import numpy as np
import hashlib
import blake3
from Crypto.Hash import SHAKE256
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.stats import chisquare
import math
import random

# ---------- Core FKR Components ----------

def generate_affine_map(a11, a12, a21, a22, bmin, bmax):
    A = np.array([[a11, a12], [a21, a22]])
    b = np.random.uniform(bmin, bmax, size=(2,))
    return A, b

def chaotic_walk(x0, n, epsilon, a_params, b_range):
    path = [x0]
    for _ in range(n):
        A, b = generate_affine_map(*a_params, *b_range)
        delta = np.random.uniform(-epsilon, epsilon, size=(2,))
        x_next = np.floor(A @ path[-1] + b + delta)
        path.append(x_next)
    return path

def flatten_path_bytes(path):
    return b''.join(
        int(round(v)).to_bytes(4, byteorder='little', signed=True)
        for point in path for v in point
    )

def hash_path(path, algo="SHA3"):
    data = flatten_path_bytes(path)
    if algo == "SHA3":
        return hashlib.sha3_512(data).hexdigest()
    elif algo == "SHAKE256":
        h = SHAKE256.new()
        h.update(data)
        return h.read(64).hex()
    elif algo == "BLAKE3":
        return blake3.blake3(data).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm")

def hamming_distance(hex1, hex2):
    b1 = bin(int(hex1, 16))[2:].zfill(len(hex1)*4)
    b2 = bin(int(hex2, 16))[2:].zfill(len(hex2)*4)
    dist = sum(c1 != c2 for c1, c2 in zip(b1, b2))
    bit_vec = np.array([int(c1 != c2) for c1, c2 in zip(b1, b2)])
    return dist, bit_vec

def compute_bit_flip_rate(hamming_dist, hash_bits=512):
    return hamming_dist / hash_bits

def compute_entropy(hexstr):
    b = bin(int(hexstr, 16))[2:].zfill(len(hexstr)*4)
    counts = np.bincount([int(bit) for bit in b], minlength=2)
    probs = counts / counts.sum()
    return -sum(p * math.log2(p) for p in probs if p > 0)

def compute_chi_square(bit_matrix):
    counts = bit_matrix.sum(axis=0)
    total = counts.sum()
    expected = np.full(counts.shape, total / counts.size)
    return chisquare(counts, f_exp=expected)

# ---------- Visualization ----------

def smooth_plot_path(path, title="Chaotic Walk Trajectory"):
    coords = np.array(path)
    x, y = coords[:,0], coords[:,1]
    plt.figure(figsize=(7,5))
    try:
        tck, u = splprep([x,y], s=0)
        xs, ys = splev(np.linspace(0,1,300), tck)
        plt.plot(xs, ys, label="Path", color="darkblue")
    except Exception:
        plt.plot(x, y, marker='o', linestyle='-', label="Raw Path", color="darkblue")
    plt.scatter(x[0], y[0], color="green", label="Start", zorder=3)
    plt.scatter(x[-1], y[-1], color="red", label="End", zorder=3)
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True); plt.legend(); plt.axis('equal')
    plt.show()

# ---------- Avalanche Experiment ----------

def run_avalanche_experiment(
    x0=np.array([0,0]),
    n=128,
    epsilon=0.25,
    trials=50,
    a_params=(0.6, -0.2, 0.3, 0.5),
    b_range=(-5,5),
    perturb_offsets=(-5,-1,0,1,5),
    show_plots=True,
    plot_path=True,
    hash_algos=("SHA3", "SHAKE256", "BLAKE3")
):
    records = []
    bit_rows = {algo: [] for algo in hash_algos}

    for trial in range(trials):
        path1 = chaotic_walk(x0, n, epsilon, a_params, b_range)
        if trial == 0 and plot_path:
            smooth_plot_path(path1)

        mid = n // 2
        for offset in perturb_offsets:
            path2 = [p.copy() for p in path1]
            i = np.clip(mid + offset, 1, n-1)
            path2[i] += np.array([1,0])

            for algo in hash_algos:
                hash1 = hash_path(path1, algo)
                hash2 = hash_path(path2, algo)
                ent1 = compute_entropy(hash1)
                ent2 = compute_entropy(hash2)
                ham, bit_vec = hamming_distance(hash1, hash2)
                rate = compute_bit_flip_rate(ham, len(bit_vec))

                records.append({
                    'trial': trial,
                    'algorithm': algo,
                    'path_length': n,
                    'position': i,
                    'hamming': ham,
                    'bit_flip_rate': rate,
                    'entropy_diff': ent2 - ent1
                })
                bit_rows[algo].append(bit_vec)

    df = pd.DataFrame(records)

    chi2_stats = {}
    for algo in hash_algos:
        bit_matrix = np.vstack(bit_rows[algo])
        chi2_stat, p_val = compute_chi_square(bit_matrix)
        chi2_stats[algo] = (chi2_stat, p_val)

    if show_plots:
        plt.figure(figsize=(10,4))
        sns.lineplot(
            data=df, x='position', y='hamming', hue='algorithm',
            palette="Set2", errorbar='sd', marker="o", linewidth=2, legend=True
        )
        plt.title("Hamming Distance by Hash Algorithm")
        plt.grid(True); plt.tight_layout(); plt.show()

        plt.figure(figsize=(10,4))
        sns.lineplot(
            data=df, x='position', y='bit_flip_rate', hue='algorithm',
            palette="Set3", errorbar='sd', marker="o", linewidth=2, legend=True
        )
        plt.title("Bit-flip Rate by Hash Algorithm")
        plt.grid(True); plt.tight_layout(); plt.show()

        plt.figure(figsize=(10,4))
        sns.lineplot(
            data=df, x='position', y='entropy_diff', hue='algorithm',
            palette="Pastel1", errorbar='sd', marker="o", linewidth=2, legend=True
        )
        plt.title("Entropy Difference by Hash Algorithm")
        plt.grid(True); plt.tight_layout(); plt.show()


    return df, chi2_stats

# ---------- Entry Point ----------

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    df, chi2_stats = run_avalanche_experiment(
        x0=np.array([0,0]), n=128, epsilon=0.25,
        trials=50, a_params=(0.6, -0.2, 0.3, 0.5), b_range=(-5,5),
        perturb_offsets=(-5,-1,0,1,5), show_plots=True, plot_path=True,
        hash_algos=("SHA3", "SHAKE256", "BLAKE3")
    )

    print("\n--- Chi-square Results by Algorithm ---")
    for algo, (chi2, p_val) in chi2_stats.items():
        print(f"[{algo}] Chi-square: {chi2:.2f}, p-value: {p_val:.4f}")

    print("\n--- Summary Statistics ---")
    print(df.groupby('algorithm')[['hamming','bit_flip_rate','entropy_diff']].describe())
