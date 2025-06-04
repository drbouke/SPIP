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

def smooth_plot_path(path, title="Chaotic Walk Trajectory", grid_size=1024):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import splprep, splev

    coords = np.array(path)
    x, y = coords[:, 0], coords[:, 1]
    n_points = len(coords)
    steps = np.diff(coords, axis=0)

    unique_positions = len(np.unique(coords, axis=0))
    total_length = np.sum(np.linalg.norm(steps, axis=1))
    mean_step_length = np.mean(np.linalg.norm(steps, axis=1))
    std_step_length = np.std(np.linalg.norm(steps, axis=1))

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    grid = np.zeros((grid_size, grid_size), dtype=bool)
    if width == 0 or height == 0:
        fd = float('nan')
    else:
        scaled_coords = ((coords - [x_min, y_min]) / [width, height] * (grid_size - 1)).astype(int)
        for xg, yg in scaled_coords:
            if 0 <= xg < grid_size and 0 <= yg < grid_size:
                grid[yg, xg] = True

        def box_count(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return np.count_nonzero(S)

        def fractal_dimension(Z):
            p = min(Z.shape)
            n = 2 ** np.floor(np.log2(p)).astype(int)
            sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)

            counts = []
            valid_sizes = []
            for size in sizes:
                c = box_count(Z, size)
                if c > 0:
                    counts.append(c)
                    valid_sizes.append(size)

            if len(valid_sizes) < 2:
                return float('nan')

            coeffs = np.polyfit(np.log(valid_sizes), np.log(counts), 1)
            return -coeffs[0]

        fd = fractal_dimension(grid)

    # Prepare the figure
    plt.figure(figsize=(8, 6))
    try:
        tck, u = splprep([x, y], s=0)
        xs, ys = splev(np.linspace(0, 1, 300), tck)
        plt.plot(xs, ys, label="Path", color="darkblue")
    except Exception:
        plt.plot(x, y, marker='o', linestyle='-', label="Raw Path", color="darkblue")

    plt.scatter(x[0], y[0], color="green", label="Start", zorder=3)
    plt.scatter(x[-1], y[-1], color="red", label="End", zorder=3)

    plt.title(f"{title} (n = {n_points})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    # Prepare metrics string
    metrics_text = (
        f"Unique positions: {unique_positions}\n"
        f"Total length: {total_length:.1f}\n"
        f"Mean step: {mean_step_length:.2f}\n"
        f"Std step: {std_step_length:.2f}\n"
        f"Bounding box: {width:.0f} × {height:.0f}\n"
        f"Fractal dimension: {fd:.3f}" if not np.isnan(fd) else "Fractal dimension: undefined"
    )

    plt.gcf().text(0.1, 0.92, metrics_text, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()
    plt.show()



def analyze_path_geometry(path, grid_size=1024):
    coords = np.array(path)
    steps = np.diff(coords, axis=0)

    unique_positions = len(np.unique(coords, axis=0))
    total_length = np.sum(np.linalg.norm(steps, axis=1))
    mean_step_length = np.mean(np.linalg.norm(steps, axis=1))
    std_step_length = np.std(np.linalg.norm(steps, axis=1))

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    print("\n--- Geometric Analysis of Chaotic Walk ---")
    print(f"Unique positions: {unique_positions}")
    print(f"Total path length: {total_length:.4f}")
    print(f"Mean step length: {mean_step_length:.4f}")
    print(f"Std of step length: {std_step_length:.4f}")
    print(f"Bounding box: width={width:.4f}, height={height:.4f}")

    grid = np.zeros((grid_size, grid_size), dtype=bool)
    if width == 0 or height == 0:
        print("Fractal dimension: undefined (flat path)")
        return

    scaled_coords = ((coords - [x_min, y_min]) / [width, height] * (grid_size - 1)).astype(int)
    for x, y in scaled_coords:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y, x] = True 

    def box_count(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return np.count_nonzero(S)

    def fractal_dimension(Z):
        p = min(Z.shape)
        n = 2 ** np.floor(np.log2(p)).astype(int)
        sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)

        counts = []
        valid_sizes = []
        for size in sizes:
            c = box_count(Z, size)
            if c > 0:
                counts.append(c)
                valid_sizes.append(size)

        if len(valid_sizes) < 2:
            return float('nan')

        coeffs = np.polyfit(np.log(valid_sizes), np.log(counts), 1)
        return -coeffs[0]

    fd = fractal_dimension(grid)
    if np.isnan(fd):
        print("Estimated fractal dimension (box-counting): undefined (insufficient variation)")
    else:
        print(f"Estimated fractal dimension (box-counting): {fd:.4f}")


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
            analyze_path_geometry(path1)

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
        x0=np.array([0,0]), n=127, epsilon=0.25,
        trials=50, a_params=(0.6, -0.2, 0.3, 0.5), b_range=(-5,5),
        perturb_offsets=(-5,-1,0,1,5), show_plots=True, plot_path=True,
        hash_algos=("SHA3", "SHAKE256", "BLAKE3")
    )

    print("\n--- Chi-square Results by Algorithm ---")
    for algo, (chi2, p_val) in chi2_stats.items():
        print(f"[{algo}] Chi-square: {chi2:.4f}, p-value: {p_val:.4f}")

    print("\n--- Summary Statistics ---")
    print(df.groupby('algorithm')[['hamming','bit_flip_rate','entropy_diff']].describe())
