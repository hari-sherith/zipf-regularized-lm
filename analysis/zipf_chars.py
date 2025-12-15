from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def count_chars(text: str, drop_whitespace: bool = False) -> Counter:
    if drop_whitespace:
        text = "".join(ch for ch in text if not ch.isspace())
    return Counter(text)

def zipf_alpha_estimate(freqs, r_min=5, r_max=50):
    freqs = np.array(freqs, dtype=np.float64)
    ranks = np.arange(1, len(freqs) + 1)

    mask = (ranks >= r_min) & (ranks <= r_max)
    log_r = np.log(ranks[mask])
    log_f = np.log(freqs[mask])

    slope, _ = np.polyfit(log_r, log_f, 1)
    return -slope

def plot_zipf_with_fit(freqs, title: str, skip_top_n: int = 2):
    """
    Plot rank vs frequency on log-log axes and overlay a fitted power-law line
    using log-log linear regression over ranks [skip_top_n:].
    """
    freqs = np.array(freqs, dtype=np.float64)
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)

    # Fit on mid-region (skip top ranks to reduce whitespace dominance)
    fit_freqs = freqs[skip_top_n:]
    fit_ranks = ranks[skip_top_n:]

    log_r = np.log(fit_ranks)
    log_f = np.log(fit_freqs)

    slope, intercept = np.polyfit(log_r, log_f, 1)
    alpha = -slope

    # Predicted line in original scale: f = exp(intercept) * r^(slope)
    fitted = np.exp(intercept) * (ranks ** slope)

    plt.figure()
    plt.loglog(ranks, freqs, marker="o", linestyle="none", label="Observed")
    plt.loglog(ranks, fitted, linestyle="-", label=f"Fit (alpha≈{alpha:.2f})")

    plt.xlabel("Rank (most frequent = 1)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    return float(alpha)

def main():
    # Path to dataset
    path = "data/tiny_shakespeare.txt"  # change if needed

    text = load_text(path)
    counts = count_chars(text, drop_whitespace=False)

    freqs = sorted(counts.values(), reverse=True)

    plot_zipf_with_fit(freqs, "Zipf plot (characters): rank vs frequency (log–log)")

    alpha = plot_zipf_with_fit(freqs, "Zipf plot (characters): rank vs frequency (log–log)", skip_top_n=2)
    print(f"Estimated Zipf exponent alpha ≈ {alpha:.2f}\n")

    print("Top 20 characters:")
    for ch, c in counts.most_common(20):
        shown = ch
        if ch == "\n":
            shown = "\\n"
        elif ch == "\t":
            shown = "\\t"
        elif ch == " ":
            shown = "<space>"
        print(f"{shown!r}: {c}")

if __name__ == "__main__":
    main()
