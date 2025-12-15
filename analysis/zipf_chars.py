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

def zipf_alpha_estimate(freqs, skip_top_n: int = 2) -> float:
    """
    Estimate Zipf exponent alpha via log-log linear fit:
      log(freq) ≈ -alpha * log(rank) + b
    """
    freqs = np.array(freqs, dtype=np.float64)
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)

    # skip the first few ranks (often dominated by whitespace/newlines)
    freqs = freqs[skip_top_n:]
    ranks = ranks[skip_top_n:]

    log_r = np.log(ranks)
    log_f = np.log(freqs)

    slope, intercept = np.polyfit(log_r, log_f, 1)
    alpha = -slope
    return float(alpha)

def plot_zipf(freqs, title: str):
    ranks = np.arange(1, len(freqs) + 1)
    plt.figure()
    plt.loglog(ranks, freqs, marker="o", linestyle="none")
    plt.xlabel("Rank (most frequent = 1)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

def main():
    # Path to dataset
    path = "data/tiny_shakespeare.txt"  # change if needed

    text = load_text(path)
    counts = count_chars(text, drop_whitespace=False)

    freqs = sorted(counts.values(), reverse=True)

    plot_zipf(freqs, "Zipf plot (characters): rank vs frequency (log–log)")

    alpha = zipf_alpha_estimate(freqs, skip_top_n=2)
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
