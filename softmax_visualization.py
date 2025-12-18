import numpy as np
import matplotlib.pyplot as plt


def softmax(q_values: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Compute softmax probabilities for an array of values (e.g., Q-values / logits)
    with an inverse temperature parameter beta.

    P(i) = exp(beta * q_i) / sum_j exp(beta * q_j)
    """
    q_values = np.asarray(q_values, dtype=float)
    # Numerical stability: subtract max to avoid overflow
    shifted = q_values - np.max(q_values)
    exp_vals = np.exp(beta * shifted)
    return exp_vals / np.sum(exp_vals)


def visualize_softmax_temperature():
    """
    Show how the inverse temperature beta controls how "peaked" the softmax is.
    Larger beta -> more greedy (one action dominates).
    Smaller beta -> more uniform distribution.
    """
    # Some example logits / values (e.g., Q-values or utilities)
    q_values = np.array([-2, -1, 0, 1, 2], dtype=float)
    actions = np.arange(len(q_values))

    betas = [0.2, 0.5, 1.0, 2.0, 5.0]

    plt.figure(figsize=(8, 5))
    for beta in betas:
        probs = softmax(q_values, beta=beta)
        plt.plot(
            actions,
            probs,
            marker="o",
            label=rf"$\beta={beta}$",
        )

    plt.xticks(actions, [f"a{i}" for i in actions])
    plt.xlabel("Action / Option", fontweight="bold")
    plt.ylabel("Softmax probability", fontweight="bold")
    # Use lambda (λ) as the inverse temperature symbol in the title
    plt.title(
        "Softmax probabilities for different inverse temperatures $\\lambda$",
        fontweight="bold",
    )
    # Explicitly indicate which logits are used at the top of the chart
    logits_str = ", ".join(f"{v:g}" for v in q_values)
    plt.text(
        0.5,
        1.04,
        rf"Logits used: $[{logits_str}]$",
        ha="center",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=11,
        fontweight="bold",
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def visualize_difference_preservation():
    """
    Show that vectors with the same dimension and the same value differences
    (i.e., differing only by an additive constant shift) give the same softmax
    output, thanks to the shift-invariance of softmax.

    If q' = q + c·1 for any constant c, then softmax(q') = softmax(q).
    """
    # Base vector (logits) defining a pattern of value differences
    base_q = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    actions = np.arange(len(base_q))

    # Create several shifted versions: q' = base_q + c
    shifts = [-5.0, -2.0, 0.0, 3.0, 7.0]
    shifted_vectors = [base_q + c for c in shifts]

    plt.figure(figsize=(8, 5))

    for idx, (c, q_vec) in enumerate(zip(shifts, shifted_vectors)):
        probs = softmax(q_vec, beta=1.0)
        # Tiny horizontal offset per curve so you can SEE that
        # there are multiple curves that would otherwise lie exactly
        # on top of each other (same softmax output).
        x = actions + 0.04 * (idx - len(shifts) / 2)

        # Use dashed lines to emphasise that all curves share the same shape
        plt.plot(
            x,
            probs,
            marker="o",
            linestyle="--",
            alpha=0.8,
            label=rf"$\mathbf{{q}} = \mathbf{{q}}_0 + {c}$",
        )

    plt.xticks(actions, [f"a{i}" for i in actions])
    plt.xlabel("Index / Dimension", fontweight="bold")
    plt.ylabel("Softmax probability", fontweight="bold")
    # Short, clean title at the top
    plt.title("Shift-invariance of softmax", y=1.02, fontweight="bold")
    # Explanatory text moved to the bottom to avoid overlapping with the title
    plt.text(
        0.5,
        -0.18,
        (
            "All vectors have the same value differences (q' = q + c·1), "
            "so their softmax outputs are identical.\n"
            "Curves are slightly shifted along the x-axis only for visibility."
        ),
        ha="center",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=10,
        fontweight="bold",
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    visualize_softmax_temperature()
    visualize_difference_preservation()
    plt.show()


if __name__ == "__main__":
    main()


