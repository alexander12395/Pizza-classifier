"""
Two-spirals benchmark for your `network` class.

The classic Lang & Witbrock (1988) setup: two interleaved spirals,
97 points each, three full revolutions. Coordinates are normalised
to roughly [-1, 1] because your layers are tanh.

Targets are -1 / +1 and the threshold is 0, matching your tanh output.

Change the import below to your filename.
"""

import numpy as np
from functions import *


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------

def make_spirals(n_per_arm=97, offset=0.0):
    """
    offset=0.0  -> the standard benchmark points (use for training)
    offset=0.5  -> points sitting halfway between them (use for testing)

    The offset set is the interesting one: it asks whether the network
    learned the spiral or just memorised 194 coordinates.
    """
    i = np.arange(n_per_arm) + offset

    angle = i * np.pi / 16.0
    radius = 6.5 * (104.0 - i) / 104.0

    arm = np.stack([radius * np.sin(angle),
                    radius * np.cos(angle)], axis=1) / 6.5

    X = np.vstack([arm, -arm])                     # second arm is the negation
    Y = np.concatenate([np.ones(n_per_arm),
                        -np.ones(n_per_arm)])
    return X, Y


# ---------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------

def accuracy(net, X, Y):
    out = np.array([net.forward_pass(x)[0] for x in X])
    return np.mean(np.sign(out) == Y)


def train(X, Y, hidden, lr=0.02, epochs=8000, seed=0,
          Xte=None, Yte=None, log_every=None):
    """
    hidden : list of hidden layer sizes, e.g. [50] or [20, 20]
    """
    np.random.seed(seed)

    spec = list(hidden) + [1]                      # output layer is 1 neuron
    net = network(len(spec), spec, 2)

    order = np.arange(len(X))

    for epoch in range(epochs):
        np.random.shuffle(order)                   # essential: fixed order cycles
        total = 0.0

        for i in order:
            net.forward_pass(X[i])
            total += net.backpropagation([Y[i]], lr, X[i])

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            tr = accuracy(net, X, Y)
            msg = f"    epoch {epoch:6d}   MSE {total/len(X):.5f}   train {tr:6.1%}"
            if Xte is not None:
                msg += f"   test {accuracy(net, Xte, Yte):6.1%}"
            print(msg)

    return net


# ---------------------------------------------------------------
# ASCII decision boundary
# ---------------------------------------------------------------

def plot_boundary(net, X, Y, w=61, h=31):
    lo, hi = -1.15, 1.15
    xs = np.linspace(lo, hi, w)
    ys = np.linspace(hi, lo, h)

    shades = " .:-=+*#%@"
    grid = []
    for b in ys:
        row = ""
        for a in xs:
            v = net.forward_pass(np.array([a, b]))[0]
            row += shades[min(9, max(0, int((v + 1) / 2 * 9.999)))]
        grid.append(row)

    for (px, py), t in zip(X, Y):
        c = int(round((px - lo) / (hi - lo) * (w - 1)))
        r = int(round((py - hi) / (lo - hi) * (h - 1)))
        if 0 <= r < h and 0 <= c < w:
            row = list(grid[r])
            row[c] = "+" if t > 0 else "o"
            grid[r] = "".join(row)

    print("   dark = -1,  bright = +1,   o / + = training points")
    for line in grid:
        print("   " + line)


# ---------------------------------------------------------------

if __name__ == "__main__":

    Xtr, Ytr = make_spirals()                      # 194 benchmark points
    Xte, Yte = make_spirals(offset=0.5)            # 194 interleaved points

    print(f"train {Xtr.shape}   test {Xte.shape}")
    print(f"range [{Xtr.min():.2f}, {Xtr.max():.2f}]")

    print()
    print("=" * 66)
    print("1. Single run  (two hidden layers of 20, lr 0.02)")
    print("=" * 66)
    net = train(Xtr, Ytr, [20, 20], lr=0.02, epochs=8000, seed=0,
                Xte=Xte, Yte=Yte, log_every=500)

    print()
    print("=" * 66)
    print("2. Depth vs width")
    print("=" * 66)
    print(f"  {'architecture':<22} {'params':>8} {'train':>8} {'test':>8}")
    print("  " + "-" * 48)

    for hidden in [[10], [30], [50], [10, 10], [20, 20], [30, 30]]:
        n = train(Xtr, Ytr, hidden, lr=0.02, epochs=8000, seed=0)

        p = sum(l.weight_array.size + l.bias_array.size for l in n.layer_array)
        label = "2-" + "-".join(str(x) for x in hidden) + "-1"

        print(f"  {label:<22} {p:>8,} "
              f"{accuracy(n, Xtr, Ytr):>7.1%} {accuracy(n, Xte, Yte):>8.1%}")

    print()
    print("=" * 66)
    print("3. Boundary learned by the run from section 1")
    print("=" * 66)
    plot_boundary(net, Xtr, Ytr)