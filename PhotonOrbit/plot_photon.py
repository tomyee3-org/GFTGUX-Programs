"""
plot_photon.py — PhotonOrbit

Matplotlib visualization of photon trajectories around a Schwarzschild black hole.
"""

import matplotlib.pyplot as plt


def plot_photon_orbit(x_values, y_values, b, info):
    """Plot a photon trajectory, event horizon, photon sphere, and diagnostics."""
    if not x_values or len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must be nonempty arrays of equal length.")

    r_s = info["r_s"]
    r_photon = info["r_photon"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", "box")
    ax.plot(x_values, y_values, label="Photon trajectory")

    horizon = plt.Circle((0.0, 0.0), r_s, alpha=0.3, label="Event horizon")
    ax.add_artist(horizon)
    sphere = plt.Circle(
        (0.0, 0.0), r_photon, alpha=0.6,
        fill=False, linestyle="--", label="Photon sphere",
    )
    ax.add_artist(sphere)

    ax.set_xlabel("x (same length units as GM/c^2)")
    ax.set_ylabel("y (same length units as GM/c^2)")
    ax.set_title(f"Photon orbit around a Schwarzschild black hole (b = {b:.4g})")
    summary = (
        f"status: {info['status']}\n"
        f"closest r: {info['closest_approach']:.5g}\n"
        f"Δφ: {info['delta_phi']:.5g} rad"
    )
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, va="bottom")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()
