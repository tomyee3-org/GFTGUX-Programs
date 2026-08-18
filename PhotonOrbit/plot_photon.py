"""
plot_photon.py — PhotonOrbit

Matplotlib visualization of photon trajectories around a Schwarzschild black hole.
"""

import matplotlib.pyplot as plt


def plot_photon_orbit(x_values, y_values, r_s, r_photon, b):
    """
    Plot the photon trajectory and mark the event horizon + photon sphere.
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', 'box')

    ax.plot(x_values, y_values, color='blue', label='Photon trajectory')

    # Event horizon
    horizon = plt.Circle((0.0, 0.0), r_s, color='black', alpha=0.3, label='Event horizon')
    ax.add_artist(horizon)

    # Photon sphere
    sphere = plt.Circle((0.0, 0.0), r_photon, color='red', alpha=0.3,
                        fill=False, linestyle='--', label='Photon sphere')
    ax.add_artist(sphere)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Photon orbit around Schwarzschild black hole (b = {b:.2f})")

    ax.legend(loc='upper right')
    ax.grid(True)

    plt.show()
