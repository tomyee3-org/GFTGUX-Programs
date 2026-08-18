"""
driver_photon.py — PhotonOrbit

Driver module for setting parameters and calling the physics and plot modules.
"""

from physics_photon import integrate_photon_orbit
from plot_photon import plot_photon_orbit


def driver_photon_orbit(
    GM_over_c2=1.0,
    r0=20.0,
    b=5.0,
    lambda_max=200.0,
    d_lambda=0.01,
):
    """Run one photon-orbit calculation and display the result."""
    x_vals, y_vals, info = integrate_photon_orbit(
        GM_over_c2=GM_over_c2,
        r0=r0,
        b=b,
        lambda_max=lambda_max,
        d_lambda=d_lambda,
    )
    plot_photon_orbit(x_vals, y_vals, b, info)
    return info
