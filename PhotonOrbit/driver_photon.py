"""
driver_photon.py — PhotonOrbit

Driver module for setting parameters and calling the physics + plot modules.
"""

from physics_photon import integrate_photon_orbit
from plot_photon import plot_photon_orbit

def driver_photon_orbit(
    GM_over_c2 = 1.0,
    r0 = 20.0,
    b = 5.0,
    lambda_max = 200.0,
    d_lambda = 0.01
):
    """
    Driver routine for the photon orbit visualizer.
    """

    # Run physics integration
    x_vals, y_vals, r_s, r_photon = integrate_photon_orbit(
        GM_over_c2=GM_over_c2,
        r0=r0,
        b=b,
        lambda_max=lambda_max,
        d_lambda=d_lambda
    )

    # Plot results
    plot_photon_orbit(x_vals, y_vals, r_s, r_photon, b)
