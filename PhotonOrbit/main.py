"""
main.py — PhotonOrbit

Top-level launcher for the Photon Orbit visualizer.
"""

from driver_photon import driver_photon_orbit

if __name__ == "__main__":
    driver_photon_orbit(
        GM_over_c2= 1.0,
        r0 = 20.0,
        b = 5.0,
        lambda_max = 200.0,
        d_lambda = 0.01
    )
