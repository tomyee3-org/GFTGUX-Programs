"""
physics_photon.py — PhotonOrbit

Contains the GR equations of motion and RK4 integrator for photon trajectories
in the Schwarzschild spacetime (equatorial plane).
"""

import math


def dr_dlambda(r, L, E, GM_over_c2):
    """
    Radial equation for null geodesics:

        (dr/dlambda)^2 = E^2 - (L^2 / r^2) * (1 - 2GM/r)

    We choose inward motion (negative root) by default.
    """
    if r <= 0.0:
        return 0.0

    term = E*E - (L*L / (r*r)) * (1.0 - 2.0 * GM_over_c2 / r)
    if term < 0.0:
        return 0.0

    return -math.sqrt(term)


def dphi_dlambda(r, L):
    """
    Angular equation:

        dphi/dlambda = L / r^2
    """
    if r <= 0.0:
        return 0.0
    return L / (r*r)


def rk4_step(r, phi, L, E, GM_over_c2, d_lambda):
    """
    One RK4 step for (r, phi).
    """

    # k1
    k1_r   = dr_dlambda(r, L, E, GM_over_c2)
    k1_phi = dphi_dlambda(r, L)

    # k2
    r2     = r   + 0.5 * d_lambda * k1_r
    phi2   = phi + 0.5 * d_lambda * k1_phi
    k2_r   = dr_dlambda(r2, L, E, GM_over_c2)
    k2_phi = dphi_dlambda(r2, L)

    # k3
    r3     = r   + 0.5 * d_lambda * k2_r
    phi3   = phi + 0.5 * d_lambda * k2_phi
    k3_r   = dr_dlambda(r3, L, E, GM_over_c2)
    k3_phi = dphi_dlambda(r3, L)

    # k4
    r4     = r   + d_lambda * k3_r
    phi4   = phi + d_lambda * k3_phi
    k4_r   = dr_dlambda(r4, L, E, GM_over_c2)
    k4_phi = dphi_dlambda(r4, L)

    # Combine
    r_next   = r   + (d_lambda / 6.0) * (k1_r   + 2*k2_r   + 2*k3_r   + k4_r)
    phi_next = phi + (d_lambda / 6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)

    return r_next, phi_next


def integrate_photon_orbit(GM_over_c2, r0, b, lambda_max, d_lambda):
    """
    Integrate a photon trajectory starting at radius r0 with impact parameter b.

    Returns:
        x_values, y_values, r_s, r_photon
    """

    E = 1.0
    L = b

    r   = r0
    phi = 0.0

    x_values = []
    y_values = []

    r_s      = 2.0 * GM_over_c2
    r_photon = 3.0 * GM_over_c2

    lambda_value = 0.0

    while lambda_value <= lambda_max:
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        x_values.append(x)
        y_values.append(y)

        if r <= r_s:
            break
        if r >= 2.0 * r0:
            break

        r, phi = rk4_step(r, phi, L, E, GM_over_c2, d_lambda)
        lambda_value += d_lambda

    return x_values, y_values, r_s, r_photon
