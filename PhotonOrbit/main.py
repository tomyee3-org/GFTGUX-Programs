"""
main.py — PhotonOrbit

Command-line entry point for PhotonOrbit.

Integrates a photon (null-geodesic) trajectory in the equatorial plane of
Schwarzschild spacetime and plots it together with the event horizon and
photon sphere.

Examples
--------
  # Default trajectory: GM_over_c2=1, r0=20, b=5 (captured)
  python main.py

  # A scattering trajectory that escapes
  python main.py --b 6.0

  # Near-critical whirling close to the photon sphere
  python main.py --b 5.205

  # Also save a timestamped PNG (in addition to the on-screen display)
  python main.py --b 6.0 --outdir ./runs
"""

import argparse

from driver_photon import driver_photon_orbit


def parse_args():
    p = argparse.ArgumentParser(
        prog="PhotonOrbit",
        description=(
            "Integrate a photon (null-geodesic) trajectory in the "
            "equatorial plane of Schwarzschild spacetime with a "
            "fourth-order Runge-Kutta scheme, and plot the resulting path "
            "together with the event horizon and photon sphere."
        ),
    )

    g = p.add_argument_group("Physics parameters")
    g.add_argument("--GM_over_c2", type=float, default=1.0, metavar="LEN",
                   help="gravitational length GM/c^2 [length units]; must "
                        "be greater than zero (default 1.0)")
    g.add_argument("--r0", type=float, default=20.0, metavar="LEN",
                   help="starting Schwarzschild radius; must lie outside "
                        "the event horizon, r0 > 2*GM_over_c2 (default 20.0)")
    g.add_argument("--b", type=float, default=5.0, metavar="LEN",
                   help="nonnegative impact parameter b=L/E; must also "
                        "permit a real initially ingoing radial velocity "
                        "at r0 (default 5.0)")
    g.add_argument("--lambda_max", type=float, default=200.0, metavar="LAMBDA",
                   help="maximum affine parameter to integrate (default 200.0)")
    g.add_argument("--d_lambda", type=float, default=0.01, metavar="LAMBDA",
                   help="RK4 step size in the affine parameter; reduce for "
                        "convergence tests near the photon sphere (default 0.01)")

    g = p.add_argument_group("Plot style")
    g.add_argument("--lw", type=float, default=1.5, metavar="PT",
                   help="trajectory line width [points] (default 1.5)")
    g.add_argument("--dpi", type=int, default=150, metavar="N",
                   help="PNG resolution [dpi] (default 150)")

    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG in PATH, in addition "
                        "to displaying it on screen (default: display only)")

    return p.parse_args()


def main():
    args = parse_args()
    try:
        driver_photon_orbit(
            GM_over_c2=args.GM_over_c2,
            r0=args.r0,
            b=args.b,
            lambda_max=args.lambda_max,
            d_lambda=args.d_lambda,
            outdir=args.outdir,
            dpi=args.dpi,
            lw=args.lw,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"PhotonOrbit: {exc}") from exc


if __name__ == "__main__":
    main()
