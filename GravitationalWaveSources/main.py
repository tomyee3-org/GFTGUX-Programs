"""
main.py
=======
Command-line entry point for GravitationalWaveSources.

Examples
--------
  # Default 1.4 + 1.4 M_sun inspiral from 20 Hz (about 158 s)
  python main.py

  # Compare a heavier binary
  python main.py --m1 36 --m2 29 --d 440

  # Add the optional illustrative Schwarzschild ringdown for a BBH example
  python main.py --m1 36 --m2 29 --d 440 --ringdown --t_before 0.2 --t_after 0.05

  # Also save a timestamped PNG (in addition to the on-screen display)
  python main.py --outdir ./runs
"""

import argparse
import driver_gw
import physics_gw


def parse_args():
    p = argparse.ArgumentParser(
        prog="GravitationalWaveSources",
        description=(
            "Simulate a leading-order quasi-circular compact-binary inspiral to "
            "the Schwarzschild ISCO cutoff. An optional Schwarzschild QNM "
            "ringdown is available as an explicitly illustrative extension."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=(f"GravitationalWaveSources {physics_gw.MODEL_VERSION} "
                 f"(build {physics_gw.BUILD_ID})"),
    )

    g = p.add_argument_group("Binary parameters")
    g.add_argument("--m1", type=float, default=1.4, metavar="M_SUN",
                   help="Primary mass [solar masses] (default 1.4)")
    g.add_argument("--m2", type=float, default=1.4, metavar="M_SUN",
                   help="Secondary mass [solar masses] (default 1.4)")
    g.add_argument("--d", type=float, default=400.0, metavar="MPC",
                   help="Luminosity distance [Mpc] (default 400)")

    g = p.add_argument_group("Integration parameters")
    g.add_argument("--dt", type=float, default=2e-4, metavar="SEC",
                   help="RK4 timestep [s] (default 2e-4; must resolve the ISCO waveform)")
    g.add_argument("--f_start", type=float, default=20.0, metavar="HZ",
                   help="Starting GW frequency [Hz] (default 20)")
    g.add_argument("--ringdown", action="store_true",
                   help="append an illustrative Schwarzschild QNM after the ISCO cutoff")
    g.add_argument("--n_tau", type=int, default=6, metavar="N",
                   help="ringdown duration in QNM decay times (default 6)")
    g.add_argument("--rd_pts", type=int, default=4000, metavar="N",
                   help="ringdown sample points, 2..500000 (default 4000)")

    g = p.add_argument_group("Zoom parameters (time relative to ISCO cutoff)")
    g.add_argument("--t_before", type=float, default=None, metavar="SEC",
                   help="show only this many seconds before ISCO (default: full inspiral)")
    g.add_argument("--t_after", type=float, default=None, metavar="SEC",
                   help="show only this many seconds after ISCO (default: all available data)")

    g = p.add_argument_group("Plot style")
    g.add_argument("--lw", type=float, default=0.4, metavar="PT",
                   help="waveform line width [points] (default 0.4)")
    g.add_argument("--dpi", type=int, default=150, metavar="N",
                   help="PNG resolution [dpi] (default 150)")

    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG in PATH, in addition to "
                        "displaying it on screen (default: display only)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        driver_gw.run(
            m1_msun=args.m1,
            m2_msun=args.m2,
            d_mpc=args.d,
            dt=args.dt,
            f_start=args.f_start,
            outdir=args.outdir,
            t_before=args.t_before,
            t_after=args.t_after,
            lw=args.lw,
            dpi=args.dpi,
            include_ringdown=args.ringdown,
            n_ringdown_tau=args.n_tau,
            ringdown_pts=args.rd_pts,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"GravitationalWaveSources: {exc}") from exc


if __name__ == "__main__":
    main()
