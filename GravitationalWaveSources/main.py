"""
main.py
=======
Command-line entry point for GravitationalWaveSources.

Output behaviour
----------------
No --outdir supplied  →  interactive screen display (plt.show())
--outdir PATH         →  PNG saved to PATH/gw_inspiral_YYYYMMDD_HHmm.png

Zoom examples
-------------
  # Full 150-second BNS inspiral on screen
  python main.py

  # Zoom to last 2 s + 50 ms of ringdown, screen
  python main.py --t_before 2.0 --t_after 0.05

  # Extreme zoom — see individual ringdown cycles
  python main.py --t_before 0.05 --t_after 0.02 --lw 0.5

  # BBH, zoom, save to subfolder
  python main.py --m1 36 --m2 29 --d 440 --t_before 0.5 --t_after 0.05 --outdir ./output

  # BNS, save full plot with timestamp
  python main.py --outdir ./runs
"""

import argparse
import driver_gw


def parse_args():
    p = argparse.ArgumentParser(
        prog="GravitationalWaveSources",
        description=(
            "Simulate a quasi-circular binary inspiral with QNM ringdown.\n"
            "No --outdir → screen display.  --outdir PATH → timestamped PNG."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Binary ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Binary parameters")
    g.add_argument("--m1", type=float, default=1.4,  metavar="M_SUN",
                   help="Primary mass [solar masses]  (default 1.4)")
    g.add_argument("--m2", type=float, default=1.4,  metavar="M_SUN",
                   help="Secondary mass [solar masses]  (default 1.4)")
    g.add_argument("--d",  type=float, default=400.0, metavar="MPC",
                   help="Luminosity distance [Mpc]  (default 400)")

    # ── Integration ───────────────────────────────────────────────────────────
    g = p.add_argument_group("Integration parameters")
    g.add_argument("--dt",      type=float, default=1e-4, metavar="SEC",
                   help="RK4 time step [s]  (default 1e-4; smaller = smoother)")
    g.add_argument("--f_start", type=float, default=10.0, metavar="HZ",
                   help="Starting GW frequency [Hz]  (default 10)")
    g.add_argument("--n_tau",   type=int,   default=6,    metavar="N",
                   help="Ringdown duration in units of tau_QNM  (default 6)")
    g.add_argument("--rd_pts",  type=int,   default=4000, metavar="N",
                   help="Ringdown sample points  (default 4000)")

    # ── View / zoom ───────────────────────────────────────────────────────────
    g = p.add_argument_group("Zoom parameters  (time relative to merger)")
    g.add_argument("--t_before", type=float, default=None, metavar="SEC",
                   help="Show only this many seconds BEFORE merger "
                        "(default: full inspiral)")
    g.add_argument("--t_after",  type=float, default=None, metavar="SEC",
                   help="Show only this many seconds AFTER  merger "
                        "(default: full ringdown)")

    # ── Plot style ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Plot style")
    g.add_argument("--lw",  type=float, default=0.4,  metavar="PT",
                   help="Waveform line width [points]  (default 0.4)")
    g.add_argument("--dpi", type=int,   default=150,  metavar="N",
                   help="PNG resolution [dpi]  (default 150)")

    # ── Output ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="Directory for PNG output.  "
                        "Omit to display on screen instead of saving.\n"
                        "Filename: gw_inspiral_YYYYMMDD_HHmm.png")

    return p.parse_args()


def main():
    args = parse_args()

    driver_gw.run(
        m1_msun        = args.m1,
        m2_msun        = args.m2,
        d_mpc          = args.d,
        dt             = args.dt,
        f_start        = args.f_start,
        outdir         = args.outdir,
        t_before       = args.t_before,
        t_after        = args.t_after,
        lw             = args.lw,
        dpi            = args.dpi,
        n_ringdown_tau = args.n_tau,
        ringdown_pts   = args.rd_pts,
    )


if __name__ == "__main__":
    main()
