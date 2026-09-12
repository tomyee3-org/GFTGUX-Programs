"""
main.py
=======
Command-line entry point for BlackHoleShadow.

Eight calculations share one program, chosen with --mode, and they follow
the teaching beats in the Help file:

    rays       two PhotonOrbit-class geodesics
    pixels     two impact parameters on a coarse camera grid
    capture    geometric capture map (the shadow)
    ring       high-winding-ray overlay on that critical curve
    radii      spacetime radii versus the image-plane critical curve
    weak       exact deflection versus the weak-field 4M/b formula
    compare    a galaxy Einstein radius versus this hole's b_crit
    backlight  schematic periapsis overlay (false colour)

Length flags (--r_cam, --b_in, --b_out, --fov, --r_hot) are in units of M.

Examples
--------
  python main.py --mode rays
  python main.py --mode capture
  python main.py --mode capture --M 5
  python main.py --mode weak --outdir ./runs
"""

import argparse
import os

import driver_bhs
import physics_bhs


def parse_args():
    p = argparse.ArgumentParser(
        prog="BlackHoleShadow",
        description=(
            "Image-plane teaching program: the Schwarzschild capture map, "
            "the critical curve, and why neither is a thin-lens Einstein ring."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version", action="version",
        version=(f"BlackHoleShadow {physics_bhs.MODEL_VERSION} "
                 f"(build {physics_bhs.BUILD_ID})"),
    )
    p.add_argument("--mode", choices=driver_bhs.MODES, default="capture",
                   help="which drawing to make")

    g = p.add_argument_group("Spacetime")
    g.add_argument("--M", type=float, default=1.0, metavar="LEN",
                   help="gravitational length GM/c^2; teaching default is 1")
    g.add_argument("--r_cam", type=float, default=None, metavar="M",
                   help="camera / launch radius in units of M "
                        f"(default {physics_bhs.R_CAM_DEFAULT:g}; must exceed 3)")

    g = p.add_argument_group("Two demonstration rays")
    g.add_argument("--b_in", type=float, default=5.0, metavar="M",
                   help="first impact parameter in units of M")
    g.add_argument("--b_out", type=float, default=6.0, metavar="M",
                   help="second impact parameter in units of M")
    g.add_argument("--lambda_max", type=float, default=200.0, metavar="LAMBDA",
                   help="maximum affine parameter for ray integrations")
    g.add_argument("--d_lambda", type=float, default=0.02, metavar="LAMBDA",
                   help="RK4 step for ray integrations")

    g = p.add_argument_group("Camera grid")
    g.add_argument("--n_pix", type=int, default=161, metavar="N",
                   help="pixels along one side; even values become the next odd. "
                        f"pixels mode caps this at {physics_bhs.PIXELS_N_PIX_MAX}")
    g.add_argument("--fov", type=float, default=16.0, metavar="M",
                   help="field of view on a side, in units of M")

    g = p.add_argument_group("Backlight and compare")
    g.add_argument("--r_hot", type=float, default=None, metavar="M",
                   help="turning-point radius used by the schematic overlay, "
                        f"in units of M (default {physics_bhs.HOT_RING_R_DEFAULT:g})")
    g.add_argument("--inclination", type=float, default=0.0, metavar="DEG",
                   help="display inclination for backlight shading, in [0, 90]")
    g.add_argument("--logM", type=float, default=12.0, metavar="LOG10",
                   help="log10(M/M_sun) of the comparison galaxy lens")

    g = p.add_argument_group("Grid and output")
    g.add_argument("--dpi", type=int, default=140, metavar="N",
                   help="PNG resolution")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG (and provenance sidecar)")
    g.add_argument("--interactive", action="store_true",
                   help="open the figure on screen (no-op under MPLBACKEND=Agg)")
    p.add_argument("--sync-help", action="store_true",
                   help="write MODEL_VERSION and BUILD_ID into the Help file")
    return p.parse_args()


def _validate_args(args):
    physics_bhs.validate_user_value("M", args.M, positive=True)
    physics_bhs.validate_user_value("b_in", args.b_in, min_value=0.0)
    physics_bhs.validate_user_value("b_out", args.b_out, min_value=0.0)
    physics_bhs.validate_user_value("lambda_max", args.lambda_max, positive=True)
    physics_bhs.validate_user_value("d_lambda", args.d_lambda, positive=True)
    physics_bhs.validate_user_value("fov", args.fov, positive=True)
    physics_bhs.validate_user_value("dpi", args.dpi, positive=True)
    physics_bhs.validate_user_value(
        "inclination", args.inclination, min_value=0.0, max_value=90.0,
    )
    physics_bhs.validate_user_value(
        "logM", args.logM,
        min_value=physics_bhs.LOGM_MIN, max_value=physics_bhs.LOGM_MAX,
    )
    if args.r_cam is not None:
        physics_bhs.validate_user_value("r_cam", args.r_cam, positive=True)
        if args.r_cam <= 3.0:
            raise ValueError("--r_cam must exceed 3 (units of M)")
    if args.r_hot is not None:
        physics_bhs.validate_user_value("r_hot", args.r_hot, positive=True)
        if args.r_hot <= 3.0:
            raise ValueError("--r_hot must exceed 3 (units of M)")
    if int(args.n_pix) != args.n_pix or args.n_pix < 9:
        raise ValueError("n_pix must be an integer >= 9")


def main():
    args = parse_args()
    if args.sync_help:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(os.path.dirname(here),
                         "BlackHoleShadow-Documentation", "BlackHoleShadow.html"),
            os.path.join(os.path.dirname(here), "BlackHoleShadow.html"),
            os.path.join(here, "BlackHoleShadow.html"),
        ]
        html = next((p for p in candidates if os.path.isfile(p)), candidates[0])
        try:
            ver, bid = physics_bhs.patch_help_version(html)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"BlackHoleShadow: {exc}") from exc
        print(f"Wrote Version {ver}  Build {bid} -> {html}")
        return
    try:
        _validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"BlackHoleShadow: {exc}") from exc

    show = args.interactive or args.outdir is None
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        show = False

    common = dict(outdir=args.outdir, dpi=args.dpi, show=show,
                  interactive=args.interactive, M=args.M)
    try:
        if args.mode == "rays":
            driver_bhs.run_rays(
                r_cam=args.r_cam, b_in=args.b_in, b_out=args.b_out,
                lambda_max=args.lambda_max, d_lambda=args.d_lambda, **common)
        elif args.mode == "pixels":
            n_use = min(int(args.n_pix), physics_bhs.PIXELS_N_PIX_MAX)
            driver_bhs.run_pixels(
                b_in=args.b_in, b_out=args.b_out,
                n_pix=n_use, n_pix_requested=args.n_pix,
                fov_M=args.fov, **common)
        elif args.mode == "capture":
            driver_bhs.run_capture(n_pix=args.n_pix, fov_M=args.fov, **common)
        elif args.mode == "ring":
            driver_bhs.run_ring(n_pix=args.n_pix, fov_M=args.fov, **common)
        elif args.mode == "radii":
            driver_bhs.run_radii(n_pix=args.n_pix, fov_M=args.fov, **common)
        elif args.mode == "weak":
            driver_bhs.run_weak(**common)
        elif args.mode == "compare":
            driver_bhs.run_compare(logM=args.logM, **common)
        elif args.mode == "backlight":
            driver_bhs.run_backlight(
                n_pix=args.n_pix, fov_M=args.fov, r_hot=args.r_hot,
                inclination=args.inclination, **common)
    except (ValueError, RuntimeError, OSError, OverflowError,
            ZeroDivisionError) as exc:
        raise SystemExit(f"BlackHoleShadow: {exc}") from exc


if __name__ == "__main__":
    main()
