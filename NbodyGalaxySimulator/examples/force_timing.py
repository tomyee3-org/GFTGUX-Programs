"""Time force evaluations, including tree construction, at fixed particle configurations."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physics_nbg as phys


from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 300])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not np.isfinite(args.theta) or not 0 <= args.theta <= 2:
        parser.error('--theta must be finite and in [0, 2]')
    if not all(3 <= n <= phys.MAX_BODIES for n in args.sizes):
        parser.error(f'--sizes must be in [3, {phys.MAX_BODIES}]')
    print(f'seed={args.seed}, theta={args.theta}, repeats={args.repeats}; '
          'median seconds per call; tree includes construction')
    print('N,direct_seconds,tree_seconds,tree_over_direct')
    for n in args.sizes:
        eps = phys.athanassoula_softening(n, phys.PC)
        initial = phys.plummer_sphere(n, 1000.0, 1.0, softening=eps, seed=args.seed)
        pos, masses = initial['positions'], initial['masses']
        calls = {
            'direct': lambda: phys.compute_accelerations_direct(pos, masses, softening=eps),
            'tree': lambda: phys.compute_accelerations_tree(pos, masses,
                                                           theta=args.theta, softening=eps),
        }
        samples = {name: [] for name in calls}
        for call in calls.values():
            call()  # Untimed warm-up, on the same configuration.
        for repeat in range(args.repeats):
            order = ('direct', 'tree') if repeat % 2 == 0 else ('tree', 'direct')
            for name in order:
                start = perf_counter()
                calls[name]()
                samples[name].append(perf_counter()-start)
        direct, tree = (float(np.median(samples[name])) for name in ('direct', 'tree'))
        print(f'{n},{direct:.12g},{tree:.12g},{tree/direct:.12g}', flush=True)


if __name__ == '__main__':
    main()
