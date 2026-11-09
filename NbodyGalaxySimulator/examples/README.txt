Starter scripts for the Suggested Experiments
=============================================
Run these commands from GFTGUX-Programs/NbodyGalaxySimulator, beside main.py.
They use the same NumPy dependency as the program; no source edits are required.

EXP-8:  python examples/force_accuracy.py
        python examples/force_accuracy.py --n_bodies 200 --seed 1
Repeat seeds 0 through 19 to reproduce the Help table's cross-seed ranges.
Each output row describes one seed, not the range across seeds.

EXP-9:  python examples/force_timing.py
        python examples/force_timing.py --sizes 100 300 1000 --repeats 20
The default is a short trial. Increase the sizes/repetitions only after checking
runtime. Reuse this script for EXP-15. A full run is deliberately not timed.
Redirect output to a text file if desired; the numeric rows are comma-separated.

EXP-10: python examples/momentum_comparison.py
        python examples/momentum_comparison.py --theta 1.0
Both methods use identical initial conditions and the same time resolution.
The normalizing speed is evaluated once at the start. The output reports the
change in momentum, rather than dividing by the nearly zero initial momentum.

The API uses meters, kilograms and seconds internally. The force-accuracy
script prints softening in parsecs and force errors as both fractions and
percentages. The timing script reports seconds. These scripts expose the
measurements; interpreting them remains the exercise.
