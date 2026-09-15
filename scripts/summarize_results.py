"""Summarize the signal/spatial columns of per-scene evaluation CSVs."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='Per-scene CSV files')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    keys = ['reference_scale_invariant_signal_noise_ratio_i', 'reference_signal_noise_ratio_i',
            'pesq_nb', 'stoi', 'delta_IPD_mean', 'delta_ITD_gccphat_mean', 'delta_ILD_mean']
    output = []
    for name in args.inputs:
        with open(name, newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            parser.error('Empty input: ' + name)
        for key in keys:
            values = [float(row[key]) for row in rows if key in row and row[key] != '']
            finite = [value for value in values if math.isfinite(value)]
            output.append({'file': name, 'metric': key, 'n_scenes': len(rows),
                           'n_finite': len(finite), 'n_missing_or_nonfinite': len(rows) - len(finite),
                           'mean': mean(finite) if finite else ''})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == '__main__':
    main()
