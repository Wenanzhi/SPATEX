# Archived results

These files summarize existing experiment exports matched to the supplied PACT
manuscript. They are not measurements from a fresh training run or a fresh
full-testset evaluation of this release candidate.

- `table1.csv`: ten PACT/independent/objective rows, with SRP-PHAT directions.
- `table2_external.csv`: eight single-output reference rows (four per condition).
- `table3_pact.csv`, `table3_external.csv`: fixed-array comparison.
- `figure2_signed.csv`: signed offsets up to ±15 degrees for the six systems.
- `per_scene/`: 500-scene signal and pairwise metrics for each PACT/control row.
- `SOURCES.json`: exact source-relative export paths and SHA256 hashes.

`checkpoint` names identify historical artifacts; those weights are not bundled.
Analytic DSB is parameter-free: the historical `0.pt` field is a legacy export
placeholder and is not a neural checkpoint requirement. Its public adapter is
not included in this candidate. NB-PESQ and STOI are reference-channel scores.
For Figure 2, average signed offsets of equal magnitude for each model.

The input paper has an unresolved three-term-objective versus archived Act-loss
discrepancy, documented in `docs/REPRODUCIBILITY.md`. These files faithfully retain
the archived measurements rather than suggesting the no-Act recipes produced them.

One source/PDF discrepancy remains: Table 3 DeFTAN-II ILD is printed as 0.082 dB,
but the identified 500-scene CSV gives 0.0814733225 dB (0.081 to three decimals).
The candidate retains the full archived mean; an alternative export matching
0.082 was not identified. The other 171 checked numeric table cells round to
the supplied PDF values.
