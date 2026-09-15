# Baseline coverage

The runnable code in this candidate covers PACT, the matched independent-temporal-
attention control, objective variants and fixed-four PACT. Shared TAC, geometry,
DOA encoding and MIMO output remain in the independent control.

The manuscript also reports the following locally reproduced/adapted systems:

| Paper location | Systems | Candidate coverage |
|---|---|---|
| Table 2 | Analytic DSB, SteerNet-GEV, D-SE-BSRNN, BG-TSE | Reported comparison values and source references |
| Table 3 | JNF-SSF, DSENet, Waveformer-M2M, DeFTAN-II, M2M-TSE-DOA, SoundCompass | Reported comparison values and source references |

Those external training implementations and their third-party repositories are
not vendored in this PACT candidate. This package therefore does not claim
end-to-end retraining coverage for every external baseline table row. In the
original workspace, adapters live in `TAC-TSE-prloss/src/training/network_external_*`,
with additional JNF-SSF/FaSNet code in sibling projects. Packaging those adapters
would also require their exact upstream revisions, local changes, separate
dependency sets and license notices. The Table 2 systems are single-output
references; their output must not be presented as reconstructed microphone-wise
MIMO target images.

This scope is explicit so that the author can review the PACT release without
mistaking historical comparison scores for bundled baseline implementations.
