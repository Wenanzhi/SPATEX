# Attribution and source provenance

This candidate was assembled from the local `TAC-TSE-prloss` project, which
developed from the M2M-TSE / M2M-SPKTSE code family. The GPL v2 text is copied
unchanged from `M2M-TSE-main/LICENSE`. That source's README identifies its
training/evaluation framework as based on Waveformer and its backbone as based
on DeFTAN-II. These origins must remain attributed when publishing modifications.

- DeFTAN-II: <https://github.com/donghoney0416/DeFTAN-II>
- Waveformer: <https://github.com/vb000/Waveformer>
- Pyroomacoustics (installed dependency): <https://github.com/LCAV/pyroomacoustics>

The original M2M-TSE Scaper notice is retained verbatim in
`licenses/UPSTREAM_THIRD_PARTY_NOTICES.txt` for source provenance. This candidate's
LibriSpeech dataset does not import the vendored `scaper_edited` or
`pyloudnorm_edited` packages, so those packages are omitted.

Installed dependencies retain their own licenses. No third-party speech corpus,
model weights, downloaded baseline repository or manuscript PDF is bundled.
Author release review should confirm the source lineage and copyright attribution
before publication; this file records the evidence available in the workspace
and does not declare a new permissive license over inherited code.
