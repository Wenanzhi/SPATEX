# Attribution and source provenance

SPATEX developed from the M2M-TSE / M2M-SPKTSE code family. The GPL v2 text is
retained unchanged from the upstream M2M-TSE source. Its README identifies its
training/evaluation framework as based on Waveformer and its backbone as based
on DeFTAN-II. These origins must remain attributed when publishing modifications.

- DeFTAN-II: <https://github.com/donghoney0416/DeFTAN-II>
- Waveformer: <https://github.com/vb000/Waveformer>
- Pyroomacoustics (installed dependency): <https://github.com/LCAV/pyroomacoustics>

The original M2M-TSE Scaper notice is retained verbatim in
`licenses/UPSTREAM_THIRD_PARTY_NOTICES.txt` for source provenance. SPATEX's
LibriSpeech dataset does not import the vendored `scaper_edited` or
`pyloudnorm_edited` packages, so those packages are omitted.

Installed dependencies retain their own licenses. The distributed checkpoint is
the authors' trained SPATEX model. Third-party speech corpora, pretrained baseline
weights and downloaded baseline repositories are not bundled.
