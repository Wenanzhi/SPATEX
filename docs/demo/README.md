# SPATEX demo

Open `index.html` in a browser, or serve the repository with
`python -m http.server 8000` and visit `http://localhost:8000/docs/demo/`.
The page has no external JavaScript, font or chart dependencies.

## Reproduce the assets

From the repository root, using the single `requirements.txt` environment:

```bash
python scripts/build_demo.py --testset-root /path/to/Testset/seg_6s --device cuda:0
```

The generator uses the released variable-array epoch-85 checkpoint. For each
condition, it selects the first saved scene in filename order matching the
geometry and microphone count: four-mic linear, six-mic planar and eight-mic
random. Selection does not depend on model scores.

- Audio and charts show real model outputs for six-second saved test mixtures.
- SI-SNRi is measured at the selected microphone. Spatial errors average all
  valid microphone pairs in that scene. Scores are computed before export.
- A common gain, at most one, is applied to every signal and microphone in a
  scene for PCM16 browser playback. Signals are never normalized independently.
- All waveform panels share the scene's amplitude scale. All spectrograms use
  the same scene-wide magnitude reference, with a −70 to 0 dB range.
- The multichannel download preserves the unscaled FLOAT WAV output.
- The geometry plot shows the XY projection of microphone coordinates relative
  to their centroid. The target arrow indicates direction only, not distance.

Speech is from the LibriSpeech test-clean split with simulated room responses.
`manifest.json` records checkpoint/configuration hashes, source-audio hashes,
scene metadata and display conventions. No aggregate benchmark results are
computed by this demo.
