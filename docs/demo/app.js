(() => {
  'use strict';
  const data = window.SPATEX_DEMO;
  const status = document.getElementById('load-status');
  if (!data?.scenes?.length) {
    status.textContent = 'Audio examples could not be loaded. Please reload this page.';
    return;
  }

  const names = {mixture: 'Mixture', estimate: 'SPATEX', reference: 'Target reference'};
  const colors = {mixture: '#8590a3', estimate: '#315abc', reference: '#5a8081'};
  const labels = {
    fixed: ['4-mic linear', 'Fixed array'],
    matched: ['6-mic planar', 'Matched geometry'],
    unmatched: ['8-mic random', 'Unmatched geometry']
  };
  const kinds = Object.keys(names);
  const rows = [];
  const announcer = document.getElementById('announcer');
  const dialog = document.getElementById('spectrogram-dialog');
  let activeAudio = null;

  function alignTime(row, time, source) {
    row.time = Math.max(0, Math.min(time, row.scene.duration));
    for (const player of row.players) {
      if (player !== source && player.readyState >= 1 && Math.abs(player.currentTime - row.time) > .12) {
        player.currentTime = row.time;
      }
    }
    drawWaveforms(row);
  }

  function drawWaveforms(row) {
    const channel = row.scene.channels[row.mic];
    row.element.querySelectorAll('.waveform').forEach(canvas => {
      const {width: w, height: h} = canvas.getBoundingClientRect();
      if (!w || !h) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const kind = canvas.dataset.kind;
      const envelope = channel.tracks[kind].envelope;
      const compact = h < 50;
      const bottom = compact ? 3 : 16, top = 5, plotHeight = h - bottom - top;
      const amplitude = Math.max(row.scene.waveform_peak, 1e-6);
      const x = i => 2 + i / (envelope.length - 1) * (w - 4);
      const y = value => top + plotHeight / 2 - value / amplitude * plotHeight / 2;
      ctx.strokeStyle = '#e5e9f0';
      ctx.beginPath();ctx.moveTo(0, y(0));ctx.lineTo(w, y(0));ctx.stroke();
      ctx.fillStyle = colors[kind];
      ctx.beginPath();
      envelope.forEach((pair, i) => i ? ctx.lineTo(x(i), y(pair[1])) : ctx.moveTo(x(i), y(pair[1])));
      for (let i = envelope.length - 1; i >= 0; i--) ctx.lineTo(x(i), y(envelope[i][0]));
      ctx.closePath();ctx.fill();
      if (!compact) {
        ctx.font = '10px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
        ctx.fillStyle = '#7b8494';
        ctx.textAlign = 'left';ctx.fillText('0', 1, h - 1);
        ctx.textAlign = 'center';ctx.fillText('3', w / 2, h - 1);
        ctx.textAlign = 'right';ctx.fillText('6 s', w - 1, h - 1);
      }
      if (row.time > 0) {
        const cursor = row.time / row.scene.duration * w;
        ctx.strokeStyle = '#25334e';ctx.lineWidth = 1;
        ctx.beginPath();ctx.moveTo(cursor, top);ctx.lineTo(cursor, h - bottom);ctx.stroke();
      }
      canvas.setAttribute('aria-label', `${names[kind]} waveform, microphone ${row.mic + 1}, ${row.scene.duration} seconds; common scene amplitude scale`);
    });
  }

  function drawGeometry(row) {
    const scene = row.scene, host = row.element.querySelector('.geometry-plot');
    const center = [0, 1].map(axis => scene.mic_xyz.reduce((sum, p) => sum + p[axis], 0) / scene.microphones);
    const points = scene.mic_xyz.map(p => [p[0] - center[0], p[1] - center[1]]);
    const radius = Math.max(.08, ...points.flat().map(Math.abs)) * 1.3;
    const scale = 68 / radius, px = x => 110 + x * scale, py = y => 86 - y * scale;
    const a = scene.azimuth * Math.PI / 180, length = radius * .82;
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 186" role="group" aria-label="${scene.microphones} microphones and target direction; XY projection in centimetres"><defs><marker id="arrow-${scene.id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 10 5 0 10Z" fill="#bb7834"/></marker></defs>`;
    const tick = Math.ceil(radius * 100 / 5) * 5 / 200;
    [-tick, 0, tick].forEach(v => {
      svg += `<path d="M${px(-radius)} ${py(v)}H${px(radius)}M${px(v)} ${py(-radius)}V${py(radius)}" fill="none" stroke="#e8ecf2"/><text x="${px(v)}" y="170" text-anchor="middle" font-size="10" fill="#737e90">${(v * 100).toFixed(0)}</text><text x="29" y="${py(v) + 3}" text-anchor="end" font-size="10" fill="#737e90">${(v * 100).toFixed(0)}</text>`;
    });
    svg += `<text x="193" y="90" font-size="10" fill="#737e90">x</text><text x="114" y="11" font-size="10" fill="#737e90">y</text><path d="M110 86L${px(Math.sin(a) * length)} ${py(Math.cos(a) * length)}" stroke="#bb7834" stroke-width="1.6" marker-end="url(#arrow-${scene.id})"/>`;
    points.forEach((p, i) => {
      const selected = row.mic === i;
      svg += `<g data-mic="${i}" role="button" tabindex="0" aria-label="Select microphone ${i + 1}" aria-pressed="${selected}" style="cursor:pointer"><title>Mic ${i + 1}: x ${(p[0] * 100).toFixed(1)}, y ${(p[1] * 100).toFixed(1)} cm</title><circle cx="${px(p[0])}" cy="${py(p[1])}" r="10" fill="transparent"/><circle cx="${px(p[0])}" cy="${py(p[1])}" r="${selected ? 5.5 : 3.5}" fill="${selected ? '#2a50bd' : '#98acca'}" stroke="#fff" stroke-width="1"/>${selected ? `<text x="${px(p[0]) + 9}" y="${py(p[1]) - 8}" font-size="10" fill="#2a50bd" stroke="#fff" stroke-width="3" paint-order="stroke" pointer-events="none">Mic ${i + 1}</text>` : ''}</g>`;
    });
    host.innerHTML = svg + '</svg>';
    host.querySelectorAll('[data-mic]').forEach(button => {
      button.addEventListener('click', () => selectMic(row, Number(button.dataset.mic), true));
      button.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();selectMic(row, Number(button.dataset.mic), true);
        }
      });
    });
  }

  function updateDetails(row) {
    if (!row.details.open) return;
    drawGeometry(row);
    const channel = row.scene.channels[row.mic];
    row.element.querySelectorAll('.spectrum').forEach(figure => {
      const kind = figure.dataset.kind, img = figure.querySelector('img');
      img.src = channel.tracks[kind].spectrogram;
      img.alt = `${names[kind]} spectrogram, ${labels[row.scene.id][0]}, microphone ${row.mic + 1}`;
      figure.querySelector('button').setAttribute('aria-label', 'Enlarge ' + img.alt);
    });
    row.element.querySelector('[data-metric="sisnr"]').textContent = channel.si_snri.toFixed(2) + ' dB';
    row.element.querySelector('[data-metric="sisnr-label"]').textContent = `SI-SNRi · Mic ${row.mic + 1}`;
  }

  function selectMic(row, index, restoreFocus = false) {
    const resume = row.players.find(player => !player.paused);
    const resumeKind = resume?.dataset.kind;
    if (resume) row.time = resume.currentTime;
    if (row.players.includes(activeAudio)) activeAudio = null;
    const version = ++row.version;
    row.mic = index;
    row.element.querySelector('select').value = String(index);
    const channel = row.scene.channels[index];
    row.players.forEach(player => {
      player.pause();
      player.onloadedmetadata = () => {
        if (version !== row.version) return;
        player.currentTime = Math.min(row.time, Math.max(0, player.duration - .02));
        if (player.dataset.kind === resumeKind) {
          player.play().catch(() => { announcer.textContent = 'Press play to resume audio.'; });
        }
      };
      player.src = channel.tracks[player.dataset.kind].audio;
      player.setAttribute('aria-label', `${names[player.dataset.kind]}, ${labels[row.scene.id][0]}, microphone ${index + 1}`);
      player.load();
    });
    updateDetails(row);drawWaveforms(row);
    if (restoreFocus) row.element.querySelector(`[data-mic="${index}"]`)?.focus({preventScroll: true});
    announcer.textContent = `${labels[row.scene.id][0]}: microphone ${index + 1} selected.`;
  }

  data.scenes.forEach((scene, sceneIndex) => {
    const element = document.createElement('article');
    element.className = 'scene';element.dataset.scene = scene.id;
    element.setAttribute('aria-labelledby', `title-${scene.id}`);
    element.innerHTML = `
      <div class="scene-row comparison-grid">
        <div class="scene-info">
          <h3 id="title-${scene.id}"><span class="scene-number">0${sceneIndex + 1}</span>${labels[scene.id][0]}</h3>
          <p class="condition">${labels[scene.id][1]}</p>
          <label class="mic-picker" for="mic-${scene.id}">Listen at <select id="mic-${scene.id}" aria-label="Microphone for ${labels[scene.id][0]}">${scene.channels.map((c, i) => `<option value="${i}">Mic ${c.index}</option>`).join('')}</select></label>
          <p class="direction">Target direction ${scene.azimuth.toFixed(1)}°</p>
        </div>
        ${kinds.map(kind => `<div class="track" data-kind="${kind}"><h4>${names[kind]}</h4><canvas class="waveform" data-kind="${kind}" role="img"></canvas><audio controls preload="metadata" data-kind="${kind}"></audio></div>`).join('')}
      </div>
      <details class="scene-details">
        <summary>Array, spectrograms &amp; metrics</summary>
        <div class="detail-body">
          <div class="detail-grid comparison-grid">
            <div class="geometry-panel"><h4>Microphone positions</h4><div class="geometry-plot"></div><p>XY projection · cm<br>Arrow: target direction<br>T₆₀ = ${scene.rt60.toFixed(2)} s<br>Click a mic to listen.</p></div>
            ${kinds.map(kind => `<figure class="spectrum" data-kind="${kind}"><figcaption>${names[kind]} · 0–4 kHz</figcaption><button type="button"><img alt="" width="600" height="240"></button><div class="axis"><span>0</span><span>3</span><span>6 s</span></div></figure>`).join('')}
          </div>
          <p class="spectral-note">Shared −70 to 0 dB scale across signals and microphones. Click a spectrogram to enlarge.</p>
          <dl class="scene-metrics"><div><dt data-metric="sisnr-label">SI-SNRi · Mic 1</dt><dd data-metric="sisnr"></dd></div><div><dt>ΔIPD · all mic pairs</dt><dd>${scene.delta_ipd.toFixed(3)} rad</dd></div><div><dt>ΔITD · all mic pairs</dt><dd>${scene.delta_itd_us.toFixed(2)} μs</dd></div><div><dt>ΔILD · all mic pairs</dt><dd>${scene.delta_ild.toFixed(3)} dB</dd></div></dl>
          <div class="detail-bottom"><p>Single-example scores · Saved scene ${scene.source_scene} · 6 s</p><a href="${scene.multichannel_download}" download>Download all ${scene.microphones} output channels ↗</a></div>
        </div>
      </details>`;
    document.getElementById('scene-list').append(element);
    const row = {scene, element, mic: 0, time: 0, version: 0, players: [...element.querySelectorAll('audio')], details: element.querySelector('details')};
    rows.push(row);
    row.players.forEach(player => {
      player.addEventListener('play', () => {
        if (activeAudio !== player && row.players.includes(activeAudio)) row.time = activeAudio.currentTime;
        const position = row.time >= scene.duration - .03 ? 0 : row.time;
        document.querySelectorAll('audio').forEach(other => { if (other !== player) other.pause(); });
        activeAudio = player;
        if (Math.abs(player.currentTime - position) > .12) player.currentTime = position;
        player.closest('.track').classList.add('is-playing');
      });
      player.addEventListener('pause', () => player.closest('.track').classList.remove('is-playing'));
      player.addEventListener('timeupdate', () => { if (activeAudio === player && !player.seeking) alignTime(row, player.currentTime, player); });
      player.addEventListener('seeking', () => {
        if (player.readyState >= 1 && Math.abs(player.currentTime - row.time) > .15) alignTime(row, player.currentTime, player);
      });
      player.addEventListener('ended', () => { activeAudio = null;alignTime(row, 0); });
      player.addEventListener('error', () => {
        announcer.textContent = `${names[player.dataset.kind]} is unavailable. Please reload the page.`;
      });
    });
    element.querySelector('select').addEventListener('change', event => selectMic(row, Number(event.target.value)));
    row.details.addEventListener('toggle', () => updateDetails(row));
    element.querySelectorAll('.spectrum button').forEach(button => button.addEventListener('click', () => {
      const img = button.querySelector('img');
      const preview = document.getElementById('dialog-image');
      preview.src = img.src;preview.alt = img.alt;
      document.getElementById('dialog-title').textContent = img.alt;
      dialog.showModal();
    }));
    selectMic(row, 0);
  });

  document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);resizeTimer = setTimeout(() => rows.forEach(drawWaveforms), 80);
  });
  status.hidden = true;
  announcer.textContent = '';
})();
