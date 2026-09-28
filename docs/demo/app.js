(() => {
  'use strict';
  const data = window.SPATEX_DEMO;
  const byId = id => document.getElementById(id);
  if (!data?.scenes?.length) {
    byId('load-status').textContent = 'Example data could not be loaded. Please check that data.js is beside this page.';
    return;
  }
  const names = {mixture: 'Input mixture', estimate: 'SPATEX', reference: 'Target reference'};
  const colors = {mixture: '#7a8796', estimate: '#087f81', reference: '#607eac'};
  const descriptions = {
    fixed: 'Four microphones with 4 cm spacing. The variable-array model is used directly, without fixed-array fine-tuning.',
    matched: 'A regular planar grid from a geometry family used in training, with six valid microphone channels.',
    unmatched: 'An irregular eight-microphone array with 1 cm position perturbations, outside the training geometry families.'
  };
  const conditions = {fixed: 'Fixed-array evaluation', matched: 'Matched geometry', unmatched: 'Unmatched geometry'};
  const state = {scene: 0, mic: 0, track: 'estimate', view: 'waveform'};
  const player = byId('player');
  const images = new Map();
  let audioVersion = 0;
  const current = () => data.scenes[state.scene];
  const channel = () => current().channels[state.mic];
  const announce = text => { byId('announcer').textContent = text; };

  function updateTrackUI() {
    document.querySelectorAll('[data-track]').forEach(el => el.classList.toggle('selected', el.dataset.track === state.track));
    document.querySelectorAll('[data-listen]').forEach(button => {
      const selected = button.dataset.listen === state.track;
      button.setAttribute('aria-pressed', String(selected));
      button.textContent = selected && !player.paused ? 'Pause' : 'Play';
      button.setAttribute('aria-label', `${button.textContent} ${names[button.dataset.listen]} at microphone ${state.mic + 1}`);
    });
    byId('now-playing').textContent = `${names[state.track]} · Mic ${state.mic + 1}`;
    player.setAttribute('aria-label', `${names[state.track]} at microphone ${state.mic + 1}`);
  }

  function changeAudio(keepTime, resume) {
    const position = keepTime && Number.isFinite(player.currentTime) ? player.currentTime : 0;
    const version = ++audioVersion;
    player.pause();
    player.src = channel().tracks[state.track].audio;
    player.onloadedmetadata = () => {
      if (version !== audioVersion) return;
      player.currentTime = Math.min(position, Math.max(0, player.duration - .02));
      if (resume) player.play().catch(() => announce('Use the audio play button to start playback.'));
      drawCharts();
    };
    player.load();
    updateTrackUI();
  }

  function drawGeometry() {
    const scene = current();
    const host = byId('geometry-plot');
    const w = Math.max(240, host.clientWidth), h = 255;
    const center = [scene.mic_xyz.reduce((s,p)=>s+p[0],0)/scene.microphones, scene.mic_xyz.reduce((s,p)=>s+p[1],0)/scene.microphones];
    const points = scene.mic_xyz.map(p => [p[0]-center[0],p[1]-center[1]]);
    const radius = Math.max(.08,...points.flat().map(Math.abs)) * 1.25;
    const scale = Math.min(w-94,h-66)/(radius*2), cx=w/2, cy=(h-12)/2;
    const px=x=>cx+x*scale, py=y=>cy-y*scale;
    const azimuth=scene.azimuth*Math.PI/180, length=radius*.88;
    let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" aria-label="${scene.microphones} microphone positions; target azimuth ${scene.azimuth.toFixed(1)} degrees"><defs><marker id="arrow-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#b96e22"/></marker></defs>`;
    const tick=Math.ceil(radius*100/5)*5/100;
    [-tick/2,0,tick/2].forEach(value => {
      svg+=`<line x1="${px(-radius)}" x2="${px(radius)}" y1="${py(value)}" y2="${py(value)}" stroke="#e9eef2"/><line x1="${px(value)}" x2="${px(value)}" y1="${py(-radius)}" y2="${py(radius)}" stroke="#e9eef2"/><text x="${px(value)}" y="${py(-radius)+18}" text-anchor="middle" font-size="11" fill="#687686">${(value*100).toFixed(0)}</text><text x="${px(-radius)-9}" y="${py(value)+4}" text-anchor="end" font-size="11" fill="#687686">${(value*100).toFixed(0)}</text>`;
    });
    svg+=`<text x="${px(radius)+12}" y="${cy+4}" font-size="11" fill="#687686">x</text><text x="${cx+6}" y="${py(radius)-8}" font-size="11" fill="#687686">y</text><line x1="${cx}" y1="${cy}" x2="${px(Math.sin(azimuth)*length)}" y2="${py(Math.cos(azimuth)*length)}" stroke="#b96e22" stroke-width="2" marker-end="url(#arrow-tip)"/><circle cx="${cx}" cy="${cy}" r="2" fill="#b96e22"/>`;
    points.forEach((p,i) => {
      const selected=i===state.mic;
      svg+=`<g data-mic="${i}" role="button" tabindex="0" aria-label="Select microphone ${i+1}" style="cursor:pointer"><title>Mic ${i+1}: x ${(p[0]*100).toFixed(1)}, y ${(p[1]*100).toFixed(1)} cm</title><circle cx="${px(p[0])}" cy="${py(p[1])}" r="11" fill="transparent"/><circle cx="${px(p[0])}" cy="${py(p[1])}" r="${selected?7:5}" fill="${selected?'#087f81':'#8dbad2'}" stroke="white" stroke-width="2"/>${selected?`<text x="${px(p[0])+11}" y="${py(p[1])-9}" font-size="12" font-weight="600" fill="#087f81">Mic ${i+1}</text>`:''}</g>`;
    });
    host.innerHTML=svg+'</svg>';
    host.querySelectorAll('[data-mic]').forEach(el => {
      const choose=()=>selectMic(Number(el.dataset.mic));
      el.addEventListener('click',choose);
      el.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});
    });
  }

  function drawCharts() {
    const scene=current();
    for(const kind of Object.keys(names)) {
      const canvas=byId('chart-'+kind), box=canvas.getBoundingClientRect();
      const w=box.width,h=box.height,dpr=window.devicePixelRatio||1;
      if(!w||!h) continue;
      canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
      const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
      const left=35,top=10,right=9,bottom=29,pw=w-left-right,ph=h-top-bottom;
      ctx.clearRect(0,0,w,h);ctx.font='11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      const track=channel().tracks[kind];
      if(state.view==='waveform') {
        ctx.strokeStyle='#e8edf1';ctx.lineWidth=1;
        [0,.5,1].forEach(v=>{ctx.beginPath();ctx.moveTo(left,top+ph*v);ctx.lineTo(left+pw,top+ph*v);ctx.stroke();});
        const amp=Math.max(scene.waveform_peak,1e-6);
        const y=v=>top+ph/2-v/amp*ph/2;
        ctx.fillStyle=colors[kind];ctx.beginPath();
        track.envelope.forEach((pair,i)=>{const x=left+i/(track.envelope.length-1)*pw; i?ctx.lineTo(x,y(pair[1])):ctx.moveTo(x,y(pair[1]));});
        for(let i=track.envelope.length-1;i>=0;i--)ctx.lineTo(left+i/(track.envelope.length-1)*pw,y(track.envelope[i][0]));
        ctx.closePath();ctx.fill();ctx.fillStyle='#697786';ctx.textAlign='right';
        ctx.fillText(amp.toFixed(2),left-5,top+4);ctx.fillText('0',left-5,top+ph/2+4);ctx.fillText('-'+amp.toFixed(2),left-5,top+ph+3);
      } else {
        let img=images.get(track.spectrogram);
        if(!img){img=new Image();images.set(track.spectrogram,img);img.onload=drawCharts;img.src=track.spectrogram;}
        if(img.complete&&img.naturalWidth)ctx.drawImage(img,left,top,pw,ph);
        ctx.fillStyle='#697786';ctx.textAlign='right';
        ctx.fillText('4 kHz',left-5,top+5);ctx.fillText('2',left-5,top+ph/2+4);ctx.fillText('0',left-5,top+ph+3);
      }
      ctx.fillStyle='#697786';ctx.textAlign='center';
      [0,scene.duration/2,scene.duration].forEach(t=>ctx.fillText(t.toFixed(0),left+t/scene.duration*pw,top+ph+17));
      ctx.textAlign='right';ctx.fillText('s',w-1,h-1);
      const time=Number.isFinite(player.currentTime)?player.currentTime:0;
      if(time>0){ctx.strokeStyle=state.view==='spectrogram'?'#ffffff':'#253649';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left+time/scene.duration*pw,top);ctx.lineTo(left+time/scene.duration*pw,top+ph);ctx.stroke();}
      canvas.setAttribute('aria-label',`${names[kind]} ${state.view} at microphone ${state.mic+1}, ${scene.duration} seconds`);
    }
  }

  function selectMic(index) {
    const playing=!player.paused; state.mic=index; byId('mic-select').value=String(index);
    byId('sisnr').textContent=channel().si_snri.toFixed(2)+' dB';
    byId('sisnr-label').textContent=`SI-SNRi · Mic ${index+1}`;
    drawGeometry();changeAudio(true,playing);drawCharts();
    announce(`Microphone ${index+1} selected.`);
  }

  function selectScene(index) {
    state.scene=index;state.mic=0;const scene=current();
    document.querySelectorAll('[data-scene]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.scene)===index)));
    byId('scene-condition').textContent=conditions[scene.id];byId('scene-title').textContent=scene.title;
    byId('scene-description').textContent=descriptions[scene.id];byId('mic-count').textContent=scene.microphones;
    byId('azimuth').textContent=scene.azimuth.toFixed(1)+'°';byId('rt60').textContent=scene.rt60.toFixed(2)+' s';
    byId('duration').textContent=scene.duration.toFixed(0)+' s / 8 kHz';
    byId('download').href=scene.multichannel_download;
    byId('mic-select').replaceChildren(...scene.channels.map((c,i)=>new Option('Mic '+c.index,i)));
    byId('ipd').textContent=scene.delta_ipd.toFixed(3)+' rad';byId('itd').textContent=scene.delta_itd_us.toFixed(2)+' μs';
    byId('ild').textContent=scene.delta_ild.toFixed(3)+' dB';
    byId('sisnr').textContent=channel().si_snri.toFixed(2)+' dB';byId('sisnr-label').textContent='SI-SNRi · Mic 1';
    byId('scene-source').textContent=`Saved scene: seg_6s/${scene.subset}/${scene.source_scene}`;
    changeAudio(false,false);drawGeometry();drawCharts();announce(scene.title+' selected.');
  }

  data.scenes.forEach((scene,index)=>{
    const button=document.createElement('button');button.type='button';button.dataset.scene=index;
    const title=document.createElement('strong');title.textContent=scene.title;
    const detail=document.createElement('span');detail.textContent=scene.microphones+' microphones · '+scene.duration+' seconds';
    button.append(title,detail);button.addEventListener('click',()=>selectScene(index));byId('scene-tabs').append(button);
  });
  byId('mic-select').addEventListener('change',event=>selectMic(Number(event.target.value)));
  document.querySelectorAll('[data-listen]').forEach(button=>button.addEventListener('click',()=>{
    if(state.track===button.dataset.listen){player.paused?player.play().catch(()=>announce('Audio could not be played.')):player.pause();}
    else{state.track=button.dataset.listen;changeAudio(true,true);}
    updateTrackUI();
  }));
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
    state.view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
    updateDisplayNote();drawCharts();
  }));
  function updateDisplayNote(){byId('display-note').textContent=state.view==='waveform'?'Waveforms share one amplitude scale across signals and microphones. Playback uses a shared scene gain.':'Spectrograms: 256-sample Hann STFT, 128-sample hop. Common −70 to 0 dB scale relative to the scene-wide maximum; dark = quieter, light = louder.';}
  player.addEventListener('play',updateTrackUI);player.addEventListener('pause',updateTrackUI);
  player.addEventListener('timeupdate',drawCharts);player.addEventListener('error',()=>announce('Audio unavailable. Check that the audio folder is included.'));
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{drawGeometry();drawCharts();},80);});
  byId('load-status').hidden=true;byId('demo-content').hidden=false;updateDisplayNote();selectScene(0);
})();
