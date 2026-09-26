// PraMD live demo: type an instruction, the local server generates Unitree G1 motion with the
// frozen OMG-100M generator and the Run 3 adapter, and this module plays it back on the G1 URDF.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const $ = (id) => document.getElementById(id);
const host = $('live-view');
if (host) init();

function init() {
  // Deployed: the model runs on a Gradio Space named in <meta name="pramd-gradio">.
  // Served locally (localhost), the page uses the local FastAPI server instead; ?gradio=<url> overrides.
  const LOCAL = ['127.0.0.1', 'localhost'].includes(location.hostname);
  const metaGradio = document.querySelector('meta[name="pramd-gradio"]');
  const GRADIO = (new URLSearchParams(location.search).get('gradio')
    || (!LOCAL && metaGradio && metaGradio.content.trim()) || '').replace(/\/$/, '') || null;
  let gradioPrefix = null;          // '/gradio_api' on current Gradio, '' on older releases
  const API_BASES = [...new Set([
    location.protocol.startsWith('http') ? `${location.origin}/api` : null,
    'http://127.0.0.1:8765/api',
  ].filter(Boolean))];

  // Gradio's HTTP API: POST starts a job and returns an event id; GET streams its result as
  // server-sent events ending in "complete" (data = [output]) or "error" (data = message).
  async function gradioCall(name, data) {
    const prefixes = gradioPrefix !== null ? [gradioPrefix] : ['/gradio_api', ''];
    let lastError = new Error('model server unreachable');
    for (const prefix of prefixes) {
      let r;
      try {
        r = await fetch(`${GRADIO}${prefix}/call/${name}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data }),
        });
      } catch (e) { lastError = e; continue; }
      if (r.status === 404) { lastError = new Error('endpoint not found'); continue; }
      if (!r.ok) throw new Error(`model server returned ${r.status}`);
      const { event_id: eventId } = await r.json();
      gradioPrefix = prefix;
      const stream = await fetch(`${GRADIO}${prefix}/call/${name}/${eventId}`);
      let event = null;
      for (const line of (await stream.text()).split('\n')) {
        if (line.startsWith('event:')) {
          event = line.slice(6).trim();
        } else if (line.startsWith('data:') && (event === 'complete' || event === 'error')) {
          const raw = line.slice(5).trim();
          let payload = null;
          try { payload = JSON.parse(raw); } catch (_) { payload = raw; }
          if (event === 'error') {        // Gradio sends {error, title, ...}, not a bare message
            const message = payload && typeof payload === 'object' ? payload.error || payload.message : payload;
            throw new Error(message || 'the model server reported an error');
          }
          return payload[0];
        }
      }
      throw new Error('the model server returned no result');
    }
    throw lastError;
  }
  let api = null;
  let jointNames = null;
  let seedFrame = null;
  let clip = null;             // { frames: number[][], fps }
  let playing = false;
  let elapsed = 0;
  let last = 0;
  let mode = 'pramd';
  let busy = false;

  // ------------------------------------------------------------------ scene
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.prepend(renderer.domElement);

  const BG = 0x141618;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 7, 20);

  const camera = new THREE.PerspectiveCamera(36, 16 / 9, 0.05, 100);
  camera.position.set(2.05, 1.2, 2.25);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.72, 0);
  controls.enableDamping = true;
  controls.minDistance = 1.2;
  controls.maxDistance = 9;
  controls.maxPolarAngle = Math.PI * 0.49;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x303338, 1.15));
  const sun = new THREE.DirectionalLight(0xffffff, 1.7);
  sun.position.set(2.5, 6, 3);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 20 });
  scene.add(sun, sun.target);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ color: 0x17191c, roughness: 1 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(80, 80, 0x8a9098, 0x3a3e43);
  grid.position.y = 0.002;
  scene.add(grid);

  // OMG and the URDF are z-up; three.js is y-up.
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;
  scene.add(world);

  let robot = null;
  new URDFLoader().load(new URL('../robot/g1_29dof.urdf', import.meta.url).href, (r) => {
    robot = r;
    r.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      const c = o.material && o.material.color;
      const dark = c && c.r + c.g + c.b < 1.2;
      o.material = new THREE.MeshStandardMaterial({
        color: dark ? 0x2a2d31 : 0xe9eaec, roughness: dark ? 0.6 : 0.45, metalness: 0.05,
      });
    });
    world.add(r);
    if (seedFrame) pose(seedFrame);
    host.classList.add('ready');
  });

  const tmp = new THREE.Vector3();
  function pose(q) {
    if (!robot || !q || !jointNames) return;
    robot.position.set(q[0], q[1], q[2]);
    robot.quaternion.set(q[4], q[5], q[6], q[3]);          // qpos is wxyz
    for (let i = 0; i < jointNames.length; i++) robot.setJointValue(jointNames[i], q[7 + i]);
  }

  function follow(snap) {
    if (!robot) return;
    robot.getWorldPosition(tmp);
    const goal = new THREE.Vector3(tmp.x, 0.72, tmp.z);
    const delta = goal.sub(controls.target).multiplyScalar(snap ? 1 : 0.08);
    controls.target.add(delta);
    camera.position.add(delta);
    sun.position.set(controls.target.x + 2.5, 6, controls.target.z + 3);
    sun.target.position.copy(controls.target);
  }

  function resize() {
    const w = host.clientWidth;
    const h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);
  resize();

  // ------------------------------------------------------------------ playback
  const scrub = $('live-scrub');
  const playBtn = $('live-play');
  const clock = $('live-clock');

  function setFrame(i) {
    if (!clip) return;
    pose(clip.frames[i]);
    scrub.value = String(i);
    clock.textContent = `${(i / clip.fps).toFixed(1)} / ${((clip.frames.length - 1) / clip.fps).toFixed(1)} s`;
  }

  function setPlaying(p) {
    playing = p && !!clip;
    playBtn.textContent = playing ? 'Pause' : 'Play';
  }

  function tick(ts) {
    requestAnimationFrame(tick);
    const dt = last ? Math.min((ts - last) / 1000, 0.1) : 0;
    last = ts;
    if (clip && playing) {
      elapsed += dt;
      let i = Math.floor(elapsed * clip.fps);
      if (i >= clip.frames.length) {       // loop, with the camera snapping back to the start
        elapsed = 0;
        i = 0;
        pose(clip.frames[0]);
        follow(true);
      }
      setFrame(i);
    }
    follow(false);
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(tick);

  playBtn.addEventListener('click', () => setPlaying(!playing));
  scrub.addEventListener('input', () => {
    if (!clip) return;
    setPlaying(false);
    const i = +scrub.value;
    elapsed = i / clip.fps;
    setFrame(i);
  });

  // ------------------------------------------------------------------ server
  const status = $('live-status');
  const meta = $('live-meta');
  const goBtn = $('live-go');
  const text = $('live-text');

  function setStatus(msg, kind = '') {
    status.textContent = msg;
    status.dataset.kind = kind;
  }

  async function connect() {
    for (const base of API_BASES) {
      try {
        const r = await fetch(`${base}/health`, { cache: 'no-store' });
        if (!r.ok) continue;
        api = base;
        return await r.json();
      } catch (_) { /* try the next base */ }
    }
    return null;
  }

  async function poll() {
    let h;
    if (GRADIO) {
      try {
        h = await gradioCall('info', []);
        api = GRADIO;
      } catch (_) {
        h = null;
      }
    } else {
      h = await connect();
    }
    if (!h) {
      setStatus('The model server is starting or waking up. After a period of inactivity this can take a '
        + 'few minutes; this page keeps retrying.', 'off');
      goBtn.disabled = true;
      setTimeout(poll, 5000);
      return;
    }
    if (h.joint_names && !jointNames) {
      jointNames = h.joint_names;
      seedFrame = h.seed_frame;
      pose(seedFrame);
      follow(true);
    }
    if (h.error) {
      setStatus(`The model failed to load: ${h.error}`, 'err');
      goBtn.disabled = true;
      return;
    }
    if (!h.ready) {
      setStatus(`Loading the model: ${h.stage}…`, 'wait');
      goBtn.disabled = true;
      setTimeout(poll, 1500);
      return;
    }
    setStatus(`Model ready on ${h.device.toUpperCase()}.`, 'ok');
    goBtn.disabled = busy;
  }

  async function generate() {
    const instruction = text.value.trim();
    if (!instruction || !api || busy) return;
    busy = true;
    goBtn.disabled = true;
    setPlaying(false);
    const seconds = +$('live-dur').value;
    const started = Date.now();
    const tick = () => setStatus(`Generating ${seconds} s of motion… ${Math.round((Date.now() - started) / 1000)} s`, 'wait');
    tick();
    const timer = setInterval(tick, 1000);
    try {
      let j;
      if (GRADIO) {
        j = await gradioCall('generate', [instruction, seconds, mode, 0]);
      } else {
        const r = await fetch(`${api}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: instruction, seconds, mode }),
        });
        j = await r.json();
        if (!r.ok) throw new Error(j.detail || r.statusText);
      }
      jointNames = j.joint_names;
      clip = { frames: j.frames, fps: j.fps };
      scrub.max = String(clip.frames.length - 1);
      scrub.disabled = false;
      playBtn.disabled = false;
      elapsed = 0;
      setFrame(0);
      follow(true);
      setPlaying(true);
      setStatus(`Generated in ${(j.gen_ms / 1000).toFixed(1)} s on ${j.device.toUpperCase()}.`, 'ok');
      meta.textContent = `${j.language} · ${j.path} · ${(clip.frames.length / clip.fps).toFixed(1)} s`
        + (j.note ? ` · ${j.note}` : '');
    } catch (e) {
      setStatus(visitorMessage(e.message), 'err');
    } finally {
      clearInterval(timer);
      busy = false;
      goBtn.disabled = !api;
    }
  }

  // The GPU allowance is per visitor: when it runs out, say so in plain words and point at the
  // examples, which play from stored motions and need no server.
  function visitorMessage(message) {
    if (/zerogpu|quota|runs limit/i.test(message)) {
      return 'The free GPU time for your visit is used up for now. The examples below still play, '
        + 'and live generation works again after a short wait.';
    }
    return `Generation failed: ${message}`;
  }

  goBtn.addEventListener('click', generate);
  text.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate();
  });
  // Clicking an example plays its verified motion straight away, with no server round trip, so the
  // examples work even while the model server is asleep. Generate still runs the model live.
  async function playPrecomputed(ex, data) {
    try {
      const r = await fetch(new URL(ex.motion, import.meta.url));
      if (!r.ok) throw new Error(r.statusText);
      const m = await r.json();
      if (!jointNames && data.joint_names) jointNames = data.joint_names;
      clip = { frames: m.frames, fps: m.fps };
      scrub.max = String(clip.frames.length - 1);
      scrub.disabled = false;
      playBtn.disabled = false;
      elapsed = 0;
      setFrame(0);
      follow(true);
      setPlaying(true);
      meta.textContent = `Precomputed by this model for this example (${data.seconds} s, seed ${data.seed}). `
        + 'Press Generate motion to run it live.';
    } catch (_) {
      meta.textContent = '';                         // no stored motion: Generate runs it live
    }
  }

  // Examples: held-out captions chosen by select_examples.py, one verified list per language.
  const LANGS = [['hi', 'Hindi'], ['bn', 'Bengali'], ['ta', 'Tamil'], ['te', 'Telugu'], ['en', 'English']];
  const exTabs = $('live-ex-tabs');
  const exList = $('live-examples');
  const exNote = $('live-ex-note');
  fetch(new URL('./examples.json', import.meta.url), { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
    .then((data) => {
      const langs = LANGS.filter(([code]) => (data.languages[code] || []).length);
      const show = (code) => {
        exTabs.querySelectorAll('button')
          .forEach((b) => b.setAttribute('aria-selected', String(b.dataset.lang === code)));
        exList.replaceChildren(...data.languages[code].map((ex) => {
          const button = document.createElement('button');
          button.type = 'button';
          const t = document.createElement('span');
          t.className = 't';
          t.textContent = ex.text;
          button.append(t);
          if (code !== 'en') {
            const g = document.createElement('span');
            g.className = 'g';
            g.textContent = ex.en;
            button.append(g);
          }
          button.addEventListener('click', () => {
            text.value = ex.text;
            $('live-dur').value = String(data.seconds);   // the length each example was verified at
            if (ex.motion) playPrecomputed(ex, data);
          });
          const li = document.createElement('li');
          li.append(button);
          return li;
        }));
      };
      exTabs.replaceChildren(...langs.map(([code, name]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.dataset.lang = code;
        b.textContent = name;
        b.addEventListener('click', () => show(code));
        return b;
      }));
      if (langs.length) show(langs[0][0]);
      exNote.textContent = 'Held-out test captions the model never trained on. Each was generated here, '
        + 'kept only if the motion passed a check for every action it names, and reviewed visually.';
    })
    .catch(() => { exNote.textContent = 'Examples could not be loaded.'; });
  document.querySelectorAll('#live-mode button').forEach((b) => {
    b.addEventListener('click', () => {
      mode = b.dataset.mode;
      document.querySelectorAll('#live-mode button')
        .forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    });
  });

  poll();
}
