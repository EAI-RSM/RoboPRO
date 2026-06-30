import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// surface any failure on-page instead of a blank canvas
function fatal(msg) {
  const el = document.getElementById('loading');
  if (el) { el.style.display = 'flex'; el.style.color = '#f85149'; el.style.padding = '24px';
    el.style.textAlign = 'center'; el.innerHTML =
      'Demo failed to start:<br><br><code style="color:#e6edf3">' + msg +
      '</code><br><br>Open this page through a local web server ' +
      '(<code>python -m http.server</code>), not via file://';
  }
}
window.addEventListener('error', e => fatal((e.error && e.error.stack) || e.message));
window.addEventListener('unhandledrejection', e => fatal('' + (e.reason && (e.reason.stack || e.reason))));
function status(msg) {
  const el = document.getElementById('loading');
  if (el && el.style.color !== 'rgb(248, 81, 73)') el.textContent = msg;
}

// ---------------------------------------------------------------- load manifest
status('fetching scene.json …');
const SCENE = await (await fetch('data/scenes/clean_baseline/scene.json')).json().catch(err => { fatal('scene.json: ' + err); throw err; });
const M = SCENE.meta, NF = M.n_frames, A = SCENE.objects.length;
const OP = SCENE.frames.object_pos, OQ = SCENE.frames.object_quat;

// ---------------------------------------------------------------- three.js setup
const host = document.getElementById('canvas-host');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);                       // SAPIEN world is Z-up

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.1, -1.25, 1.45);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.0, -0.05, M.table_top_z + 0.04);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xbcd2ff, 0x202830, 1.05));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(1.2, -1.0, 2.6); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5; key.shadow.camera.far = 8;
key.shadow.camera.left = -2; key.shadow.camera.right = 2;
key.shadow.camera.top = 2; key.shadow.camera.bottom = -2;
scene.add(key);
scene.add(new THREE.DirectionalLight(0xffffff, 0.4).translateZ(3));

function resize() {
  let w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) { w = Math.max(320, window.innerWidth - 360); h = Math.max(240, window.innerHeight - 64); }
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);

// tiny debug readout (so a black screen is diagnosable without a console)
const dbg = document.createElement('div');
dbg.style.cssText = 'position:absolute;top:8px;right:8px;z-index:20;font:11px monospace;' +
  'color:#8b949e;background:rgba(13,17,23,.7);padding:4px 7px;border-radius:6px;white-space:pre';
host.parentElement.appendChild(dbg);

// ---------------------------------------------------------------- helpers
const loader = new GLTFLoader();
let nWant = 0, nGot = 0;
function gltf(url, onLoad) {
  nWant++;
  loader.load(url, (g) => { nGot++; status(`loading meshes … ${nGot}/${nWant}`); onLoad(g); },
    undefined, (err) => fatal('failed to load ' + url + '\n' + (err && err.message || err)));
}
const segMaterials = new Map();          // mesh -> {orig, seg}
function registerSeg(mesh, colorArr) {
  const seg = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colorArr[0], colorArr[1], colorArr[2]),
    roughness: 0.85, metalness: 0.0, flatShading: false });
  segMaterials.set(mesh, { orig: mesh.material, seg });
}
function setPose(obj, p, q) {                   // p=[x,y,z]  q=[x,y,z,w]
  obj.position.set(p[0], p[1], p[2]);
  obj.quaternion.set(q[0], q[1], q[2], q[3]);
}

// ---------------------------------------------------------------- environment
const envGroup = new THREE.Group(); scene.add(envGroup);
(function buildEnv() {
  const ztop = M.table_top_z;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 8),
    new THREE.MeshStandardMaterial({ color: 0xe9edf2, roughness: 1 }));
  floor.receiveShadow = true; envGroup.add(floor);

  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.95, 0.04),
    new THREE.MeshStandardMaterial({ color: 0xd9dde3, roughness: 0.9 }));
  top.position.set(0.1, -0.05, ztop - 0.02); top.receiveShadow = true; top.castShadow = true;
  envGroup.add(top);
  const ped = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 0.7, ztop - 0.05),
    new THREE.MeshStandardMaterial({ color: 0xc6ccd4, roughness: 1 }));
  ped.position.set(0.1, -0.05, (ztop - 0.05) / 2); envGroup.add(ped);

  const wall = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 5),
    new THREE.MeshStandardMaterial({ color: 0xdfe4ea, roughness: 1, side: THREE.DoubleSide }));
  wall.position.set(0, 1.0, 2.0); wall.rotation.x = Math.PI / 2; envGroup.add(wall);
})();

// destination pad (primitive box at its pose) — kept separate so it updates per frame
let padMesh = null, padIndex = -1;

// ---------------------------------------------------------------- objects
const objRoots = [];                        // per-actor root Object3D or null
SCENE.objects.forEach((o, i) => {
  if (o.glb) {
    const root = new THREE.Group(); scene.add(root); objRoots[i] = root;
    gltf(o.glb, (g) => {
      const inner = g.scene;
      // SAPIEN's add_visual_from_file uses the glb coords as-is (each glb bakes its own
      // up-axis node transform), so we honor the loaded nodes and add NO correction.
      inner.scale.set(...o.scale);
      inner.traverse((m) => {
        if (m.isMesh) { m.castShadow = true; m.receiveShadow = true; registerSeg(m, o.color); }
      });
      root.add(inner);
    });
  } else if (o.primitive === 'box' && o.role === 'destination') {
    padIndex = i;
    padMesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.23, 0.17, 0.010),
      new THREE.MeshStandardMaterial({ color: 0x2d333d, roughness: 0.7 }));
    padMesh.castShadow = false; padMesh.receiveShadow = true;
    registerSeg(padMesh, o.color); scene.add(padMesh); objRoots[i] = padMesh;
  } else {
    objRoots[i] = null;                     // ground/table/wall handled by envGroup
  }
});

// ---------------------------------------------------------------- robot
const robotGroup = new THREE.Group(); scene.add(robotGroup);
const robMeta = { name: 'robot', color: [0.62, 0.66, 0.72] };
const movingLinks = [];                     // {group}
if (SCENE.robot.static_glb) {
  gltf(SCENE.robot.static_glb, (g) => {
    g.scene.traverse(m => { if (m.isMesh) { m.castShadow = true; registerSeg(m, robMeta.color); } });
    robotGroup.add(g.scene);
  });
}
(SCENE.robot.moving_links || []).forEach((lk, j) => {
  const grp = new THREE.Group(); robotGroup.add(grp); movingLinks[j] = grp;
  gltf(lk.glb, (g) => {
    g.scene.traverse(m => { if (m.isMesh) { m.castShadow = true; registerSeg(m, robMeta.color); } });
    grp.add(g.scene);
  });
});
const FL6 = (SCENE.robot.moving_links || []).findIndex(l => l.name === 'fl_link6');

// ---------------------------------------------------------------- trajectories
const trajGroup = new THREE.Group(); trajGroup.visible = false; scene.add(trajGroup);
(function buildTraj() {
  const mi = SCENE.signals && SCENE.objects.findIndex(o => o.name === SCENE.signals.target);
  function line(getXYZ, color) {
    const pts = [];
    for (let t = 0; t < NF; t++) { const p = getXYZ(t); if (p) pts.push(new THREE.Vector3(p[0], p[1], p[2])); }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    trajGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color })));
  }
  if (mi >= 0) line(t => [OP[t][mi * 3], OP[t][mi * 3 + 1], OP[t][mi * 3 + 2]], 0xf85149);
  if (FL6 >= 0) {
    const NL = SCENE.robot.n_links, LP = SCENE.robot.link_pose;
    line(t => [LP[t][FL6 * 7], LP[t][FL6 * 7 + 1], LP[t][FL6 * 7 + 2]], 0xd29922);
  }
})();

// ---------------------------------------------------------------- camera frusta
const frustaGroup = new THREE.Group(); frustaGroup.visible = false; scene.add(frustaGroup);
(function buildFrusta() {
  const STATIC = new Set(['head_camera', 'front_camera']);
  SCENE.cameras.filter(c => STATIC.has(c.name) && c.cam2world_gl).forEach(c => {
    const m = c.cam2world_gl[0];
    const C2W = new THREE.Matrix4().set(
      m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
      m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
    const K = c.K, W = c.width, H = c.height, d = 0.5;
    const origin = new THREE.Vector3().setFromMatrixPosition(C2W);
    const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) => {
      const dir = new THREE.Vector3((u - K[0][2]) / K[0][0], -(v - K[1][2]) / K[1][1], -1)
        .multiplyScalar(d);
      return dir.applyMatrix4(new THREE.Matrix4().extractRotation(C2W)).add(origin);
    });
    const pts = [origin, corners[0], origin, corners[1], origin, corners[2], origin, corners[3],
      corners[0], corners[1], corners[1], corners[2], corners[2], corners[3], corners[3], corners[0]];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    frustaGroup.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x58a6ff })));
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.012),
      new THREE.MeshBasicMaterial({ color: 0x58a6ff }));
    s.position.copy(origin); frustaGroup.add(s);
  });
})();

// ---------------------------------------------------------------- per-frame update
const S = SCENE.signals;
function applyFrame(t) {
  for (let i = 0; i < A; i++) {
    const r = objRoots[i]; if (!r) continue;
    setPose(r, [OP[t][i * 3], OP[t][i * 3 + 1], OP[t][i * 3 + 2]],
      [OQ[t][i * 4], OQ[t][i * 4 + 1], OQ[t][i * 4 + 2], OQ[t][i * 4 + 3]]);
  }
  if (SCENE.robot.link_pose) {
    const LP = SCENE.robot.link_pose;
    movingLinks.forEach((grp, j) => setPose(grp,
      [LP[t][j * 7], LP[t][j * 7 + 1], LP[t][j * 7 + 2]],
      [LP[t][j * 7 + 3], LP[t][j * 7 + 4], LP[t][j * 7 + 5], LP[t][j * 7 + 6]]));
  }
  updatePanel(t);
  if (rgbCam) updateRgb(t);
}

function updatePanel(t) {
  document.getElementById('relation').textContent = S.relation[t];
  document.getElementById('s-dist').textContent = S.dist_xy_cm[t].toFixed(1) + ' cm';
  document.getElementById('s-height').textContent = S.target_height_cm[t].toFixed(1) + ' cm';
  document.getElementById('s-perr').textContent = S.place_error_cm.toFixed(2) + ' cm';
  const grip = SCENE.robot;                 // gripper "closed" proxy: lifted while near target
  chip('c-lift', S.lifted[t], 'lift');
  chip('c-grasp', S.lifted[t] && S.dist_xy_cm[t] < 6, 'grasp');
  chip('c-place', S.placed[t], 'place');
  const placedNow = S.placed[t];
  const el = document.getElementById('s-success');
  el.textContent = placedNow ? '✓ mouse on pad' : '… in progress';
  el.style.color = placedNow ? 'var(--good)' : 'var(--dim)';
  document.getElementById('framelbl').textContent = `frame ${t} / ${NF - 1}`;
}
function chip(id, on, cls) {
  const e = document.getElementById(id);
  e.className = 'chip' + (on ? ' on ' + cls : '');
}

// ---------------------------------------------------------------- collected RGB compare
let rgbCam = null;
const rgbBox = document.getElementById('rgb-compare');
const rgbImg = document.getElementById('rgb-img');
function updateRgb(t) {
  rgbImg.src = `data/rgb/${rgbCam}/f${String(t).padStart(4, '0')}.jpg`;
  document.getElementById('rgb-cap').textContent = `collected RGB · ${rgbCam}`;
}

// ---------------------------------------------------------------- camera presets
const camButtons = document.getElementById('cam-buttons');
const camNote = document.getElementById('cam-note');
function snapTo(camName) {
  const c = SCENE.cameras.find(x => x.name === camName);
  const m = (c.cam2world_gl || [])[frame] || c.cam2world_gl[0];
  const C2W = new THREE.Matrix4().set(
    m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
    m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
  const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
  C2W.decompose(pos, quat, scl);
  camera.position.copy(pos); camera.quaternion.copy(quat);
  camera.fov = 2 * Math.atan((c.height / 2) / c.K[1][1]) * 180 / Math.PI;
  camera.updateProjectionMatrix();
  controls.enabled = false;
}
function freeView() {
  controls.enabled = true; controls.target.set(0.0, -0.05, M.table_top_z + 0.04);
  camera.fov = 45; camera.updateProjectionMatrix();
}
const PRESETS = [
  { label: 'Free orbit', collected: false, act: () => { rgbCam = null; rgbBox.style.display = 'none';
      freeView(); tagFree(); } },
  { label: 'Head cam', cam: 'head_camera', rgb: true },
  { label: 'External cam', cam: 'demo_camera', rgb: true },
];
PRESETS.forEach((p, idx) => {
  const b = document.createElement('button'); b.className = 'cam'; b.textContent = p.label;
  b.onclick = () => {
    [...camButtons.children].forEach(c => c.classList.remove('active')); b.classList.add('active');
    if (p.cam) {
      rgbCam = p.rgb ? p.cam : null;
      rgbBox.style.display = p.rgb ? 'block' : 'none';
      snapTo(p.cam); updateRgb(frame);
      camNote.innerHTML = `Matching the collected <b>${p.cam}</b>. The 3D scene is reconstructed; ` +
        `the inset is the real recorded pixels — they line up. Now hit <b>Free orbit</b>.`;
      setTag(`reproducing collected camera: <b>${p.cam}</b>`);
    } else { p.act(); }
  };
  camButtons.appendChild(b);
});
function tagFree() {
  camNote.innerHTML = 'Drag to orbit to <b>any</b> viewpoint — including ones the cameras never captured.';
  setTag('<b>novel view</b> — never collected, rendered from the state trace');
}
function setTag(html) { document.getElementById('view-tag').innerHTML = html; }
controls.addEventListener('start', () => {
  if (controls.enabled) { setTag('<b>novel view</b> — never collected, rendered from the state trace'); }
});

// ---------------------------------------------------------------- toggles
document.getElementById('t-seg').onchange = e => {
  const seg = e.target.checked;
  segMaterials.forEach((v, mesh) => { mesh.material = seg ? v.seg : v.orig; });
};
document.getElementById('t-robot').onchange = e => robotGroup.visible = e.target.checked;
document.getElementById('t-frusta').onchange = e => frustaGroup.visible = e.target.checked;
document.getElementById('t-traj').onchange = e => trajGroup.visible = e.target.checked;
document.getElementById('t-env').onchange = e => envGroup.visible = e.target.checked;

// ---------------------------------------------------------------- transport
let frame = 0, playing = false, acc = 0, last = performance.now();
const scrub = document.getElementById('scrub'); scrub.max = NF - 1;
const playbtn = document.getElementById('playbtn');
scrub.oninput = () => { frame = +scrub.value; applyFrame(frame); };
playbtn.onclick = () => { playing = !playing; playbtn.textContent = playing ? '❚❚' : '▶'; };

// ---------------------------------------------------------------- header + storage
document.getElementById('title').innerHTML =
  `RoboPRO · <b>${M.task_name}</b> <span class="badge good">${M.outcome}</span>`;
document.getElementById('subtitle').innerHTML =
  `“${M.instruction}” &nbsp;·&nbsp; seed ${M.seed} &nbsp;·&nbsp; ${NF} frames &nbsp;·&nbsp; ` +
  `reconstructed offline from the logged state trace`;
const st = SCENE.storage;
document.getElementById('st-kb').textContent = st.state_trace_kb + ' KB';
document.getElementById('st-mb').textContent = st.pixels_mb + ' MB';
document.getElementById('st-cams').textContent = st.n_cameras;
document.getElementById('st-ratio').textContent = st.ratio_pct + '%';
document.getElementById('st-bar').style.width = Math.max(st.ratio_pct, 0.6) + '%';

// ---------------------------------------------------------------- run
resize(); applyFrame(0);
[...camButtons.children][0].click();        // start in free-orbit
status(nWant ? `loading meshes … ${nGot}/${nWant}` : 'rendering …');

let firstFrame = true;
function tick(now) {
  const dt = (now - last) / 1000; last = now;
  if (playing) {
    acc += dt * M.fps;
    if (acc >= 1) { frame = (frame + Math.floor(acc)) % NF; acc = 0; scrub.value = frame; applyFrame(frame); }
  }
  if (controls.enabled) controls.update();   // when snapped to a collected cam, hold the pose
  renderer.render(scene, camera);
  const sz = renderer.getSize(new THREE.Vector2());
  dbg.textContent = `canvas ${sz.x|0}x${sz.y|0}  meshes ${nGot}/${nWant}\n` +
    `cam ${camera.position.x.toFixed(2)},${camera.position.y.toFixed(2)},${camera.position.z.toFixed(2)}` +
    `  frame ${frame}`;
  if (firstFrame) {                          // hide overlay only once we've actually drawn a frame
    firstFrame = false;
    const el = document.getElementById('loading');
    if (el && el.style.color !== 'rgb(248, 81, 73)') el.style.display = 'none';
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
