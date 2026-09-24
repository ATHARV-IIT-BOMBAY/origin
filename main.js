// HoloControl — rotate & zoom a 3D model with webcam hand gestures.
// Flow: webcam -> MediaPipe HandLandmarker -> gesture math (gestures.js) -> three.js transform.
// No LLM, no backend. The hand model is only a *sensor*; the interaction is hand-written geometry.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { HandLandmarker, FilesetResolver, DrawingUtils }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
import { handCenter, pinchStrength, twoHandSpread, twoHandAngle, landmarkToWorld, handPose, INDEX_TIP } from './gestures.js';

// ---- Tuning knobs. A webcam is a messy sensor; these are the calibration dials. ----
// The first four are `let` because the on-screen calibration panel adjusts them live.
let ROT_SPEED = 6.0;      // how far a hand move rotates the model
let ZOOM_SPEED = 6.0;     // how strongly two-hand spread scales the model
let SMOOTH = 0.20;        // 0..1 low-pass ease; lower = smoother but laggier
let MIRROR_X = -1;        // flip so moving your hand right rotates the model right
const SCALE_MIN = 0.3, SCALE_MAX = 4.0;
const HAND_SPAN = 4.2;    // how wide the tracked hand maps into the 3D scene
const HAND_DEPTH = 1.5;   // how strongly landmark depth pushes hand joints in/out
const IDLE_SPIN = 0.0015; // lazy auto-rotate (rad/frame) when you're not controlling it
// Flick-to-spin: while pinch-dragging we track the swipe velocity; on release the model keeps
// spinning and eases to rest (friction). CAPTURE = how much the latest frame feeds the estimate.
const SPIN_FRICTION = 0.96, SPIN_CAPTURE = 0.5, SPIN_MIN = 0.001;
const EXPLODE_K = 1.6;                        // full-explosion expansion: parts push out from center
const EXPLODE_MIN = 0.20, EXPLODE_MAX = 0.75; // two-hand spread range mapped onto 0..1 explosion
const BASE_EMISSIVE = 0.6, HIGHLIGHT_EMISSIVE = 2.4; // part glow: resting vs. aimed-at
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const video = $('video'), overlay = $('overlay'), stateEl = $('state'), errEl = $('err'), poseEl = $('pose');
const modeEl = $('modePill'), recEl = $('rec');

// ---------- HUD: live mode pill + stats readout ----------
const MODE_LABEL = { rotate:'ROTATE', zoom:'ZOOM', explode:'EXPLODE', inspect:'INSPECT', park:'PARKED', pause:'PAUSED', idle:'IDLE' };
function updateModePill(m) { if (modeEl) { modeEl.textContent = MODE_LABEL[m] || m.toUpperCase(); modeEl.className = 'pill ' + m; } }
function updateStats(nHands) {
  $('stHands').textContent = nHands;
  $('stParts').textContent = parts.length;
  $('stScale').textContent = Math.round(current.scale * 100) + '%';
  $('stExpl').textContent = Math.round(current.explode * 100) + '%';
}

// ---------- Web Audio: tiny synth blips for gesture feedback (no audio files) ----------
// A gesture is invisible until it "clicks" — a short tone on each mode change makes the
// interface feel physical. Ctx is created on a user click (autoplay policy) and reused.
let audioCtx = null;
function initAudio() {
  if (!audioCtx) { try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch {} }
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}
function blip(freq = 660, dur = 0.08, type = 'sine', peak = 0.06) {
  if (!audioCtx) return;
  const t = audioCtx.currentTime, osc = audioCtx.createOscillator(), g = audioCtx.createGain();
  osc.type = type; osc.frequency.setValueAtTime(freq, t);
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(peak, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(g).connect(audioCtx.destination); osc.start(t); osc.stop(t + dur + 0.02);
}
const SFX = {                       // sound played when we ENTER each mode (fires once per transition)
  rotate:  () => blip(520, 0.07, 'sine'),
  zoom:    () => blip(760, 0.09, 'triangle'),
  explode: () => blip(300, 0.12, 'sawtooth', 0.05),
  inspect: () => blip(880, 0.05, 'square', 0.03),
  park:    () => blip(200, 0.10, 'sine'),
  pause:   () => blip(150, 0.16, 'sine'),
};
let lastMode = '';
function setMode(m) { updateModePill(m); if (m === lastMode) return; lastMode = m; (SFX[m] || null)?.(); }

// ---------- three.js scene ----------
// preserveDrawingBuffer lets the Snapshot button read the canvas back as a PNG.
const renderer = new THREE.WebGLRenderer({ canvas: $('three'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 0, 4);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; // soft PBR reflections
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(3, 4, 5); scene.add(key);

// Post-processing: bloom makes the emissive wireframe model and the hand joints glow,
// which is what turns a plain mesh into a "hologram". Threshold keeps dim things un-bloomed.
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.9, 0.5, 0.7);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// Faint "holo-deck" grid so the model reads as floating in a space, not on a black void.
const grid = new THREE.GridHelper(20, 40, 0x1e6fff, 0x0a2a4a);
grid.position.y = -1.6; scene.add(grid);

// Holo-projector base: two counter-rotating tech rings under the model (bloom makes them glow).
const reticle = new THREE.Group(); reticle.position.y = -1.55; scene.add(reticle);
const _up = new THREE.Vector3(0, 1, 0);
for (const [rIn, rOut, dir] of [[1.70, 1.86, 1], [1.96, 2.02, -1]]) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(rIn, rOut, 96),
    new THREE.MeshBasicMaterial({ color: 0x2aa0ff, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.userData.dir = dir; reticle.add(ring);
}

const pivot = new THREE.Group(); // we rotate/scale this; the model lives inside it
scene.add(pivot);

// Hologram-styled part: translucent lit core + bright wireframe overlay (the bit that blooms).
// Grouped so a whole part can be raycast, highlighted, and flown out as a single unit.
function holoPart(geo, name, label, color = 0x0a2a4a, emissive = 0x0aa0ff) {
  const core = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color, emissive, emissiveIntensity: BASE_EMISSIVE,
    metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.8 }));
  const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.45 }));
  core.userData.holoNative = true; wire.userData.holoWire = true;  // so applyHoloSkin leaves the rocket alone
  const g = new THREE.Group(); g.add(core, wire);
  g.name = name; g.userData = { label, core };
  return g;
}

// Default model: a multi-part rocket (nose, tanks, engine, fins). It's recognizable, and its
// separately-named parts are exactly what the point-to-inspect and explode gestures act on.
function makeRocket() {
  const R = new THREE.Group(), r = 0.42;
  const nose = holoPart(new THREE.ConeGeometry(r, 0.9, 32), 'nose', 'Nose Cone', 0x143a5a, 0x22c0ff);
  nose.position.y = 1.15;
  const tankA = holoPart(new THREE.CylinderGeometry(r, r, 0.9, 32), 'tankA', 'Fuel Tank');
  tankA.position.y = 0.35;
  const tankB = holoPart(new THREE.CylinderGeometry(r, r, 0.9, 32), 'tankB', 'Oxidizer Tank');
  tankB.position.y = -0.55;
  const engine = holoPart(new THREE.CylinderGeometry(r * 0.55, r, 0.5, 32), 'engine', 'Engine Bell', 0x3a1e08, 0xff7a1a);
  engine.position.y = -1.2;
  R.add(nose, tankA, tankB, engine);
  for (let i = 0; i < 3; i++) {                        // three stabilizer fins around the base
    const fin = holoPart(new THREE.BoxGeometry(0.06, 0.5, 0.42), 'fin' + i, 'Stabilizer Fin');
    const a = (i / 3) * Math.PI * 2;
    fin.position.set(Math.cos(a) * (r + 0.16), -0.95, Math.sin(a) * (r + 0.16));
    fin.rotation.y = -a;
    R.add(fin);
  }
  return R;
}
let model = makeRocket();
pivot.add(model);

function fitToView(obj) { // center at origin and normalize size so any model frames nicely
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  obj.position.sub(box.getCenter(new THREE.Vector3()));
  obj.scale.multiplyScalar(1.8 / (Math.max(size.x, size.y, size.z) || 1));
}
fitToView(model);

// Parts = the model's direct children that carry geometry. Each remembers its resting
// position (home); the explode gesture pushes every part radially out from the model's center
// (an exploded-view expansion) and snaps them home. Parts sitting at dead-center don't travel.
let parts = [], partSet = new Set();
function firstMesh(o) { let m = null; o.traverse(c => { if (!m && c.isMesh) m = c; }); return m; }
function collectParts(root) {
  parts = root.children.filter(firstMesh);
  partSet = new Set(parts);
  for (const p of parts) {
    p.userData.home = p.position.clone();
    if (!p.userData.core) p.userData.core = firstMesh(p);
    if (p.userData.label == null) p.userData.label = p.name || 'Part';
  }
}
collectParts(model);

// ---------- holographic skin: re-dress ANY uploaded model in the JARVIS look ----------
// Uploaded models arrive with their own materials; this overrides every mesh with a translucent
// emissive-blue material + a cyan wireframe overlay so it reads as a hologram like the built-in
// rocket. Originals are cached so the toggle can restore them. The procedural rocket is marked
// holoNative/holoWire and skipped (it's already holo). Default on — that's the whole aesthetic.
let holoSkin = true, autoSpin = true;
function applyHoloSkin(root, on) {
  root.traverse(o => {
    if (!o.isMesh || o.userData.holoNative || o.userData.holoWire) return;
    if (on) {
      if (!o.userData.origMat) o.userData.origMat = o.material;
      if (!o.userData.holoMat) o.userData.holoMat = new THREE.MeshStandardMaterial({
        color: 0x0a2a4a, emissive: 0x0aa0ff, emissiveIntensity: BASE_EMISSIVE,
        metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.82 });
      if (!o.userData.wire) {
        const w = new THREE.Mesh(o.geometry, new THREE.MeshBasicMaterial({
          color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.28 }));
        w.userData.holoWire = true; w.raycast = () => {};   // never intercept point-to-inspect rays
        o.add(w); o.userData.wire = w;
      }
      o.material = o.userData.holoMat; o.userData.wire.visible = true;
    } else if (o.userData.origMat) {
      o.material = o.userData.origMat;
      if (o.userData.wire) o.userData.wire.visible = false;
    }
  });
}

// ---------- hand skeleton drawn INTO the 3D scene (your hand appears inside the hologram) ----------
const MAX_JOINTS = 42;               // 2 hands * 21 landmarks
const MAX_BONES = 2 * 24;            // 2 hands * (HAND_CONNECTIONS is 21, 24 is safe headroom)
const joints = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.045, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0xffffff }), MAX_JOINTS);   // per-instance color set in updateHandViz
joints.frustumCulled = false; scene.add(joints);
const JOINT_LIVE = new THREE.Color(0x7CFFB2), JOINT_PARK = new THREE.Color(0xff5a5a); // green = live, red = parked (fist)
const boneGeo = new THREE.BufferGeometry();
boneGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_BONES * 2 * 3), 3));
const bones = new THREE.LineSegments(boneGeo, new THREE.LineBasicMaterial({ color: 0x7CFFB2, transparent: true, opacity: 0.85 }));
bones.frustumCulled = false; scene.add(bones);

// Fingertip comet-trails: a short additive line trailing each hand's index tip, fading tail->tip (JARVIS feel).
const TRAIL_LEN = 18, TRAIL_RGB = [0.4, 0.9, 1.0];
const trails = [0, 1].map(() => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
  const line = new THREE.Line(g, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  line.frustumCulled = false; scene.add(line);
  return { line, hist: [] };
});
function updateTrail(slot, tip) {            // tip = world {x,y,z}, or null to shrink the tail away
  const t = trails[slot]; if (!t) return;
  if (tip) { t.hist.push(tip); while (t.hist.length > TRAIL_LEN) t.hist.shift(); }
  else if (t.hist.length) t.hist.shift();
  const n = t.hist.length, P = t.line.geometry.attributes.position.array, C = t.line.geometry.attributes.color.array;
  for (let k = 0; k < n; k++) {
    const p = t.hist[k], f = n > 1 ? k / (n - 1) : 1;    // 0 at tail, 1 at fingertip
    P[k*3] = p.x; P[k*3+1] = p.y; P[k*3+2] = p.z;
    C[k*3] = TRAIL_RGB[0]*f; C[k*3+1] = TRAIL_RGB[1]*f; C[k*3+2] = TRAIL_RGB[2]*f;
  }
  t.line.geometry.setDrawRange(0, n);
  t.line.geometry.attributes.position.needsUpdate = true;
  t.line.geometry.attributes.color.needsUpdate = true;
}
const _m = new THREE.Matrix4(), _hidden = new THREE.Matrix4().makeScale(0, 0, 0);
function hideHands() {
  for (let k = 0; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden);
  joints.instanceMatrix.needsUpdate = true; boneGeo.setDrawRange(0, 0);
  for (const t of trails) { t.hist.length = 0; t.line.geometry.setDrawRange(0, 0); }
}
function updateHandViz(hands) {
  const conns = HandLandmarker.HAND_CONNECTIONS, pos = boneGeo.attributes.position.array;
  let j = 0, v = 0, hi = 0; // joint index, bone-vertex index, hand slot
  for (const hand of hands) {
    const world = hand.map(p => landmarkToWorld(p, HAND_SPAN, HAND_DEPTH));
    const col = handPose(hand) === 'fist' ? JOINT_PARK : JOINT_LIVE;     // parked hand's joints glow red
    for (const p of world) if (j < MAX_JOINTS) { _m.makeTranslation(p.x, p.y, p.z); joints.setMatrixAt(j, _m); joints.setColorAt(j, col); j++; }
    for (const c of conns) if (v + 2 <= MAX_BONES * 2) {
      const a = world[c.start], d = world[c.end];
      pos.set([a.x, a.y, a.z, d.x, d.y, d.z], v * 3); v += 2;
    }
    updateTrail(hi++, world[INDEX_TIP]);
  }
  for (let s = hi; s < trails.length; s++) updateTrail(s, null); // fade out trails for absent hands
  for (let k = j; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden); // hide unused instances
  joints.instanceMatrix.needsUpdate = true;
  if (joints.instanceColor) joints.instanceColor.needsUpdate = true;
  boneGeo.setDrawRange(0, v); boneGeo.attributes.position.needsUpdate = true;
}
hideHands();

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

// ---------- load your own model (object URL avoids any CORS/backend) ----------
// Pick a loader by file extension and normalize its result to an Object3D we can drop in.
// STL carries only geometry, so we wrap it in a holo-styled mesh. .blend is Blender's own
// binary format — no browser can read it; the standard path is to export glTF from Blender.
const LOADERS = {
  glb:  { L: GLTFLoader, pick: r => r.scene },
  gltf: { L: GLTFLoader, pick: r => r.scene },
  obj:  { L: OBJLoader,  pick: r => r },
  fbx:  { L: FBXLoader,  pick: r => r },
  stl:  { L: STLLoader,  pick: g => new THREE.Mesh(g, new THREE.MeshStandardMaterial({
           color: 0x143a5a, emissive: 0x0aa0ff, emissiveIntensity: BASE_EMISSIVE, metalness: 0.3, roughness: 0.35 })) },
};
function swapModel(obj) {
  pivot.remove(model);
  model = obj; fitToView(model); pivot.add(model);
  collectParts(model); applyHoloSkin(model, holoSkin); clearHighlight();
  target.explode = current.explode = 0; target.rx = target.ry = target.rz = 0; spinVel.rx = spinVel.ry = 0;
  updateStats(0);   // refresh the parts count for the newly loaded model (live tracking re-fills hands next frame)
}
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
  initAudio();
  const ext = (f.name.split('.').pop() || '').toLowerCase();
  if (ext === 'blend') {
    errEl.textContent = ".blend can't be read in the browser. In Blender: File ▸ Export ▸ glTF 2.0 (.glb), then load that file.";
    e.target.value = ''; return;
  }
  const entry = LOADERS[ext];
  if (!entry) { errEl.textContent = `Unsupported file “.${ext}” — use .glb, .gltf, .obj, .fbx or .stl`; e.target.value = ''; return; }
  errEl.textContent = ''; stateEl.textContent = `Loading ${f.name}…`;
  const url = URL.createObjectURL(f);
  new entry.L().load(url, (res) => {
    swapModel(entry.pick(res)); URL.revokeObjectURL(url);
    stateEl.textContent = `Loaded ${f.name}`; blip(720, 0.14, 'triangle');
  }, undefined, (err) => { URL.revokeObjectURL(url); errEl.textContent = 'Could not load model: ' + err; });
});

// ---------- HUD controls: holo skin · showcase toggles · snapshot · live calibration ----------
$('tHolo').addEventListener('change', (e) => { holoSkin = e.target.checked; applyHoloSkin(model, holoSkin); });
$('tSpin').addEventListener('change', (e) => { autoSpin = e.target.checked; });
$('tBloom').addEventListener('change', (e) => { bloom.enabled = e.target.checked; });
$('tGrid').addEventListener('change', (e) => { grid.visible = reticle.visible = e.target.checked; });
$('snap').addEventListener('click', () => {                 // one-tap PNG of the holo render (preserveDrawingBuffer)
  initAudio(); blip(900, 0.08, 'triangle');
  const a = document.createElement('a');
  a.download = 'holocontrol-' + Date.now() + '.png';
  a.href = renderer.domElement.toDataURL('image/png'); a.click();
});
$('tuneBtn').addEventListener('click', () => { initAudio(); $('tune').classList.toggle('hidden'); });
function bindRange(id, apply, outId, fmt) {                 // wire a slider to a live tuning variable
  const el = $(id), out = $(outId);
  const upd = () => { const v = parseFloat(el.value); apply(v); out.textContent = fmt(v); };
  el.addEventListener('input', upd); upd();
}
bindRange('sRot', (v) => ROT_SPEED = v, 'vRot', (v) => v.toFixed(1));
bindRange('sZoom', (v) => ZOOM_SPEED = v, 'vZoom', (v) => v.toFixed(1));
bindRange('sSmooth', (v) => SMOOTH = v, 'vSmooth', (v) => v.toFixed(2));
$('tMirror').addEventListener('change', (e) => { MIRROR_X = e.target.checked ? -1 : 1; });

// ---------- transform state: gestures set `target`, each frame eases `current` toward it ----------
const target = { rx: 0, ry: 0, rz: 0, scale: 1, explode: 0 };
const current = { rx: 0, ry: 0, rz: 0, scale: 1, explode: 0 };
$('reset').addEventListener('click', () => { initAudio(); blip(420, 0.12, 'sine'); target.rx = 0; target.ry = 0; target.rz = 0; target.scale = 1; target.explode = 0; spinVel.rx = spinVel.ry = 0; });

// ---------- point-to-inspect: raycast from the camera through the index fingertip ----------
// The fingertip is drawn into the scene at a fixed plane; a ray from the camera through it
// continues on to whatever part sits behind it — so aiming reads exactly as it looks on screen.
const labelEl = $('label');
const raycaster = new THREE.Raycaster();
const _tip = new THREE.Vector3(), _proj = new THREE.Vector3();
let highlighted = null;
function findPart(obj) { while (obj) { if (partSet.has(obj)) return obj; obj = obj.parent; } return null; }
function setHighlight(part) {
  if (highlighted && highlighted !== part) {
    const m = highlighted.userData.core?.material;
    if (m && 'emissiveIntensity' in m) m.emissiveIntensity = BASE_EMISSIVE;
  }
  highlighted = part;
  const m = part?.userData.core?.material;
  if (m && 'emissiveIntensity' in m) m.emissiveIntensity = HIGHLIGHT_EMISSIVE;
}
function clearHighlight() { setHighlight(null); labelEl.style.opacity = '0'; }
function pointAt(hand) {                          // highlight+label the aimed part; return its label or null
  const w = landmarkToWorld(hand[INDEX_TIP], HAND_SPAN, HAND_DEPTH);
  raycaster.set(camera.position, _tip.set(w.x, w.y, w.z).sub(camera.position).normalize());
  const hit = raycaster.intersectObject(pivot, true)[0];
  const part = hit ? findPart(hit.object) : null;
  setHighlight(part);
  if (!part) { labelEl.style.opacity = '0'; return null; }
  _proj.copy(hit.point).project(camera);
  const x = (_proj.x * 0.5 + 0.5) * innerWidth, y = (-_proj.y * 0.5 + 0.5) * innerHeight;
  labelEl.textContent = part.userData.label;
  labelEl.style.transform = `translate(-50%,-150%) translate(${x}px,${y}px)`;
  labelEl.style.opacity = '1';
  return part.userData.label;
}

// ---------- MediaPipe hand tracking ----------
let handLandmarker = null, drawUtils = null, running = false, lastVideoTime = -1;
let prevCenter = null, prevSpread = null, prevAngle = null, idle = true;
let dragging = false;                 // true while a pinch is actively rotating the model
const spinVel = { rx: 0, ry: 0 };     // leftover angular velocity after you let go (flick-to-spin)

async function initHands() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = { baseOptions: { modelAssetPath: MODEL_URL }, runningMode: 'VIDEO', numHands: 2 };
  try { // GPU is faster but unavailable on some machines — fall back to CPU
    handLandmarker = await HandLandmarker.createFromOptions(vision, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } });
  } catch { handLandmarker = await HandLandmarker.createFromOptions(vision, opts); }
  drawUtils = new DrawingUtils(overlay.getContext('2d'));
}

async function startCamera() {
  try {
    errEl.textContent = ''; initAudio();
    if (!handLandmarker) { stateEl.textContent = 'Loading model…'; await initHands(); }
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    await video.play();
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    running = true; stateEl.textContent = 'Show a hand';
    if (recEl) { recEl.textContent = 'LIVE'; recEl.classList.add('on'); }
    const boot = $('boot');
    if (boot) { $('bootsub').textContent = 'HAND TRACKING ONLINE'; blip(600, 0.12, 'triangle'); setTimeout(() => boot.classList.add('hidden'), 900); }
  } catch (err) { errEl.textContent = 'Camera/model error: ' + err; }
}
$('start').addEventListener('click', startCamera);

function applyGestures(hands) {
  // A fist "parks" a hand — we ignore it — so you can drive one-hand gestures with your other
  // hand while both stay comfortably in frame (no yanking a hand out of view to change modes).
  // Two fists = pause everything. We classify each hand once, then act on the live ones only.
  const classified = hands.map(h => ({ h, pose: handPose(h) }));
  const active = classified.filter(a => a.pose !== 'fist');
  const parked = hands.length - active.length;
  dragging = false;                                                        // set true only while rotating
  poseEl.textContent = classified.map((a, i) => `H${i + 1} ${a.pose} ${pinchStrength(a.h).toFixed(2)}`).join('    ');

  if (active.length >= 2) {                                                  // two live hands
    const spread = twoHandSpread(active[0].h, active[1].h);
    const ang = twoHandAngle(active[0].h, active[1].h);                      // twist both hands (like a wheel) => roll
    if (prevAngle != null) { let d = ang - prevAngle; d = Math.atan2(Math.sin(d), Math.cos(d)); target.rz += d * MIRROR_X; }
    prevAngle = ang;                                                         // unwrapped so it never jumps at ±π
    prevCenter = null; idle = false; clearHighlight();
    if (active[0].pose === 'pinch' && active[1].pose === 'pinch') {          // two pinches => zoom (+ roll)
      if (prevSpread != null) target.scale = clamp(target.scale + (spread - prevSpread) * ZOOM_SPEED, SCALE_MIN, SCALE_MAX);
      prevSpread = spread;
      stateEl.textContent = 'Zoom + roll'; setMode('zoom');
    } else {                                                                 // else => explode by gap (+ roll)
      target.explode = clamp((spread - EXPLODE_MIN) / (EXPLODE_MAX - EXPLODE_MIN), 0, 1);
      prevSpread = null;
      stateEl.textContent = target.explode > 0.05 ? `Exploded ${Math.round(target.explode * 100)}%` : 'Spread hands to explode · twist to roll';
      setMode('explode');
    }
  } else if (active.length === 1 && active[0].pose === 'pinch') {            // one pinch => rotate
    const c = handCenter(active[0].h);
    if (prevCenter) {
      const dry = (c.x - prevCenter.x) * ROT_SPEED * MIRROR_X;
      const drx = (c.y - prevCenter.y) * ROT_SPEED;
      target.ry += dry; target.rx += drx;
      spinVel.ry = spinVel.ry * (1 - SPIN_CAPTURE) + dry * SPIN_CAPTURE;    // recent-weighted flick speed
      spinVel.rx = spinVel.rx * (1 - SPIN_CAPTURE) + drx * SPIN_CAPTURE;
    }
    prevCenter = c; prevSpread = null; prevAngle = null; idle = false; dragging = true; clearHighlight();
    stateEl.textContent = `Rotate (pinch ${pinchStrength(active[0].h).toFixed(2)})`; setMode('rotate');
  } else if (active.length === 1 && active[0].pose === 'point') {            // one point => inspect
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false;
    const name = pointAt(active[0].h);
    stateEl.textContent = name ? `▶ ${name}` : 'Point at a part'; setMode('inspect');
  } else {                                                                   // nothing live => drift
    prevCenter = null; prevSpread = null; prevAngle = null; idle = true; clearHighlight();
    stateEl.textContent = parked >= 2 ? '✊ paused'
      : parked ? '✊ parked — other hand is free'
      : hands.length ? 'Pinch = rotate · point = inspect · ✊ = park'
      : 'Show a hand';
    setMode(parked >= 2 ? 'pause' : parked ? 'park' : 'idle');
  }
  updateStats(hands.length);
}

function drawHands(hands) {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const lm of hands) {
    drawUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: '#7CFFB2', lineWidth: 3 });
    drawUtils.drawLandmarks(lm, { color: '#8ab4ff', radius: 3 });
  }
}

function loop() {
  requestAnimationFrame(loop);
  if (running && handLandmarker && video.currentTime !== lastVideoTime && video.videoWidth) {
    lastVideoTime = video.currentTime;
    const hands = handLandmarker.detectForVideo(video, performance.now()).landmarks || [];
    drawHands(hands);
    applyGestures(hands);
    updateHandViz(hands);
  }
  if (!dragging && (spinVel.rx || spinVel.ry)) {   // flick momentum: keep spinning, then ease to rest
    target.ry += spinVel.ry; target.rx += spinVel.rx;
    spinVel.ry *= SPIN_FRICTION; spinVel.rx *= SPIN_FRICTION;
    if (Math.hypot(spinVel.rx, spinVel.ry) < SPIN_MIN) spinVel.rx = spinVel.ry = 0;
  } else if (idle && autoSpin) target.ry += IDLE_SPIN;   // gentle turntable so the hologram feels alive
  current.rx += (target.rx - current.rx) * SMOOTH; // ease toward target every frame
  current.ry += (target.ry - current.ry) * SMOOTH;
  current.rz += (target.rz - current.rz) * SMOOTH;
  current.scale += (target.scale - current.scale) * SMOOTH;
  current.explode += (target.explode - current.explode) * SMOOTH;
  for (const p of parts) p.position.copy(p.userData.home).multiplyScalar(1 + current.explode * EXPLODE_K);
  pivot.rotation.set(current.rx, current.ry, current.rz);
  pivot.scale.setScalar(current.scale);
  for (const r of reticle.children) r.rotateOnWorldAxis(_up, 0.004 * r.userData.dir); // flat spin, projector look
  composer.render();
}
updateStats(0);   // seed the HUD readout (rocket = 6 parts) before the first frame
loop();
