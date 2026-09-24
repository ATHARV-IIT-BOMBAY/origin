// HoloControl — rotate & zoom a 3D model with webcam hand gestures.
// Flow: webcam -> MediaPipe HandLandmarker -> gesture math (gestures.js) -> three.js transform.
// No LLM, no backend. The hand model is only a *sensor*; the interaction is hand-written geometry.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { HandLandmarker, FilesetResolver, DrawingUtils }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
import { handCenter, isPinching, pinchStrength, twoHandSpread, landmarkToWorld } from './gestures.js';

// ---- Tuning knobs. A webcam is a messy sensor; these are the calibration dials. ----
const ROT_SPEED = 6.0;    // how far a hand move rotates the model
const ZOOM_SPEED = 6.0;   // how strongly two-hand spread scales the model
const SMOOTH = 0.20;      // 0..1 low-pass ease; lower = smoother but laggier
const MIRROR_X = -1;      // flip so moving your hand right rotates the model right
const PINCH_ON = 0.6;     // pinch strength (0..1) needed to start rotating
const SCALE_MIN = 0.3, SCALE_MAX = 4.0;
const HAND_SPAN = 4.2;    // how wide the tracked hand maps into the 3D scene
const HAND_DEPTH = 1.5;   // how strongly landmark depth pushes hand joints in/out
const IDLE_SPIN = 0.0015; // lazy auto-rotate (rad/frame) when you're not controlling it
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const video = $('video'), overlay = $('overlay'), stateEl = $('state'), errEl = $('err');

// ---------- three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: $('three'), antialias: true });
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

const pivot = new THREE.Group(); // we rotate/scale this; the model lives inside it
scene.add(pivot);

function makeDefaultModel() {
  const geo = new THREE.TorusKnotGeometry(0.7, 0.24, 220, 32);
  const g = new THREE.Group();
  // translucent lit core so the 3D form still reads as a solid object...
  g.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color: 0x0a2a4a, emissive: 0x0aa0ff, emissiveIntensity: 0.6,
    metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.75,
  })));
  // ...plus a bright wireframe overlay — the part that blooms into the hologram glow.
  g.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.9,
  })));
  return g;
}
let model = makeDefaultModel();
pivot.add(model);

function fitToView(obj) { // center at origin and normalize size so any model frames nicely
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  obj.position.sub(box.getCenter(new THREE.Vector3()));
  obj.scale.multiplyScalar(1.8 / (Math.max(size.x, size.y, size.z) || 1));
}
fitToView(model);

// ---------- hand skeleton drawn INTO the 3D scene (your hand appears inside the hologram) ----------
const MAX_JOINTS = 42;               // 2 hands * 21 landmarks
const MAX_BONES = 2 * 24;            // 2 hands * (HAND_CONNECTIONS is 21, 24 is safe headroom)
const joints = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.045, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0x7CFFB2 }), MAX_JOINTS);
joints.frustumCulled = false; scene.add(joints);
const boneGeo = new THREE.BufferGeometry();
boneGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_BONES * 2 * 3), 3));
const bones = new THREE.LineSegments(boneGeo, new THREE.LineBasicMaterial({ color: 0x7CFFB2, transparent: true, opacity: 0.85 }));
bones.frustumCulled = false; scene.add(bones);
const _m = new THREE.Matrix4(), _hidden = new THREE.Matrix4().makeScale(0, 0, 0);
function hideHands() {
  for (let k = 0; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden);
  joints.instanceMatrix.needsUpdate = true; boneGeo.setDrawRange(0, 0);
}
function updateHandViz(hands) {
  const conns = HandLandmarker.HAND_CONNECTIONS, pos = boneGeo.attributes.position.array;
  let j = 0, v = 0; // joint index, bone-vertex index
  for (const hand of hands) {
    const world = hand.map(p => landmarkToWorld(p, HAND_SPAN, HAND_DEPTH));
    for (const p of world) if (j < MAX_JOINTS) { _m.makeTranslation(p.x, p.y, p.z); joints.setMatrixAt(j++, _m); }
    for (const c of conns) if (v + 2 <= MAX_BONES * 2) {
      const a = world[c.start], d = world[c.end];
      pos.set([a.x, a.y, a.z, d.x, d.y, d.z], v * 3); v += 2;
    }
  }
  for (let k = j; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden); // hide unused instances
  joints.instanceMatrix.needsUpdate = true;
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
const loader = new GLTFLoader();
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  loader.load(url, (gltf) => {
    pivot.remove(model);
    model = gltf.scene; fitToView(model); pivot.add(model);
    URL.revokeObjectURL(url);
  }, undefined, (err) => { errEl.textContent = 'Could not load model: ' + err; });
});

// ---------- transform state: gestures set `target`, each frame eases `current` toward it ----------
const target = { rx: 0, ry: 0, scale: 1 };
const current = { rx: 0, ry: 0, scale: 1 };
$('reset').addEventListener('click', () => { target.rx = 0; target.ry = 0; target.scale = 1; });

// ---------- MediaPipe hand tracking ----------
let handLandmarker = null, drawUtils = null, running = false, lastVideoTime = -1;
let prevCenter = null, prevSpread = null, idle = true;

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
    errEl.textContent = '';
    if (!handLandmarker) { stateEl.textContent = 'Loading model…'; await initHands(); }
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    await video.play();
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    running = true; stateEl.textContent = 'Show a hand';
  } catch (err) { errEl.textContent = 'Camera/model error: ' + err; }
}
$('start').addEventListener('click', startCamera);

function applyGestures(hands) {
  if (hands.length >= 2) {                       // two hands => zoom by their separation
    const spread = twoHandSpread(hands[0], hands[1]);
    if (prevSpread != null) target.scale = clamp(target.scale + (spread - prevSpread) * ZOOM_SPEED, SCALE_MIN, SCALE_MAX);
    prevSpread = spread; prevCenter = null; idle = false;
    stateEl.textContent = 'Zooming';
  } else if (hands.length === 1 && isPinching(hands[0], PINCH_ON)) { // one pinch => rotate by its motion
    const c = handCenter(hands[0]);
    if (prevCenter) {
      target.ry += (c.x - prevCenter.x) * ROT_SPEED * MIRROR_X;
      target.rx += (c.y - prevCenter.y) * ROT_SPEED;
    }
    prevCenter = c; prevSpread = null; idle = false;
    stateEl.textContent = `Rotating (pinch ${pinchStrength(hands[0]).toFixed(2)})`;
  } else {                                        // idle => release the anchors, let it drift
    prevCenter = null; prevSpread = null; idle = true;
    stateEl.textContent = hands.length ? 'Pinch to rotate' : 'Show a hand';
  }
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
  if (idle) target.ry += IDLE_SPIN;                // gentle drift so the hologram feels alive
  current.rx += (target.rx - current.rx) * SMOOTH; // ease toward target every frame
  current.ry += (target.ry - current.ry) * SMOOTH;
  current.scale += (target.scale - current.scale) * SMOOTH;
  pivot.rotation.set(current.rx, current.ry, 0);
  pivot.scale.setScalar(current.scale);
  composer.render();
}
loop();
