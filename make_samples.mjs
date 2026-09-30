// make_samples.mjs — generate sample models in every format the loader accepts, no deps.
// Run:  node make_samples.mjs   (writes ./models/*, then self-checks and prints a summary)
//
// One shared unit box drives the glTF/GLB robot (6 named nodes reusing the same mesh, each
// with its own translation+scale so explode pulls the limbs out) and the multi-part OBJ.
// The STL is a single octahedron "crystal". FBX is intentionally skipped — see README.
import { writeFileSync, readFileSync, mkdirSync } from 'node:fs';

const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const norm = (a) => { const L = Math.hypot(...a) || 1; return a.map(x => x/L); };

// unit box centered at origin: positions(24×3), normals(24×3), indices(36). CCW winding => outward normals.
function unitBox() {
  const faces = [
    { n:[1,0,0],  u:[0,0,-1], v:[0,1,0] }, { n:[-1,0,0], u:[0,0,1],  v:[0,1,0] },
    { n:[0,1,0],  u:[1,0,0],  v:[0,0,-1] }, { n:[0,-1,0], u:[1,0,0],  v:[0,0,1] },
    { n:[0,0,1],  u:[1,0,0],  v:[0,1,0] }, { n:[0,0,-1], u:[-1,0,0], v:[0,1,0] },
  ];
  const P = [], N = [], I = [];
  faces.forEach((f, fi) => {
    const c = f.n.map(x => x*0.5);
    const corner = (su, sv) => [0,1,2].map(k => c[k] + f.u[k]*0.5*su + f.v[k]*0.5*sv);
    for (const [su, sv] of [[-1,-1],[1,-1],[1,1],[-1,1]]) { P.push(...corner(su, sv)); N.push(...f.n); }
    const b = fi*4; I.push(b, b+1, b+2, b, b+2, b+3);
  });
  return { P, N, I };
}

// robot: torso stays at center; head/arms/legs radiate out so exploding spreads them.
const PARTS = [
  { name: 'Torso', t: [0, 0, 0],     s: [1.0, 1.4, 0.6] },
  { name: 'Head',  t: [0, 1.15, 0],  s: [0.7, 0.7, 0.7] },
  { name: 'ArmL',  t: [-0.85, 0.1, 0], s: [0.35, 1.2, 0.35] },
  { name: 'ArmR',  t: [0.85, 0.1, 0],  s: [0.35, 1.2, 0.35] },
  { name: 'LegL',  t: [-0.32, -1.4, 0], s: [0.4, 1.3, 0.4] },
  { name: 'LegR',  t: [0.32, -1.4, 0],  s: [0.4, 1.3, 0.4] },
];

const box = unitBox();

// ---- shared binary buffer: [positions | normals | indices] ----
const pos = Float32Array.from(box.P), nrm = Float32Array.from(box.N), idx = Uint16Array.from(box.I);
const OFF_N = pos.byteLength, OFF_I = OFF_N + nrm.byteLength, BIN_LEN = OFF_I + idx.byteLength;
const bin = Buffer.alloc(BIN_LEN);
Buffer.from(pos.buffer).copy(bin, 0);
Buffer.from(nrm.buffer).copy(bin, OFF_N);
Buffer.from(idx.buffer).copy(bin, OFF_I);

function gltfJSON(withUri) {
  return {
    asset: { version: '2.0', generator: 'origin make_samples' },
    scene: 0,
    scenes: [{ nodes: PARTS.map((_, i) => i) }],
    nodes: PARTS.map(p => ({ name: p.name, mesh: 0, translation: p.t, scale: p.s })),
    meshes: [{ name: 'unitbox', primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2, material: 0 }] }],
    materials: [{ name: 'holo', pbrMetallicRoughness: { baseColorFactor: [0.05, 0.22, 0.45, 1], metallicFactor: 0.2, roughnessFactor: 0.4 }, emissiveFactor: [0.04, 0.55, 1.0] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 24, type: 'VEC3', min: [-0.5,-0.5,-0.5], max: [0.5,0.5,0.5] },
      { bufferView: 1, componentType: 5126, count: 24, type: 'VEC3' },
      { bufferView: 2, componentType: 5123, count: 36, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0,     byteLength: pos.byteLength, target: 34962 },
      { buffer: 0, byteOffset: OFF_N, byteLength: nrm.byteLength, target: 34962 },
      { buffer: 0, byteOffset: OFF_I, byteLength: idx.byteLength, target: 34963 },
    ],
    buffers: [withUri
      ? { byteLength: BIN_LEN, uri: 'data:application/octet-stream;base64,' + bin.toString('base64') }
      : { byteLength: BIN_LEN }],
  };
}

function toGLB(gltf) {
  const json = Buffer.from(JSON.stringify(gltf), 'utf8');
  const jsonChunk = Buffer.concat([json, Buffer.alloc((4 - json.length % 4) % 4, 0x20)]);
  const binChunk = Buffer.concat([bin, Buffer.alloc((4 - bin.length % 4) % 4, 0x00)]);
  const total = 12 + 8 + jsonChunk.length + 8 + binChunk.length;
  const head = Buffer.alloc(20);
  head.writeUInt32LE(0x46546C67, 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(total, 8);
  head.writeUInt32LE(jsonChunk.length, 12); head.writeUInt32LE(0x4E4F534A, 16); // JSON
  const binHead = Buffer.alloc(8);
  binHead.writeUInt32LE(binChunk.length, 0); binHead.writeUInt32LE(0x004E4942, 4); // BIN\0
  return Buffer.concat([head, jsonChunk, binHead, binChunk]);
}

// multi-object OBJ: bake each part's transform into vertices (OBJ has no node transforms).
function toOBJ() {
  let out = '# Origin sample — robot (6 named parts)\n', base = 0;
  for (const p of PARTS) {
    out += `o ${p.name}\n`;
    for (let i = 0; i < box.P.length; i += 3)
      out += `v ${box.P[i]*p.s[0]+p.t[0]} ${box.P[i+1]*p.s[1]+p.t[1]} ${box.P[i+2]*p.s[2]+p.t[2]}\n`;
    for (let i = 0; i < box.N.length; i += 3) out += `vn ${box.N[i]} ${box.N[i+1]} ${box.N[i+2]}\n`;
    for (let i = 0; i < box.I.length; i += 3) {
      const a = base+box.I[i]+1, b = base+box.I[i+1]+1, c = base+box.I[i+2]+1;
      out += `f ${a}//${a} ${b}//${b} ${c}//${c}\n`;
    }
    base += box.P.length / 3;
  }
  return out;
}

// octahedron "crystal" as ASCII STL, one solid. Winding forced outward per facet.
function toSTL() {
  const V = { A:[1,0,0], B:[-1,0,0], C:[0,1,0], D:[0,-1,0], E:[0,0,1], F:[0,0,-1] };
  const tris = [['C','E','A'],['C','A','F'],['C','F','B'],['C','B','E'],
                ['D','A','E'],['D','F','A'],['D','B','F'],['D','E','B']];
  let out = 'solid crystal\n';
  for (let [a, b, c] of tris) {
    let p = [V[a], V[b], V[c]];
    let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    const cen = [0,1,2].map(k => (p[0][k]+p[1][k]+p[2][k])/3);
    if (dot(n, cen) < 0) { p = [p[0], p[2], p[1]]; n = cross(sub(p[1], p[0]), sub(p[2], p[0])); } // face outward
    n = norm(n);
    out += ` facet normal ${n[0]} ${n[1]} ${n[2]}\n  outer loop\n`;
    for (const q of p) out += `   vertex ${q[0]} ${q[1]} ${q[2]}\n`;
    out += '  endloop\n endfacet\n';
  }
  return out + 'endsolid crystal\n';
}

mkdirSync('models', { recursive: true });
const glbBuf = toGLB(gltfJSON(false));
writeFileSync('models/robot.gltf', JSON.stringify(gltfJSON(true), null, 2));
writeFileSync('models/robot.glb', glbBuf);
writeFileSync('models/robot.obj', toOBJ());
writeFileSync('models/crystal.stl', toSTL());

// ---- self-checks: fail loudly if the geometry or binary packing breaks ----
import assert from 'node:assert';
assert.equal(box.P.length, 72); assert.equal(box.I.length, 36);
for (let i = 0; i < box.N.length; i += 3)                    // every normal is unit-length
  assert.ok(Math.abs(Math.hypot(box.N[i], box.N[i+1], box.N[i+2]) - 1) < 1e-9, 'box normals must be unit');
for (let f = 0; f < 6; f++) {                                // declared normal matches winding (outward)
  const i0 = box.I[f*6]*3, i1 = box.I[f*6+1]*3, i2 = box.I[f*6+2]*3;
  const P = box.P, tn = cross(sub([P[i1],P[i1+1],P[i1+2]],[P[i0],P[i0+1],P[i0+2]]),
                              sub([P[i2],P[i2+1],P[i2+2]],[P[i0],P[i0+1],P[i0+2]]));
  assert.ok(dot(tn, [box.N[i0],box.N[i0+1],box.N[i0+2]]) > 0, `face ${f} winding must match its normal`);
}
assert.equal(glbBuf.readUInt32LE(0), 0x46546C67, 'GLB magic');
assert.equal(glbBuf.readUInt32LE(4), 2, 'GLB version 2');
assert.equal(glbBuf.readUInt32LE(8), glbBuf.length, 'GLB header length == file length');
const jLen = glbBuf.readUInt32LE(12);
assert.equal(glbBuf.readUInt32LE(16), 0x4E4F534A, 'first chunk is JSON');
JSON.parse(glbBuf.slice(20, 20 + jLen).toString('utf8'));    // JSON chunk must parse
assert.equal(glbBuf.readUInt32LE(20 + jLen + 4), 0x004E4942, 'second chunk is BIN');
assert.equal(JSON.parse(readFileSync('models/robot.gltf', 'utf8')).buffers[0].byteLength, BIN_LEN, 'gltf buffer length');
console.log(`✓ wrote models/{robot.gltf, robot.glb (${glbBuf.length}B), robot.obj, crystal.stl} — all checks passed`);
