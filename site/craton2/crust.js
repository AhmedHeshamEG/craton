// Craton 2 beta: the hidden layer, live. A 12 x 4 x 8 block of 5 km cubes coloured by stress.
// One loop (14 s): the plates load the fault, it slips, the stress moves to the tips, aftershocks land there, it heals.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const box = document.getElementById('crust');
const R = new THREE.WebGLRenderer({ antialias: true, alpha: true });
R.setPixelRatio(Math.min(2, window.devicePixelRatio));
box.appendChild(R.domElement);
const S = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(34, 2, 0.1, 200);
cam.position.set(15, 11, 17);
const ctl = new OrbitControls(cam, R.domElement);
ctl.target.set(0, 1.6, 0); ctl.enableDamping = true; ctl.enablePan = false; ctl.enableZoom = false;
ctl.minPolarAngle = 0.3; ctl.maxPolarAngle = 1.45; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.6;
S.add(new THREE.HemisphereLight(0xffffff, 0x8a8f96, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(6, 14, 9); S.add(sun);

// stress colour: calm blue-grey -> yellow -> orange -> red
const STOPS = [[0, '#b3c3d4'], [0.45, '#f2d16b'], [0.75, '#ff7a2a'], [1, '#d1261c']].map(([u, c]) => [u, new THREE.Color(c)]);
const ramp = (v, o) => { v = Math.min(1, Math.max(0, v)); for (let i = 1; i < STOPS.length; i++) if (v <= STOPS[i][0]) { const a = STOPS[i - 1], b = STOPS[i]; return o.copy(a[1]).lerp(b[1], (v - a[0]) / (b[0] - a[0])); } return o.copy(STOPS[3][1]); };
const cl = (v) => Math.min(1, Math.max(0, v)), sm = (v) => { v = cl(v); return v * v * (3 - 2 * v); }, lerp = (a, b, u) => a + (b - a) * u;
let seed = 7; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

// the fault: a vertical sheet on the diagonal of the block
const FA = new THREE.Vector3(-4, 0, 3.5), FB = new THREE.Vector3(4, 0, -3.5), FD = FB.clone().sub(FA).normalize(), FL = FA.distanceTo(FB);
const cells = [];
for (let i = 0; i < 12; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 8; k++) {
  const x = -5.5 + i, y = 0.5 + j, z = -3.5 + k, along = x * FD.x + z * FD.z, d = -x * FD.z + z * FD.x;
  const s0 = 0.16 + 0.6 * Math.exp(-(d * d) / 1.8) * [0.6, 1, 0.9, 0.6][j] + (rnd() - 0.5) * 0.1;
  const rup = Math.abs(along) < 2.8 && Math.abs(d) < 1.3 && j > 0, tip = !rup && Math.abs(along) < 4.6 && Math.abs(d) < 1.5 && j > 0, side = !rup && !tip && Math.abs(d) < 3 && Math.abs(along) < 3.2;
  cells.push({ x, y, z, d, s0, rup, tip, side });
}
const cubes = new THREE.InstancedMesh(new THREE.BoxGeometry(0.62, 0.62, 0.62), new THREE.MeshLambertMaterial({ color: 0xffffff }), cells.length);
const M4 = new THREE.Matrix4(), C = new THREE.Color(), WHITE = new THREE.Color(0xffffff);
S.add(cubes);
const fault = new THREE.Mesh(new THREE.PlaneGeometry(FL, 4), new THREE.MeshLambertMaterial({ color: 0xff7a2a, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
fault.position.set(0, 2, 0); fault.rotation.y = Math.atan2(-FD.z, FD.x); S.add(fault);
fault.add(new THREE.LineSegments(new THREE.EdgesGeometry(fault.geometry), new THREE.LineBasicMaterial({ color: 0x17171a })));
const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(12.2, 4.2, 8.2)), new THREE.LineBasicMaterial({ color: 0x86868b, transparent: true, opacity: 0.5 }));
frame.position.set(0, 2, 0); S.add(frame);

// quakes: one main shock in the middle of the fault, aftershocks on the tips
const ballTex = (() => { const c = document.createElement('canvas'); c.width = 128; c.height = 64; const x = c.getContext('2d'); x.fillStyle = '#f6f1e6'; x.fillRect(0, 0, 128, 64); x.fillStyle = '#1e1e24'; x.fillRect(0, 0, 32, 32); x.fillRect(64, 0, 32, 32); x.fillRect(32, 32, 32, 32); x.fillRect(96, 32, 32, 32); const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t; })();
const ball = (r) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), new THREE.MeshLambertMaterial({ map: ballTex })); S.add(m); return m; };
const main = ball(0.45); main.position.set(0, 4.5, 0);
const tips = cells.filter((q) => q.tip && q.y > 3);
const after = Array.from({ length: 8 }, (_, n) => { const q = tips[Math.floor(rnd() * tips.length)], b = ball(0.14 + rnd() * 0.1); b.position.set(q.x + (rnd() - 0.5) * 0.4, 4.25, q.z + (rnd() - 0.5) * 0.4); b.userData.t = 0.66 + n * 0.022; return b; });
const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 64), new THREE.MeshBasicMaterial({ color: 0xd1261c, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
ring.rotation.x = -Math.PI / 2; ring.position.y = 4.05; S.add(ring);

const LOOP = 14, SLIP = 0.55;
const phaseEl = document.getElementById('phase'), dayEl = document.getElementById('day');
function stress(q, u) {
  let s = q.s0 + (1.02 - q.s0) * sm(u / SLIP) * Math.exp(-(q.d * q.d) / 1.4) * (q.y > 1 ? 1 : 0.5);
  if (u >= SLIP) { const post = q.rup ? 0.12 : q.tip ? Math.min(1, s + 0.32) : q.side ? Math.max(0.05, s - 0.28) : s; s = lerp(s, post, sm((u - SLIP - 0.01) / 0.08)); s = lerp(s, q.s0, sm((u - 0.88) / 0.12)); }
  return s;
}
function paint(u) {
  cells.forEach((q, n) => {
    let jig = 0; if (q.rup && u > SLIP && u < SLIP + 0.06) jig = Math.sin((u - SLIP) * 600) * 0.06 * (1 - (u - SLIP) / 0.06);
    M4.makeTranslation(q.x + jig * FD.x, q.y, q.z + jig * FD.z); cubes.setMatrixAt(n, M4);
    ramp(stress(q, u), C); if (q.rup && u >= SLIP) C.lerp(WHITE, cl(1 - (u - SLIP) / 0.04)); cubes.setColorAt(n, C);
  });
  cubes.instanceMatrix.needsUpdate = true; cubes.instanceColor.needsUpdate = true;
  let fc = lerp(0.62, 1, sm(u / SLIP)); if (u >= SLIP) fc = lerp(fc, 0.22, sm((u - SLIP) / 0.1)); fc = lerp(fc, 0.62, sm((u - 0.88) / 0.12)); ramp(fc, fault.material.color);
  const m = u >= SLIP && u < 0.9; main.visible = m; main.scale.setScalar(m ? Math.min(1, (u - SLIP) / 0.02) * (1 - sm((u - 0.85) / 0.05)) + 1e-3 : 1e-3);
  const ru = cl((u - SLIP) / 0.12); ring.visible = ru > 0 && ru < 1; ring.scale.setScalar(0.5 + ru * 7); ring.material.opacity = 0.9 * (1 - ru);
  after.forEach((b) => { const v = u >= b.userData.t && u < 0.9; b.visible = v; b.scale.setScalar(v ? Math.min(1, (u - b.userData.t) / 0.015) : 1e-3); });
  const label = u < SLIP ? 'the plates load the fault' : u < SLIP + 0.05 ? 'it slips' : u < 0.65 ? 'the stress moves to the tips' : u < 0.88 ? 'aftershocks land where it moved' : 'it heals, and loads again';
  if (phaseEl.textContent !== label) phaseEl.textContent = label;
  dayEl.textContent = u < SLIP ? `day ${Math.round(1 + 2400 * sm(u / SLIP)).toLocaleString('en')}` : '';
}

function size() { const w = box.clientWidth, h = box.clientHeight; R.setSize(w, h, false); R.domElement.style.width = '100%'; R.domElement.style.height = '100%'; cam.aspect = w / h; cam.fov = w < 600 ? 46 : 34; cam.updateProjectionMatrix(); }
new ResizeObserver(size).observe(box); size();

const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
let playing = !reduce, t = reduce ? LOOP * 0.7 : 0, last = performance.now(), visible = true;
const btn = document.getElementById('play');
const setBtn = () => { btn.textContent = playing ? '❚❚' : '▶'; btn.setAttribute('aria-label', playing ? 'Pause' : 'Play'); ctl.autoRotate = playing; };
btn.addEventListener('click', () => { playing = !playing; setBtn(); }); setBtn();
new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(box);
function frameLoop(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (visible) { if (playing) t = (t + dt) % LOOP; paint(t / LOOP); ctl.update(); R.render(S, cam); }
  requestAnimationFrame(frameLoop);
}
requestAnimationFrame(frameLoop);
