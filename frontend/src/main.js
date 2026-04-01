import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API = window.location.port === '8001' ? '/api' : 'http://localhost:8001/api';
const R = 1; // earth radius
const POLL = 30_000; // 30s poll

// ===== Renderer =====
const box = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
const camera = new THREE.PerspectiveCamera(45, box.clientWidth / box.clientHeight, 0.01, 100);
camera.position.set(0, 1.2, 2.8);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(box.clientWidth, box.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
box.appendChild(renderer.domElement);
const ctrl = new OrbitControls(camera, renderer.domElement);
ctrl.enableDamping = true;
ctrl.dampingFactor = 0.06;
ctrl.minDistance = 1.15;
ctrl.maxDistance = 6;

// ===== Lighting =====
scene.add(new THREE.AmbientLight(0x334466, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(5, 2, 5);
scene.add(sun);

// Stars
const starGeo = new THREE.BufferGeometry();
const sv = new Float32Array(4000 * 3);
for (let i = 0; i < sv.length; i++) sv[i] = (Math.random() - 0.5) * 60;
starGeo.setAttribute('position', new THREE.BufferAttribute(sv, 3));
scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x667788, size: 0.015 })));

// ===== Globe =====
const earthGeo = new THREE.SphereGeometry(R, 128, 128);
const earthMat = new THREE.MeshPhongMaterial({
    color: 0x2244aa, emissive: 0x112244, specular: 0x222244, shininess: 12,
});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);

// Add a wireframe so the globe is visible even without textures
const wireGeo = new THREE.SphereGeometry(R * 1.001, 36, 18);
const wireMat = new THREE.MeshBasicMaterial({ color: 0x334466, wireframe: true, transparent: true, opacity: 0.15 });
const wireframe = new THREE.Mesh(wireGeo, wireMat);
scene.add(wireframe);

const tl = new THREE.TextureLoader();
tl.crossOrigin = 'anonymous';
tl.load('https://unpkg.com/three-globe@2.31.1/example/img/earth-blue-marble.jpg',
    t => {
        earthMat.map = t; earthMat.color.set(0xffffff); earthMat.emissive.set(0x000000);
        earthMat.needsUpdate = true;
        wireframe.visible = false; // hide wireframe once texture loads
        console.log('Earth texture loaded');
    },
    undefined,
    e => console.warn('Earth texture failed:', e)
);
tl.load('https://unpkg.com/three-globe@2.31.1/example/img/earth-night.jpg',
    t => { earthMat.emissiveMap = t; earthMat.emissive.set(0x333333); earthMat.needsUpdate = true; },
    undefined,
    e => console.warn('Night texture failed:', e)
);

// Atmosphere
const atmosMat = new THREE.ShaderMaterial({
    vertexShader: `
        varying vec3 vNormal;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
    fragmentShader: `
        varying vec3 vNormal;
        void main() {
            float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.5);
            gl_FragColor = vec4(0.3, 0.6, 1.0, intensity * 0.4);
        }`,
    transparent: true, side: THREE.BackSide, depthWrite: false,
});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R * 1.025, 64, 64), atmosMat));

// ===== Coordinate helpers =====
function ll2v(lat, lon, r = R * 1.001) {
    const p = (90 - lat) * Math.PI / 180;
    const t = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(-r * Math.sin(p) * Math.cos(t), r * Math.cos(p), r * Math.sin(p) * Math.sin(t));
}

function greatCirclePoints(lat1, lon1, radiusDeg, nPts = 120) {
    const pts = [];
    const slat = lat1 * Math.PI / 180, slon = lon1 * Math.PI / 180;
    const rd = radiusDeg * Math.PI / 180;
    for (let i = 0; i <= nPts; i++) {
        const a = (i / nPts) * 2 * Math.PI;
        const lat2 = Math.asin(Math.sin(slat) * Math.cos(rd) + Math.cos(slat) * Math.sin(rd) * Math.cos(a));
        const lon2 = slon + Math.atan2(Math.sin(a) * Math.sin(rd) * Math.cos(slat), Math.cos(rd) - Math.sin(slat) * Math.sin(lat2));
        pts.push(ll2v(lat2 * 180 / Math.PI, lon2 * 180 / Math.PI, R * 1.002));
    }
    return pts;
}

// ===== Layer system =====
const layerGroups = {};
const rotatingLayers = new Set();

function getLayer(name) {
    if (!layerGroups[name]) {
        layerGroups[name] = new THREE.Group();
        scene.add(layerGroups[name]);
        rotatingLayers.add(name);
    }
    return layerGroups[name];
}

function clearLayer(name) {
    const g = layerGroups[name];
    if (!g) return;
    while (g.children.length) {
        const c = g.children[0];
        if (c.geometry) c.geometry.dispose();
        if (c.material) {
            if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
            else c.material.dispose();
        }
        g.remove(c);
    }
}

// ===== EARTHQUAKE LAYER =====
// Color: recency (white/red = recent, blue/dim = old)
// Size: animated ring = wave propagation
// Height: inverse depth (shallow = tall, deep = flat)

const eqWaves = []; // active wave animations

function recencyColor(ageHours) {
    // 0h = white-hot, 6h = red, 24h = orange, 72h = dim blue
    if (ageHours < 1) return new THREE.Color(1, 1, 1);
    if (ageHours < 6) return new THREE.Color(1, 0.3 + 0.7 * (1 - ageHours / 6), 0.1);
    if (ageHours < 24) return new THREE.Color(1, 0.3 * (1 - (ageHours - 6) / 18), 0);
    if (ageHours < 48) return new THREE.Color(0.5, 0.2, 0.5);
    return new THREE.Color(0.2, 0.2, 0.5);
}

function depthToHeight(depth) {
    // Height IS depth: deep earthquakes get tall spikes
    // 700km deep = 0.07 above surface (tall spike)
    // 10km deep = 0.001 above surface (barely visible nub)
    // The spike represents how far down the rupture actually is
    const maxH = 0.07;
    return maxH * Math.min((depth || 10) / 700, 1);
}

function magToWaveSpeed(mag) {
    // Larger magnitude = faster/wider wave propagation
    // M5 = slow, M7+ = fast
    return 0.3 + (mag - 4) * 0.15;
}

function magToMaxRadius(mag) {
    // Wave expands to this angular radius (degrees) then fades
    return 2 + Math.pow(mag - 4, 2) * 1.5;
}

function updateEarthquakes(data) {
    clearLayer('earthquakes');
    eqWaves.length = 0;
    if (!data?.earthquakes) return;

    const now = Date.now();
    const layer = getLayer('earthquakes');

    data.earthquakes.forEach(eq => {
        const ageH = (now - eq.time) / 3600000;
        const col = recencyColor(ageH);
        const h = depthToHeight(eq.depth || 33);
        const baseSize = Math.max(0.003, Math.pow(eq.mag - 3.5, 1.3) * 0.003);

        // Spike: height above globe = inverse depth
        const pos = ll2v(eq.lat, eq.lon, R + h);

        // Core marker (spike tip)
        const coreGeo = new THREE.SphereGeometry(baseSize, 8, 8);
        const coreMat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95 });
        const core = new THREE.Mesh(coreGeo, coreMat);
        core.position.copy(pos);
        core.userData = eq;
        layer.add(core);

        // Stem connecting to surface (shows depth visually)
        if (h > 0.005) {
            const surfPos = ll2v(eq.lat, eq.lon, R * 1.001);
            const stemGeo = new THREE.BufferGeometry().setFromPoints([surfPos, pos]);
            const stemMat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.4 });
            layer.add(new THREE.Line(stemGeo, stemMat));
        }

        // Wave ring (for events < 24h old)
        if (ageH < 24) {
            const maxRad = magToMaxRadius(eq.mag);
            // Current wave radius based on age (propagation)
            const waveSpeed = magToWaveSpeed(eq.mag); // degrees per hour
            const currentRad = Math.min(ageH * waveSpeed, maxRad);
            const opacity = Math.max(0, 0.5 * (1 - currentRad / maxRad));

            if (currentRad > 0.2 && opacity > 0.02) {
                const ringPts = greatCirclePoints(eq.lat, eq.lon, currentRad, 60);
                const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPts);
                const ringMat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity });
                layer.add(new THREE.Line(ringGeo, ringMat));
            }

            // For very recent events (< 2h), add animated expanding wave
            if (ageH < 2) {
                eqWaves.push({
                    lat: eq.lat, lon: eq.lon, mag: eq.mag,
                    startTime: eq.time, color: col.clone(),
                    maxRad, waveSpeed,
                });
            }
        }

        // Glow halo for M6+
        if (eq.mag >= 6.0) {
            const glowGeo = new THREE.SphereGeometry(baseSize * 4, 12, 12);
            const glowMat = new THREE.MeshBasicMaterial({
                color: col, transparent: true, opacity: 0.08 + (eq.mag - 6) * 0.03,
            });
            const glow = new THREE.Mesh(glowGeo, glowMat);
            glow.position.copy(pos);
            layer.add(glow);
        }
    });

    document.getElementById('st-eqs').textContent = data.earthquakes.length;
}

// Animated wave rings (updated per frame)
const waveLayer = getLayer('eq-waves');

function animateWaves() {
    clearLayer('eq-waves');
    const now = Date.now();
    const layer = getLayer('eq-waves');

    for (const w of eqWaves) {
        const ageH = (now - w.startTime) / 3600000;
        const rad = ageH * w.waveSpeed;
        if (rad > w.maxRad || rad < 0.1) continue;

        const opacity = 0.6 * (1 - rad / w.maxRad);
        const pts = greatCirclePoints(w.lat, w.lon, rad, 80);
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({
            color: w.color, transparent: true, opacity: Math.max(0, opacity), linewidth: 2,
        });
        layer.add(new THREE.Line(geo, mat));

        // Second ring (P-wave vs S-wave analogy)
        const rad2 = rad * 0.6;
        if (rad2 > 0.1) {
            const pts2 = greatCirclePoints(w.lat, w.lon, rad2, 80);
            const geo2 = new THREE.BufferGeometry().setFromPoints(pts2);
            const mat2 = new THREE.LineBasicMaterial({
                color: w.color, transparent: true, opacity: opacity * 0.4,
            });
            layer.add(new THREE.Line(geo2, mat2));
        }
    }
}

// ===== JELLY BALL ZONES =====
function updateJellyBall(data) {
    clearLayer('jelly-ball');
    if (!data?.zones) return;
    const layer = getLayer('jelly-ball');

    data.zones.forEach(zone => {
        const col = parseInt(zone.color.replace('#', ''), 16);
        const pts = greatCirclePoints(data.lat, data.lon, zone.radius_deg);
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.35 });
        layer.add(new THREE.Line(geo, mat));
    });
}

// ===== SUBSOLAR POINT =====
function updateSubsolar(data) {
    clearLayer('subsolar');
    if (!data) return;
    const layer = getLayer('subsolar');

    // Pulsing marker
    const geo = new THREE.SphereGeometry(0.015, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    const m = new THREE.Mesh(geo, mat);
    m.position.copy(ll2v(data.lat, data.lon, R * 1.008));
    m.name = 'subsolar-pulse';
    layer.add(m);

    // Vertical beam
    const beamPts = [ll2v(data.lat, data.lon, R * 1.008), ll2v(data.lat, data.lon, R * 1.15)];
    const beamGeo = new THREE.BufferGeometry().setFromPoints(beamPts);
    const beamMat = new THREE.LineBasicMaterial({ color: 0xffff00, transparent: true, opacity: 0.3 });
    layer.add(new THREE.Line(beamGeo, beamMat));

    // Antipodal
    const antiLon = data.lon > 0 ? data.lon - 180 : data.lon + 180;
    const antiGeo = new THREE.SphereGeometry(0.008, 12, 12);
    const antiMat = new THREE.MeshBasicMaterial({ color: 0xcc88cc, transparent: true, opacity: 0.5 });
    const anti = new THREE.Mesh(antiGeo, antiMat);
    anti.position.copy(ll2v(-data.lat, antiLon, R * 1.005));
    layer.add(anti);

    // Update sun light
    sun.position.copy(ll2v(data.lat, data.lon, 5));
}

// ===== TERMINATOR =====
function updateTerminator(data) {
    clearLayer('terminator');
    if (!data) return;
    const pts = greatCirclePoints(data.lat, data.lon, 90, 180);
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: 0xff6600, transparent: true, opacity: 0.2 });
    getLayer('terminator').add(new THREE.Line(geo, mat));
}

// ===== PLATE BOUNDARIES =====
async function loadPlates() {
    try {
        const resp = await fetch('src/plates.json');
        const segs = await resp.json();
        const mat = new THREE.LineBasicMaterial({ color: 0x445566, transparent: true, opacity: 0.35 });
        const layer = getLayer('plates');

        for (const seg of segs) {
            if (seg.length < 2) continue;
            const pts = [];
            for (let i = 0; i < seg.length; i++) {
                if (i > 0 && Math.abs(seg[i][0] - seg[i-1][0]) > 90) {
                    if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
                    pts.length = 0;
                }
                pts.push(ll2v(seg[i][1], seg[i][0], R * 1.0005));
            }
            if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
        }
    } catch (e) { console.warn('Plates:', e.message); }
}
loadPlates();

// ===== SIDEBAR DATA =====
async function fetchJSON(ep) {
    try { return await (await fetch(`${API}${ep}`)).json(); }
    catch (e) { return null; }
}

function drawChart(id, data, opts = {}) {
    const c = document.getElementById(id);
    if (!c || !data?.length) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.clientWidth * 2, h = c.height = c.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);

    const vals = data.map(d => d.value).filter(v => v != null && isFinite(v));
    if (vals.length < 2) return;
    const mn = opts.min ?? Math.min(...vals), mx = opts.max ?? Math.max(...vals);
    const rng = mx - mn || 1;

    // Grid
    ctx.strokeStyle = '#181833'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) { const y = (i/4)*h; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

    // Zero line
    if (mn < 0 && mx > 0) {
        const zy = h - ((0-mn)/rng)*h;
        ctx.strokeStyle = '#333366'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0,zy); ctx.lineTo(w,zy); ctx.stroke();
    }

    // Negative fill
    if (opts.fillNeg) {
        const zy = h - ((0-mn)/rng)*h;
        ctx.fillStyle = 'rgba(255,50,50,0.12)';
        ctx.beginPath();
        vals.forEach((v, i) => {
            const x = (i/(vals.length-1))*w, y = h-((v-mn)/rng)*h;
            i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        });
        ctx.lineTo(w,zy); ctx.lineTo(0,zy); ctx.closePath(); ctx.fill();
    }

    // Line
    ctx.strokeStyle = opts.color || '#00ccff'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    vals.forEach((v, i) => {
        const x = (i/(vals.length-1))*w, y = Math.max(0, Math.min(h, h-((v-mn)/rng)*h));
        i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();

    // Value label
    ctx.fillStyle = opts.color || '#00ccff';
    ctx.font = 'bold 20px monospace'; ctx.textAlign = 'right';
    ctx.fillText(vals[vals.length-1].toFixed(opts.dec ?? 1), w-4, 22);
}

function drawXRS(data) {
    const c = document.getElementById('xrs-chart');
    if (!c || !data?.xrs?.length) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.clientWidth * 2, h = c.height = c.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);
    const vals = data.xrs.map(e => Math.log10(e.flux));
    const mn = -8, mx = -3, rng = mx - mn;

    [[-4,'X','#ff4444'],[-5,'M','#ffaa44'],[-6,'C','#4488ff'],[-7,'B','#222244']].forEach(([lv,lb,co]) => {
        const y = h-((lv-mn)/rng)*h;
        ctx.strokeStyle = co; ctx.lineWidth = 0.5; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = co; ctx.font = '14px monospace'; ctx.fillText(lb, 4, y-2);
    });

    ctx.strokeStyle = '#ff8844'; ctx.lineWidth = 1.5; ctx.beginPath();
    vals.forEach((v,i) => {
        const x = (i/(vals.length-1))*w, y = Math.max(0, Math.min(h, h-((v-mn)/rng)*h));
        i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
}

// ===== Status updaters =====
function updKp(d) {
    if (!d?.current) return;
    const k = d.current, el = document.getElementById('kp-metric');
    el.textContent = `Kp ${k.toFixed(0)}`;
    el.className = 'metric ' + (k<4?'quiet':k<6?'active':'storm');
    const s = document.getElementById('st-kp');
    s.textContent = k.toFixed(0); s.className = 'value ' + (k<4?'ok':k<6?'':'warn');
}

function updSW(d) {
    if (!d) return;
    if (d.current_bz != null) {
        const e = document.getElementById('st-bz');
        e.textContent = `${d.current_bz.toFixed(1)} nT`;
        e.className = 'value '+(d.current_bz < -10 ? 'warn':'ok');
    }
    if (d.current_speed != null) {
        const e = document.getElementById('st-vsw');
        e.textContent = `${d.current_speed.toFixed(0)} km/s`;
        e.className = 'value '+(d.current_speed > 600 ? 'warn':'ok');
    }
    drawChart('sw-chart', d.bz, { color: '#ff6666', fillNeg: true, dec: 1 });
}

function updXRS(d) {
    if (!d) return;
    if (d.current_flux) {
        const f = d.current_flux;
        const cl = f>=1e-4?`X${(f/1e-4).toFixed(1)}`:f>=1e-5?`M${(f/1e-5).toFixed(1)}`:f>=1e-6?`C${(f/1e-6).toFixed(1)}`:'B';
        document.getElementById('st-xrs').textContent = cl;
    }
    drawXRS(d);
    const el = document.getElementById('op-state');
    el.textContent = d.state || '?';
    el.className = 'state '+(d.state==='FALLING'?'falling':d.state==='RISING'?'rising':'stable');
}

function updSun(d) {
    if (!d?.images) return;
    window._si = d.images;
    const img = document.getElementById('sun-image');
    if (!img.dataset.loaded) {
        img.src = d.images.eit_195 || d.images.aia_193 || Object.values(d.images)[0];
        img.dataset.loaded = '1';
    }
}

function updLunar(d) {
    if (!d) return;
    document.getElementById('lunar-metric').textContent = `${d.name} (${d.illumination}%)`;
    document.getElementById('lunar-detail').textContent =
        `Force: ${d.tidal_force.toFixed(3)} | dF/dt: ${d.tidal_rate.toFixed(3)} | Full: ${d.days_to_full}d`;
    document.getElementById('st-moon').textContent = `${d.illumination}%`;
}

function updCR(d) {
    if (!d?.stations) return;
    const ks = Object.keys(d.stations);
    if (!ks.length) return;
    const avg = ks.reduce((s,k) => s + d.stations[k].deviation_pct, 0) / ks.length;
    const el = document.getElementById('cr-metric');
    el.textContent = `${avg>0?'+':''}${avg.toFixed(1)}%`;
    el.className = 'metric '+(d.forbush_detected?'storm':'quiet');
    document.getElementById('cr-detail').textContent =
        d.forbush_detected ? 'FORBUSH DECREASE' : `${ks.length} stations nominal`;
}

// ===== Sun image selector =====
document.querySelectorAll('#sun-selector button').forEach(b => {
    b.addEventListener('click', () => {
        document.querySelectorAll('#sun-selector button').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        if (window._si?.[b.dataset.img])
            document.getElementById('sun-image').src = window._si[b.dataset.img] + '?t=' + Date.now();
    });
});

// ===== MAGNETOMETER STATIONS =====
function updateMagnetometers(data) {
    clearLayer('magnetometers');
    if (!data?.stations) return;
    const layer = getLayer('magnetometers');

    data.stations.forEach(st => {
        const pos = ll2v(st.lat, st.lon, R * 1.004);
        // Diamond marker
        const geo = new THREE.OctahedronGeometry(0.008, 0);
        const mat = new THREE.MeshBasicMaterial({
            color: st.network === 'USGS' ? 0xcc44cc : 0x44cccc,
            transparent: true, opacity: 0.8,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(pos);
        mesh.userData = { type: 'magnetometer', ...st };
        layer.add(mesh);

        // Small label (only visible up close)
        // We'll rely on tooltip for now
    });
}

// ===== DST UPDATE =====
function updDst(data) {
    if (!data) return;
    const el = document.getElementById('dst-metric');
    const st = document.getElementById('st-dst');
    if (data.current != null) {
        el.textContent = `${data.current} nT`;
        el.className = 'm ' + (data.current > -30 ? 'q' : data.current > -50 ? 'a' : 's');
        st.textContent = `${data.current} nT`;
        st.className = 'v ' + (data.current > -30 ? 'g' : data.current > -50 ? '' : 'w');
    }
}

// ===== Layer toggles =====
document.querySelectorAll('.layer-toggle input').forEach(inp => {
    inp.addEventListener('change', () => {
        const g = layerGroups[inp.dataset.layer];
        if (g) g.visible = inp.checked;
    });
});
// Apply initial states
document.querySelectorAll('.layer-toggle input').forEach(inp => {
    if (!inp.checked && layerGroups[inp.dataset.layer]) layerGroups[inp.dataset.layer].visible = false;
});

// ===== Tooltip =====
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tip = document.createElement('div');
tip.style.cssText = 'position:fixed;background:rgba(5,5,16,0.95);color:#ccc;font:11px monospace;padding:6px 10px;border:1px solid #00ccff;border-radius:4px;pointer-events:none;display:none;z-index:1000;max-width:260px;';
document.body.appendChild(tip);

box.addEventListener('mousemove', e => {
    const r = box.getBoundingClientRect();
    mouse.x = ((e.clientX-r.left)/r.width)*2-1;
    mouse.y = -((e.clientY-r.top)/r.height)*2+1;
    ray.setFromCamera(mouse, camera);
    const eqLayer = layerGroups['earthquakes'];
    if (!eqLayer) { tip.style.display='none'; return; }
    const hits = ray.intersectObjects(eqLayer.children);
    const hit = hits.find(h => h.object.userData?.mag);
    if (hit) {
        const eq = hit.object.userData;
        const ageH = (Date.now() - eq.time) / 3600000;
        tip.style.display = 'block';
        tip.style.left = (e.clientX+14)+'px';
        tip.style.top = (e.clientY-10)+'px';
        const zoneColors = { eye:'#44f', inner:'#4f4', wavefront:'#f44', outer:'#ff4', far:'#888', antipodal:'#c8c' };
        tip.innerHTML = `<b style="color:#ff6644">M${eq.mag.toFixed(1)}</b> ${eq.place}<br>`+
            `Depth: ${eq.depth?.toFixed(0)||'?'}km | ${ageH.toFixed(1)}h ago<br>`+
            `${eq.ang_dist}deg | <span style="color:${zoneColors[eq.zone]||'#888'}">${eq.zone}</span>`;
        box.style.cursor = 'pointer';
    } else {
        tip.style.display = 'none';
        box.style.cursor = 'grab';
    }
});

// Click to inspect — checks earthquakes and magnetometers
box.addEventListener('click', e => {
    const r = box.getBoundingClientRect();
    mouse.x = ((e.clientX-r.left)/r.width)*2-1;
    mouse.y = -((e.clientY-r.top)/r.height)*2+1;
    ray.setFromCamera(mouse, camera);

    // Check earthquakes first
    const eqLayer = layerGroups['earthquakes'];
    if (eqLayer) {
        const hits = ray.intersectObjects(eqLayer.children);
        const hit = hits.find(h => h.object.userData?.mag);
        if (hit) { showDetail(hit.object.userData); return; }
    }

    // Check magnetometers
    const magLayer = layerGroups['magnetometers'];
    if (magLayer?.visible) {
        const hits = ray.intersectObjects(magLayer.children);
        const hit = hits.find(h => h.object.userData?.type === 'magnetometer');
        if (hit) { showMagDetail(hit.object.userData); return; }
    }

    // Click on empty space closes the panel
    document.getElementById('detail').style.display = 'none';
});

function showDetail(eq) {
    const panel = document.getElementById('detail');
    const content = document.getElementById('detail-content');
    const ageH = (Date.now() - eq.time) / 3600000;
    const dt = new Date(eq.time);
    const zoneColors = { eye:'#44f', inner:'#4f4', wavefront:'#f44', outer:'#ff4', far:'#888', antipodal:'#c8c' };
    const zoneRatios = { eye:'0.85x', inner:'1.05x', wavefront:'1.36x', outer:'1.10x', far:'1.05x', antipodal:'1.16x' };

    content.innerHTML = `
        <h3>M${eq.mag.toFixed(1)} ${eq.place || 'Unknown'}</h3>
        <div class="row"><span class="k">Time</span><span class="val">${dt.toISOString().replace('T',' ').substring(0,19)} UTC</span></div>
        <div class="row"><span class="k">Age</span><span class="val">${ageH < 1 ? (ageH*60).toFixed(0)+' min' : ageH.toFixed(1)+' hours'} ago</span></div>
        <div class="row"><span class="k">Location</span><span class="val">${eq.lat.toFixed(3)}N, ${eq.lon.toFixed(3)}E</span></div>
        <div class="row"><span class="k">Depth</span><span class="val">${eq.depth?.toFixed(1) || '?'} km</span></div>
        <div class="row"><span class="k">Subsolar dist</span><span class="val">${eq.ang_dist} deg</span></div>
        <div class="row"><span class="k">Jelly Ball zone</span>
            <span class="zone-badge" style="background:${zoneColors[eq.zone]||'#444'};color:#fff">${eq.zone} (${zoneRatios[eq.zone]||'?'})</span>
        </div>
        <div style="margin-top:8px; border-top:1px solid #222; padding-top:6px;">
            <a href="https://earthquake.usgs.gov/earthquakes/eventpage/${eq.id || ''}" target="_blank">USGS Event Page &rarr;</a>
        </div>
    `;
    panel.style.display = 'block';
}

function showMagDetail(st) {
    const panel = document.getElementById('detail');
    const content = document.getElementById('detail-content');
    content.innerHTML = `
        <h3 style="color:#cc44cc">${st.code} - ${st.name}</h3>
        <div class="row"><span class="k">Network</span><span class="val">${st.network}</span></div>
        <div class="row"><span class="k">Location</span><span class="val">${st.lat.toFixed(2)}N, ${st.lon.toFixed(2)}E</span></div>
        ${st.live ? `
        <div style="margin-top:6px; border-top:1px solid #222; padding-top:6px;">
            <div class="row"><span class="k">B_X</span><span class="val">${st.live.X?.toFixed(1) || '?'} nT</span></div>
            <div class="row"><span class="k">B_Y</span><span class="val">${st.live.Y?.toFixed(1) || '?'} nT</span></div>
            <div class="row"><span class="k">B_Z</span><span class="val">${st.live.Z?.toFixed(1) || '?'} nT</span></div>
        </div>` : '<div class="d" style="margin-top:6px">No live data (archival only)</div>'}
        <div style="margin-top:6px;">
            <a href="https://geomag.usgs.gov/ws/data/?id=${st.code}" target="_blank" style="color:#0cf;font-size:9px">USGS Data Service &rarr;</a>
        </div>
    `;
    panel.style.display = 'block';
}

// ===== PALEOMAG DEEP TIME =====
let palemagData = null;

async function loadPaleomag() {
    palemagData = await fetchJSON('/paleomag');
    if (palemagData?.sites) drawPaleomagChart();
}

function drawPaleomagChart() {
    const c = document.getElementById('paleomag-chart');
    if (!c || !palemagData?.sites) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.clientWidth * 2;
    const h = c.height = c.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);

    const times = palemagData.times; // years BCE (positive = BCE)
    const tMin = -500, tMax = 2500;
    const fMin = 30, fMax = 80;

    // Grid
    ctx.strokeStyle = '#181833'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = (i/4) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // 1200 BCE marker
    const x1200 = ((1200 - tMin) / (tMax - tMin)) * w;
    ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x1200, 0); ctx.lineTo(x1200, h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff4444'; ctx.font = '14px monospace';
    ctx.fillText('1200', x1200 + 2, 14);

    // Draw sites
    const drawSite = (name, color, thick) => {
        const vals = palemagData.sites[name]?.values;
        if (!vals) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = thick ? 2.5 : 1;
        ctx.globalAlpha = thick ? 1.0 : 0.5;
        ctx.beginPath();
        for (let i = 0; i < times.length; i++) {
            const x = ((times[i] - tMin) / (tMax - tMin)) * w;
            const y = h - ((vals[i] - fMin) / (fMax - fMin)) * h;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    };

    drawSite('Greece', '#44aaff', false);
    drawSite('Anatolia', '#ffaa44', false);
    drawSite('Egypt', '#44ff44', false);
    drawSite('Levant', '#ff4444', true);
    drawSite('China', '#ffff44', true);

    // Labels
    ctx.font = '12px monospace';
    ctx.fillStyle = '#ff4444'; ctx.fillText('Levant', 4, h - 8);
    ctx.fillStyle = '#ffff44'; ctx.fillText('China', 70, h - 8);
}

document.getElementById('paleomag-toggle')?.addEventListener('change', (e) => {
    const panel = document.getElementById('paleomag-panel');
    if (panel) panel.style.display = e.target.checked ? 'block' : 'none';
    if (e.target.checked && !palemagData) loadPaleomag();
});

// ===== TIME SLIDER (historical replay) =====
const timeSlider = document.getElementById('time-slider');
const timeVal = document.getElementById('time-val');
const timeLive = document.getElementById('time-live');
let isLive = true;
let historyHoursBack = 0;

if (timeSlider) {
    // Show the time control bar
    document.getElementById('time-control').classList.add('visible');

    timeSlider.addEventListener('input', () => {
        historyHoursBack = parseInt(timeSlider.value);
        if (historyHoursBack === 0) {
            isLive = true;
            timeVal.textContent = 'LIVE';
            timeLive.classList.add('on');
        } else {
            isLive = false;
            const d = new Date(Date.now() - historyHoursBack * 3600000);
            timeVal.textContent = `-${historyHoursBack}h`;
            timeLive.classList.remove('on');
        }
    });

    timeLive.addEventListener('click', () => {
        timeSlider.value = 0;
        historyHoursBack = 0;
        isLive = true;
        timeVal.textContent = 'LIVE';
        timeLive.classList.add('on');
    });
}

// ===== Clock =====
setInterval(() => {
    const now = isLive ? new Date() : new Date(Date.now() - historyHoursBack * 3600000);
    document.getElementById('clock').textContent =
        now.toISOString().replace('T',' ').substring(0,19) + ' UTC' + (isLive ? '' : ` (-${historyHoursBack}h)`);
}, 1000);

// ===== Poll =====
async function poll() {
    const results = await Promise.allSettled([
        fetchJSON('/earthquakes'),    // 0
        fetchJSON('/subsolar'),       // 1
        fetchJSON('/kp'),             // 2
        fetchJSON('/solar_wind'),     // 3
        fetchJSON('/xrs'),            // 4
        fetchJSON('/sun'),            // 5
        fetchJSON('/lunar'),          // 6
        fetchJSON('/cosmic_rays'),    // 7
        fetchJSON('/dst'),            // 8
        fetchJSON('/magnetometers'),  // 9
    ]);
    const v = i => results[i]?.value;
    if (v(0)) updateEarthquakes(v(0));
    if (v(1)) { updateSubsolar(v(1)); updateJellyBall(v(1)); updateTerminator(v(1)); }
    if (v(2)) updKp(v(2));
    if (v(3)) updSW(v(3));
    if (v(4)) updXRS(v(4));
    if (v(5)) updSun(v(5));
    if (v(6)) updLunar(v(6));
    if (v(7)) updCR(v(7));
    if (v(8)) updDst(v(8));
    if (v(9)) updateMagnetometers(v(9));
    // Cosmic ray in status bar
    if (v(7)?.stations) {
        const ks = Object.keys(v(7).stations);
        if (ks.length) {
            const avg = ks.reduce((s,k) => s + v(7).stations[k].deviation_pct, 0) / ks.length;
            document.getElementById('st-cr').textContent = `${avg>0?'+':''}${avg.toFixed(1)}%`;
            document.getElementById('st-cr').className = 'v ' + (v(7).forbush_detected ? 'w' : 'g');
        }
    }
}
poll();
setInterval(poll, POLL);

// ===== Animate =====
let frame = 0;
function animate() {
    requestAnimationFrame(animate);
    ctrl.update();
    frame++;

    const rot = 0.00015;
    earth.rotation.y += rot;
    wireframe.rotation.y += rot;
    for (const name of rotatingLayers) {
        if (layerGroups[name]) layerGroups[name].rotation.y += rot;
    }

    // Animate wave propagation every 30 frames
    if (frame % 30 === 0 && eqWaves.length > 0) animateWaves();

    // Pulse subsolar
    const ss = layerGroups['subsolar']?.getObjectByName('subsolar-pulse');
    if (ss) ss.scale.setScalar(1 + 0.2 * Math.sin(frame * 0.04));

    renderer.render(scene, camera);
}
animate();

// ===== Resize =====
window.addEventListener('resize', () => {
    camera.aspect = box.clientWidth / box.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(box.clientWidth, box.clientHeight);
});
