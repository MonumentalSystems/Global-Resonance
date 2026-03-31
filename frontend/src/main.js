import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API = 'http://localhost:8000/api';
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
    color: 0x1a3355, emissive: 0x0a1122, specular: 0x222244, shininess: 12,
});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);

const tl = new THREE.TextureLoader();
tl.load('https://unpkg.com/three-globe@2.31.1/example/img/earth-blue-marble.jpg', t => {
    earthMat.map = t; earthMat.color.set(0xffffff); earthMat.emissive.set(0x000000); earthMat.needsUpdate = true;
});
tl.load('https://unpkg.com/three-globe@2.31.1/example/img/earth-night.jpg', t => {
    earthMat.emissiveMap = t; earthMat.emissive.set(0x333333); earthMat.needsUpdate = true;
});

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
    if (!img.src.startsWith('http')) img.src = d.images.aia_193 + '?t=' + Date.now();
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
        tip.innerHTML = `<b style="color:#ff6644">M${eq.mag.toFixed(1)}</b> ${eq.place}<br>`+
            `Depth: ${eq.depth?.toFixed(0)||'?'}km | ${ageH.toFixed(1)}h ago<br>`+
            `${eq.ang_dist}deg from subsolar | <b style="color:${eq.zone==='wavefront'?'#ff4444':'#888'}">${eq.zone}</b>`;
    } else {
        tip.style.display = 'none';
    }
});

// ===== Clock =====
setInterval(() => {
    document.getElementById('clock').textContent =
        new Date().toISOString().replace('T',' ').substring(0,19) + ' UTC';
}, 1000);

// ===== Poll =====
async function poll() {
    const [eqs, ss, kp, sw, xrs, sunD, lunar, cr] = await Promise.allSettled([
        fetchJSON('/earthquakes'), fetchJSON('/subsolar'), fetchJSON('/kp'),
        fetchJSON('/solar_wind'), fetchJSON('/xrs'), fetchJSON('/sun'),
        fetchJSON('/lunar'), fetchJSON('/cosmic_rays'),
    ]);
    if (eqs.value) updateEarthquakes(eqs.value);
    if (ss.value) { updateSubsolar(ss.value); updateJellyBall(ss.value); updateTerminator(ss.value); }
    if (kp.value) updKp(kp.value);
    if (sw.value) updSW(sw.value);
    if (xrs.value) updXRS(xrs.value);
    if (sunD.value) updSun(sunD.value);
    if (lunar.value) updLunar(lunar.value);
    if (cr.value) updCR(cr.value);
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
