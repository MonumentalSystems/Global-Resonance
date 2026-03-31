import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API_BASE = 'http://localhost:8000/api';
const EARTH_RADIUS = 1;
const POLL_INTERVAL = 60_000; // 1 minute

// ===== Scene setup =====
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(0, 1.5, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 1.2;
controls.maxDistance = 8;

// ===== Lighting =====
scene.add(new THREE.AmbientLight(0x444466, 0.8));
const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
sunLight.position.set(5, 3, 5);
scene.add(sunLight);

// Starfield background
const starGeom = new THREE.BufferGeometry();
const starVerts = new Float32Array(3000 * 3);
for (let i = 0; i < 3000 * 3; i++) {
    starVerts[i] = (Math.random() - 0.5) * 50;
}
starGeom.setAttribute('position', new THREE.BufferAttribute(starVerts, 3));
scene.add(new THREE.Points(starGeom, new THREE.PointsMaterial({ color: 0x888899, size: 0.02 })));

// ===== Earth globe =====
const earthGeom = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
const textureLoader = new THREE.TextureLoader();
const earthMat = new THREE.MeshPhongMaterial({
    color: 0x1a3355,
    emissive: 0x0a1122,
    specular: 0x333333,
    shininess: 15,
});
const earth = new THREE.Mesh(earthGeom, earthMat);
scene.add(earth);

// Load earth textures
textureLoader.load(
    'https://unpkg.com/three-globe@2.31.1/example/img/earth-blue-marble.jpg',
    (texture) => {
        earthMat.map = texture;
        earthMat.color.set(0xffffff);
        earthMat.emissive.set(0x000000);
        earthMat.needsUpdate = true;
    }
);

// Night side emission texture
textureLoader.load(
    'https://unpkg.com/three-globe@2.31.1/example/img/earth-night.jpg',
    (texture) => {
        earthMat.emissiveMap = texture;
        earthMat.emissive.set(0x222222);
        earthMat.needsUpdate = true;
    }
);

// Atmosphere glow
const atmosGeom = new THREE.SphereGeometry(EARTH_RADIUS * 1.02, 64, 64);
const atmosMat = new THREE.MeshPhongMaterial({
    color: 0x4488ff, transparent: true, opacity: 0.06, side: THREE.BackSide,
});
scene.add(new THREE.Mesh(atmosGeom, atmosMat));

// ===== Layer groups =====
const layers = {
    earthquakes: new THREE.Group(),
    'jelly-ball': new THREE.Group(),
    subsolar: new THREE.Group(),
    plates: new THREE.Group(),
    magnetometers: new THREE.Group(),
    tidal: new THREE.Group(),
    terminator: new THREE.Group(),
};
Object.values(layers).forEach(g => scene.add(g));

// ===== Helpers =====
function latLonToVec3(lat, lon, radius = EARTH_RADIUS * 1.001) {
    const phi = (90 - lat) * Math.PI / 180;
    const theta = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(
        -radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta)
    );
}

function clearGroup(group) {
    while (group.children.length > 0) {
        const child = group.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
            else child.material.dispose();
        }
        group.remove(child);
    }
}

const ZONE_COLORS = {
    eye: 0x4444ff, inner: 0x44ff44, wavefront: 0xff4444,
    outer: 0xffff44, far: 0x888888, antipodal: 0xcc88cc,
};

async function fetchJSON(endpoint) {
    try {
        const resp = await fetch(`${API_BASE}${endpoint}`);
        return await resp.json();
    } catch (e) {
        console.warn(`Fetch ${endpoint}:`, e.message);
        return null;
    }
}

// ===== Plate boundaries =====
async function loadPlates() {
    try {
        const resp = await fetch('src/plates.json');
        const boundaries = await resp.json();
        const mat = new THREE.LineBasicMaterial({ color: 0x444466, transparent: true, opacity: 0.5 });

        for (const segment of boundaries) {
            if (segment.length < 2) continue;
            const points = [];
            for (let i = 0; i < segment.length; i++) {
                const [lon, lat] = segment[i];
                // Skip segments that wrap around the dateline
                if (i > 0) {
                    const [prevLon] = segment[i - 1];
                    if (Math.abs(lon - prevLon) > 90) {
                        // Break the line at dateline crossings
                        if (points.length >= 2) {
                            const geom = new THREE.BufferGeometry().setFromPoints(points);
                            layers.plates.add(new THREE.Line(geom, mat));
                        }
                        points.length = 0;
                    }
                }
                points.push(latLonToVec3(lat, lon, EARTH_RADIUS * 1.0005));
            }
            if (points.length >= 2) {
                const geom = new THREE.BufferGeometry().setFromPoints(points);
                layers.plates.add(new THREE.Line(geom, mat));
            }
        }
        console.log(`Loaded ${boundaries.length} plate boundary segments`);
    } catch (e) {
        console.warn('Plate boundaries not loaded:', e.message);
    }
}
loadPlates();
layers.plates.visible = false; // off by default

// ===== Earthquake markers =====
function updateEarthquakes(data) {
    clearGroup(layers.earthquakes);
    if (!data?.earthquakes) return;

    // Instanced approach for performance
    data.earthquakes.forEach(eq => {
        const size = Math.max(0.005, Math.pow(eq.mag - 3.5, 1.5) * 0.003);
        const color = ZONE_COLORS[eq.zone] || 0xffffff;

        // Outer glow
        const glowGeom = new THREE.SphereGeometry(size * 2, 8, 8);
        const glowMat = new THREE.MeshBasicMaterial({
            color, transparent: true, opacity: 0.15,
        });
        const glow = new THREE.Mesh(glowGeom, glowMat);
        glow.position.copy(latLonToVec3(eq.lat, eq.lon, EARTH_RADIUS * 1.003));
        layers.earthquakes.add(glow);

        // Core dot
        const coreGeom = new THREE.SphereGeometry(size, 8, 8);
        const coreMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 });
        const core = new THREE.Mesh(coreGeom, coreMat);
        core.position.copy(glow.position);
        core.userData = eq;
        layers.earthquakes.add(core);
    });

    document.getElementById('st-eqs').textContent = data.earthquakes.length;
}

// ===== Subsolar point + Jelly Ball zones =====
function updateSubsolar(data) {
    clearGroup(layers.subsolar);
    clearGroup(layers['jelly-ball']);
    if (!data) return;

    // Subsolar point — pulsing yellow marker
    const ssGeom = new THREE.SphereGeometry(0.018, 16, 16);
    const ssMat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    const ssMesh = new THREE.Mesh(ssGeom, ssMat);
    ssMesh.position.copy(latLonToVec3(data.lat, data.lon, EARTH_RADIUS * 1.006));
    layers.subsolar.add(ssMesh);

    // Glow ring around subsolar
    const ringGeom = new THREE.RingGeometry(0.03, 0.05, 32);
    const ringMat = new THREE.MeshBasicMaterial({
        color: 0xffff00, transparent: true, opacity: 0.4, side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.position.copy(ssMesh.position.clone().multiplyScalar(1.001));
    ring.lookAt(0, 0, 0);
    layers.subsolar.add(ring);

    // Antipodal point
    const antiGeom = new THREE.SphereGeometry(0.01, 12, 12);
    const antiMat = new THREE.MeshBasicMaterial({ color: 0xcc88cc, transparent: true, opacity: 0.6 });
    const antiMesh = new THREE.Mesh(antiGeom, antiMat);
    antiMesh.position.copy(latLonToVec3(-data.lat, data.lon > 0 ? data.lon - 180 : data.lon + 180, EARTH_RADIUS * 1.005));
    layers.subsolar.add(antiMesh);

    // Jelly Ball zone arcs on the globe surface
    if (data.zones) {
        data.zones.forEach(zone => {
            const radRad = zone.radius_deg * Math.PI / 180;
            const color = parseInt(zone.color.replace('#', ''), 16);
            const points = [];

            for (let a = 0; a <= 360; a += 3) {
                const aRad = a * Math.PI / 180;
                // Point on sphere at angular distance radRad from subsolar
                const ssLat = data.lat * Math.PI / 180;
                const ssLon = data.lon * Math.PI / 180;
                const lat2 = Math.asin(
                    Math.sin(ssLat) * Math.cos(radRad) +
                    Math.cos(ssLat) * Math.sin(radRad) * Math.cos(aRad)
                );
                const lon2 = ssLon + Math.atan2(
                    Math.sin(aRad) * Math.sin(radRad) * Math.cos(ssLat),
                    Math.cos(radRad) - Math.sin(ssLat) * Math.sin(lat2)
                );
                points.push(latLonToVec3(lat2 * 180 / Math.PI, lon2 * 180 / Math.PI, EARTH_RADIUS * 1.002));
            }

            const geom = new THREE.BufferGeometry().setFromPoints(points);
            const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 });
            layers['jelly-ball'].add(new THREE.Line(geom, mat));
        });
    }

    // Update sun light direction to match subsolar
    const ssVec = latLonToVec3(data.lat, data.lon, 5);
    sunLight.position.copy(ssVec);
}

// ===== Terminator (day/night boundary) =====
function updateTerminator(data) {
    clearGroup(layers.terminator);
    if (!data) return;

    // Terminator is 90 degrees from subsolar
    const points = [];
    const ssLat = data.lat * Math.PI / 180;
    const ssLon = data.lon * Math.PI / 180;
    const radRad = Math.PI / 2; // 90 degrees

    for (let a = 0; a <= 360; a += 2) {
        const aRad = a * Math.PI / 180;
        const lat2 = Math.asin(
            Math.sin(ssLat) * Math.cos(radRad) +
            Math.cos(ssLat) * Math.sin(radRad) * Math.cos(aRad)
        );
        const lon2 = ssLon + Math.atan2(
            Math.sin(aRad) * Math.sin(radRad) * Math.cos(ssLat),
            Math.cos(radRad) - Math.sin(ssLat) * Math.sin(lat2)
        );
        points.push(latLonToVec3(lat2 * 180 / Math.PI, lon2 * 180 / Math.PI, EARTH_RADIUS * 1.003));
    }

    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({ color: 0xff8800, transparent: true, opacity: 0.3 });
    layers.terminator.add(new THREE.Line(geom, mat));
}

// ===== Sidebar mini-charts =====
function drawMiniChart(canvasId, data, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data?.length) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * 2;
    const h = canvas.height = canvas.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);

    const values = data.map(d => d.value).filter(v => v != null && isFinite(v));
    if (values.length < 2) return;

    const min = opts.min ?? Math.min(...values);
    const max = opts.max ?? Math.max(...values);
    const range = max - min || 1;

    // Background grid
    ctx.strokeStyle = '#222244';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Zero line if range crosses zero
    if (min < 0 && max > 0) {
        const zeroY = h - ((0 - min) / range) * h;
        ctx.strokeStyle = '#444488';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
    }

    // Data line
    ctx.strokeStyle = opts.color || '#00ccff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * w;
        const y = h - ((values[i] - min) / range) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill below for negative Bz
    if (opts.fillBelow) {
        const zeroY = h - ((0 - min) / range) * h;
        ctx.fillStyle = opts.fillColor || 'rgba(255,68,68,0.15)';
        ctx.beginPath();
        for (let i = 0; i < values.length; i++) {
            const x = (i / (values.length - 1)) * w;
            const y = h - ((values[i] - min) / range) * h;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.lineTo(w, zeroY);
        ctx.lineTo(0, zeroY);
        ctx.closePath();
        ctx.fill();
    }

    // Current value label
    const last = values[values.length - 1];
    ctx.fillStyle = opts.color || '#00ccff';
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(last.toFixed(opts.decimals ?? 1), w - 4, 22);
}

// ===== XRS chart (log scale) =====
function drawXRSChart(data) {
    const canvas = document.getElementById('xrs-chart');
    if (!canvas || !data?.xrs?.length) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * 2;
    const h = canvas.height = canvas.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);

    const entries = data.xrs;
    const values = entries.map(e => Math.log10(e.flux));
    const min = -8, max = -3; // B-class to X-class range
    const range = max - min;

    // Flare class lines
    [[-4, 'X', '#ff4444'], [-5, 'M', '#ffaa44'], [-6, 'C', '#4488ff'], [-7, 'B', '#333366']].forEach(([lvl, label, color]) => {
        const y = h - ((lvl - min) / range) * h;
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = '16px monospace';
        ctx.fillText(label, 4, y - 2);
    });

    // Data line
    ctx.strokeStyle = '#ff8844';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * w;
        const y = h - ((values[i] - min) / range) * h;
        const yc = Math.max(0, Math.min(h, y));
        if (i === 0) ctx.moveTo(x, yc); else ctx.lineTo(x, yc);
    }
    ctx.stroke();
}

// ===== Data update dispatch =====
function updateKp(data) {
    if (!data?.current) return;
    const kp = data.current;
    const el = document.getElementById('kp-metric');
    el.textContent = `Kp ${kp.toFixed(0)}`;
    el.className = 'metric ' + (kp < 4 ? 'quiet' : kp < 6 ? 'active' : 'storm');
    document.getElementById('st-kp').textContent = kp.toFixed(0);
    document.getElementById('st-kp').className = 'value ' + (kp < 4 ? 'ok' : kp < 6 ? '' : 'warn');
}

function updateSolarWind(data) {
    if (!data) return;
    if (data.current_bz != null) {
        document.getElementById('st-bz').textContent = `${data.current_bz.toFixed(1)} nT`;
        document.getElementById('st-bz').className = 'value ' + (data.current_bz < -10 ? 'warn' : 'ok');
    }
    if (data.current_speed != null) {
        document.getElementById('st-vsw').textContent = `${data.current_speed.toFixed(0)} km/s`;
        document.getElementById('st-vsw').className = 'value ' + (data.current_speed > 600 ? 'warn' : 'ok');
    }
    // Draw Bz chart
    drawMiniChart('sw-chart', data.bz, {
        color: '#ff6666', fillBelow: true, fillColor: 'rgba(255,68,68,0.1)', decimals: 1,
    });
}

function updateXRS(data) {
    if (!data) return;
    if (data.current_flux) {
        const f = data.current_flux;
        const cls = f >= 1e-4 ? `X${(f/1e-4).toFixed(1)}` :
                    f >= 1e-5 ? `M${(f/1e-5).toFixed(1)}` :
                    f >= 1e-6 ? `C${(f/1e-6).toFixed(1)}` : 'B';
        document.getElementById('st-xrs').textContent = cls;
    }
    drawXRSChart(data);

    // Order parameter
    const stateEl = document.getElementById('op-state');
    stateEl.textContent = data.state || 'UNKNOWN';
    stateEl.className = 'state ' + (data.state === 'FALLING' ? 'falling' :
                                     data.state === 'RISING' ? 'rising' : 'stable');
    document.getElementById('op-detail').textContent =
        `df/dt = ${(data.current_rate || 0).toExponential(2)} | ` +
        `Schumann proxy via GOES XRS fractional rate`;
}

function updateSun(data) {
    if (!data?.images) return;
    window._sunImages = data.images;
    const img = document.getElementById('sun-image');
    if (!img.src || !img.src.startsWith('http')) {
        img.src = data.images.aia_193 + '?t=' + Date.now();
    }
}

function updateLunar(data) {
    if (!data) return;
    document.getElementById('lunar-metric').textContent = `${data.name} (${data.illumination}%)`;
    document.getElementById('lunar-detail').textContent =
        `Tidal: ${data.tidal_force.toFixed(3)} | dF/dt: ${data.tidal_rate.toFixed(3)} | Full in ${data.days_to_full}d`;
    document.getElementById('st-moon').textContent = `${data.illumination}%`;
}

function updateCosmicRays(data) {
    if (!data?.stations) return;
    const stations = Object.keys(data.stations);
    if (!stations.length) return;
    const avg = stations.reduce((s, k) => s + data.stations[k].deviation_pct, 0) / stations.length;
    const el = document.getElementById('cr-metric');
    el.textContent = `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%`;
    el.className = 'metric ' + (data.forbush_detected ? 'storm' : 'quiet');
    document.getElementById('cr-detail').textContent =
        data.forbush_detected ? 'FORBUSH DECREASE DETECTED' : `${stations.length} stations nominal`;
}

// ===== Sun image selector =====
document.querySelectorAll('#sun-selector button').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#sun-selector button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (window._sunImages?.[btn.dataset.img]) {
            document.getElementById('sun-image').src = window._sunImages[btn.dataset.img] + '?t=' + Date.now();
        }
    });
});

// ===== Layer toggles =====
document.querySelectorAll('.layer-toggle input').forEach(input => {
    input.addEventListener('change', () => {
        const layer = layers[input.dataset.layer];
        if (layer) layer.visible = input.checked;
    });
});

// ===== Clock =====
function updateClock() {
    document.getElementById('clock').textContent =
        new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

// ===== Raycaster for earthquake tooltips =====
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.createElement('div');
tooltip.style.cssText = 'position:fixed;background:rgba(10,10,26,0.95);color:#ccc;font:11px monospace;padding:6px 10px;border:1px solid #00ccff;border-radius:4px;pointer-events:none;display:none;z-index:1000;';
document.body.appendChild(tooltip);

container.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(layers.earthquakes.children);

    if (hits.length > 0) {
        const eq = hits[0].object.userData;
        if (eq?.mag) {
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top = (e.clientY - 10) + 'px';
            tooltip.innerHTML = `<b>M${eq.mag.toFixed(1)}</b> ${eq.place}<br>` +
                `Depth: ${eq.depth?.toFixed(0) || '?'}km | ${eq.ang_dist}deg | ${eq.zone}`;
        }
    } else {
        tooltip.style.display = 'none';
    }
});

// ===== Main data poll =====
async function pollData() {
    console.log('Polling data...');
    const [eqs, ss, kp, sw, xrs, sun, lunar, cr] = await Promise.allSettled([
        fetchJSON('/earthquakes'),
        fetchJSON('/subsolar'),
        fetchJSON('/kp'),
        fetchJSON('/solar_wind'),
        fetchJSON('/xrs'),
        fetchJSON('/sun'),
        fetchJSON('/lunar'),
        fetchJSON('/cosmic_rays'),
    ]);

    if (eqs.value) updateEarthquakes(eqs.value);
    if (ss.value) {
        updateSubsolar(ss.value);
        updateTerminator(ss.value);
    }
    if (kp.value) updateKp(kp.value);
    if (sw.value) updateSolarWind(sw.value);
    if (xrs.value) updateXRS(xrs.value);
    if (sun.value) updateSun(sun.value);
    if (lunar.value) updateLunar(lunar.value);
    if (cr.value) updateCosmicRays(cr.value);
}

pollData();
setInterval(pollData, POLL_INTERVAL);

// ===== Animation =====
let frame = 0;
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    frame++;

    // Slow earth rotation (1 revolution per ~6 minutes for visibility)
    const rotSpeed = 0.0002;
    earth.rotation.y += rotSpeed;

    // Keep layers synced with earth rotation
    for (const key of ['earthquakes', 'jelly-ball', 'subsolar', 'plates', 'terminator', 'magnetometers']) {
        if (layers[key]) layers[key].rotation.y += rotSpeed;
    }

    // Pulse subsolar marker
    const ssMeshes = layers.subsolar.children;
    if (ssMeshes.length > 0) {
        const scale = 1 + 0.15 * Math.sin(frame * 0.05);
        ssMeshes[0].scale.setScalar(scale);
    }

    renderer.render(scene, camera);
}
animate();

// ===== Resize =====
window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});
