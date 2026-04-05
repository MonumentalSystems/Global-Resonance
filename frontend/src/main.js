import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

const API = window.location.port === '8001' ? '/api' : 'http://localhost:8001/api';
const SOLAR_API = window.SOLAR_MONITOR_URL || API + '/solar';
const R = 1; // earth radius
const POLL = 30_000; // 30s poll

// ===== Renderer =====
const box = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030308);
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
ctrl.maxDistance = 12;

// Bloom post-processing
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(box.clientWidth, box.clientHeight),
    1.2,   // strength — cinematic glow
    0.6,   // radius — wide bloom halo
    0.7    // threshold — more elements glow
);
composer.addPass(bloomPass);

// ===== Lighting =====
scene.add(new THREE.AmbientLight(0x334466, 0.6));
const sunLight = new THREE.DirectionalLight(0xffffff, 1.4);
sunLight.position.set(5, 2, 5);
scene.add(sunLight);

// Stars — varied sizes and colors
const STAR_COUNT = 6000;
const starGeo = new THREE.BufferGeometry();
const sv = new Float32Array(STAR_COUNT * 3);
const starColors = new Float32Array(STAR_COUNT * 3);
const starSizes = new Float32Array(STAR_COUNT);
for (let i = 0; i < STAR_COUNT; i++) {
    sv[i * 3] = (Math.random() - 0.5) * 60;
    sv[i * 3 + 1] = (Math.random() - 0.5) * 60;
    sv[i * 3 + 2] = (Math.random() - 0.5) * 60;
    // Color: mostly blue-white, occasional warm
    const temp = Math.random();
    starColors[i * 3] = temp > 0.9 ? 1.0 : 0.6 + Math.random() * 0.3;
    starColors[i * 3 + 1] = temp > 0.95 ? 0.5 : 0.7 + Math.random() * 0.3;
    starColors[i * 3 + 2] = temp > 0.9 ? 0.4 : 0.8 + Math.random() * 0.2;
    starSizes[i] = 0.008 + Math.random() * 0.02 + (Math.random() > 0.98 ? 0.04 : 0);
}
starGeo.setAttribute('position', new THREE.BufferAttribute(sv, 3));
starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ vertexColors: true, size: 0.015, sizeAttenuation: true, transparent: true, opacity: 0.9 })));

// ===== Globe =====
const earthGeo = new THREE.SphereGeometry(R, 128, 128);
const earthMat = new THREE.MeshPhongMaterial({
    color: 0x2244aa, emissive: 0x112244, specular: 0x222244, shininess: 12,
});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);

const wireGeo = new THREE.SphereGeometry(R * 1.001, 36, 18);
const wireMat = new THREE.MeshBasicMaterial({ color: 0x334466, wireframe: true, transparent: true, opacity: 0.15 });
const wireframe = new THREE.Mesh(wireGeo, wireMat);
scene.add(wireframe);

const tl = new THREE.TextureLoader();
tl.crossOrigin = 'anonymous';
tl.load('https://unpkg.com/three-globe@2.31.1/example/img/earth-blue-marble.jpg',
    t => { earthMat.map = t; earthMat.color.set(0xffffff); earthMat.emissive.set(0x000000); earthMat.needsUpdate = true; wireframe.visible = false; },
    undefined, e => console.warn('Earth texture failed:', e)
);
tl.load('https://unpkg.com/three-globe@2.31.1/example/img/earth-night.jpg',
    t => { earthMat.emissiveMap = t; earthMat.emissive.set(0x333333); earthMat.needsUpdate = true; },
    undefined, e => console.warn('Night texture failed:', e)
);

// Atmosphere — dual-layer glow
const atmosMat = new THREE.ShaderMaterial({
    vertexShader: `varying vec3 vNormal; varying vec3 vPos; void main() { vNormal = normalize(normalMatrix * normal); vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
        varying vec3 vNormal; varying vec3 vPos;
        void main() {
            float intensity = pow(0.65 - dot(vNormal, vec3(0,0,1)), 2.5);
            // Blue atmosphere with cyan rim
            vec3 col = mix(vec3(0.15, 0.4, 0.9), vec3(0.3, 0.9, 1.0), intensity);
            gl_FragColor = vec4(col, intensity * 0.35);
        }`,
    transparent: true, side: THREE.BackSide, depthWrite: false, blending: THREE.AdditiveBlending,
});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R * 1.025, 64, 64), atmosMat));
// Outer haze — very faint wide glow
const hazeMat = new THREE.MeshBasicMaterial({ color: 0x2266cc, transparent: true, opacity: 0.015, side: THREE.BackSide, depthWrite: false });
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R * 1.08, 32, 32), hazeMat));

// ===== Coordinates =====
function ll2v(lat, lon, r = R * 1.001) {
    const p = (90 - lat) * Math.PI / 180, t = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(-r * Math.sin(p) * Math.cos(t), r * Math.cos(p), r * Math.sin(p) * Math.sin(t));
}

function greatCirclePoints(lat1, lon1, radiusDeg, nPts = 120) {
    const pts = [], slat = lat1 * Math.PI / 180, slon = lon1 * Math.PI / 180, rd = radiusDeg * Math.PI / 180;
    for (let i = 0; i <= nPts; i++) {
        const a = (i / nPts) * 2 * Math.PI;
        const lat2 = Math.asin(Math.sin(slat) * Math.cos(rd) + Math.cos(slat) * Math.sin(rd) * Math.cos(a));
        const lon2 = slon + Math.atan2(Math.sin(a) * Math.sin(rd) * Math.cos(slat), Math.cos(rd) - Math.sin(slat) * Math.sin(lat2));
        pts.push(ll2v(lat2 * 180 / Math.PI, lon2 * 180 / Math.PI, R * 1.002));
    }
    return pts;
}

// ===== Layer System =====
const layerGroups = {};
const rotatingLayers = new Set();
const fixedLayers = new Set(['magnetosphere', 'solar-wind', 'comet']);

function getLayer(name) {
    if (!layerGroups[name]) {
        layerGroups[name] = new THREE.Group();
        scene.add(layerGroups[name]);
        if (!fixedLayers.has(name)) rotatingLayers.add(name);
    }
    return layerGroups[name];
}

function clearLayer(name) {
    const g = layerGroups[name];
    if (!g) return;
    while (g.children.length) {
        const c = g.children[0];
        if (c.geometry) c.geometry.dispose();
        if (c.material) { if (Array.isArray(c.material)) c.material.forEach(m => m.dispose()); else c.material.dispose(); }
        g.remove(c);
    }
}

// ===== EARTHQUAKE LAYER =====
const eqWaves = [];

function recencyColor(ageH) {
    if (ageH < 1) return new THREE.Color(1, 1, 1);
    if (ageH < 6) return new THREE.Color(1, 0.3 + 0.7 * (1 - ageH / 6), 0.1);
    if (ageH < 24) return new THREE.Color(1, 0.3 * (1 - (ageH - 6) / 18), 0);
    if (ageH < 48) return new THREE.Color(0.5, 0.2, 0.5);
    return new THREE.Color(0.2, 0.2, 0.5);
}

function depthToHeight(depth) {
    return 0.07 * Math.min((depth || 10) / 700, 1);
}

function updateEarthquakes(data) {
    clearLayer('earthquakes');
    eqWaves.length = 0;
    if (!data?.earthquakes) return;
    const now = Date.now(), layer = getLayer('earthquakes');

    data.earthquakes.forEach(eq => {
        const ageH = (now - eq.time) / 3600000;
        const col = recencyColor(ageH);
        const h = depthToHeight(eq.depth || 33);
        const baseSize = Math.max(0.003, Math.pow(eq.mag - 3.5, 1.3) * 0.003);
        const pos = ll2v(eq.lat, eq.lon, R + h);

        const coreGeo = new THREE.SphereGeometry(baseSize, 8, 8);
        const coreMat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.95 });
        const core = new THREE.Mesh(coreGeo, coreMat);
        core.position.copy(pos);
        core.userData = eq;
        layer.add(core);

        if (h > 0.005) {
            const surfPos = ll2v(eq.lat, eq.lon, R * 1.001);
            const stemGeo = new THREE.BufferGeometry().setFromPoints([surfPos, pos]);
            layer.add(new THREE.Line(stemGeo, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.4 })));
        }

        if (ageH < 24) {
            const waveSpeed = 0.3 + (eq.mag - 4) * 0.15;
            const maxRad = 2 + Math.pow(eq.mag - 4, 2) * 1.5;
            const currentRad = Math.min(ageH * waveSpeed, maxRad);
            const opacity = Math.max(0, 0.5 * (1 - currentRad / maxRad));
            if (currentRad > 0.2 && opacity > 0.02) {
                const ringPts = greatCirclePoints(eq.lat, eq.lon, currentRad, 60);
                layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts),
                    new THREE.LineBasicMaterial({ color: col, transparent: true, opacity })));
            }
            if (ageH < 2) {
                eqWaves.push({ lat: eq.lat, lon: eq.lon, mag: eq.mag, startTime: eq.time, color: col.clone(), maxRad, waveSpeed });
            }
        }

        if (eq.mag >= 6.0) {
            const glowGeo = new THREE.SphereGeometry(baseSize * 4, 12, 12);
            const glow = new THREE.Mesh(glowGeo, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.08 + (eq.mag - 6) * 0.03 }));
            glow.position.copy(pos);
            layer.add(glow);
        }
    });
    document.getElementById('st-eqs').textContent = data.earthquakes.length;
}

function animateWaves() {
    clearLayer('eq-waves');
    const now = Date.now(), layer = getLayer('eq-waves');
    for (const w of eqWaves) {
        const ageH = (now - w.startTime) / 3600000, rad = ageH * w.waveSpeed;
        if (rad > w.maxRad || rad < 0.1) continue;
        const opacity = 0.6 * (1 - rad / w.maxRad);
        layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(greatCirclePoints(w.lat, w.lon, rad, 80)),
            new THREE.LineBasicMaterial({ color: w.color, transparent: true, opacity: Math.max(0, opacity) })));
        const rad2 = rad * 0.6;
        if (rad2 > 0.1) {
            layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(greatCirclePoints(w.lat, w.lon, rad2, 80)),
                new THREE.LineBasicMaterial({ color: w.color, transparent: true, opacity: opacity * 0.4 })));
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
        layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.35 })));
    });
}

// ===== SUBSOLAR POINT =====
function updateSubsolar(data) {
    clearLayer('subsolar');
    if (!data) return;
    const layer = getLayer('subsolar');
    const geo = new THREE.SphereGeometry(0.015, 16, 16);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xffff00 }));
    m.position.copy(ll2v(data.lat, data.lon, R * 1.008));
    m.name = 'subsolar-pulse';
    layer.add(m);
    layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([ll2v(data.lat, data.lon, R * 1.008), ll2v(data.lat, data.lon, R * 1.15)]),
        new THREE.LineBasicMaterial({ color: 0xffff00, transparent: true, opacity: 0.3 })));
    const antiLon = data.lon > 0 ? data.lon - 180 : data.lon + 180;
    const anti = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 12), new THREE.MeshBasicMaterial({ color: 0xcc88cc, transparent: true, opacity: 0.5 }));
    anti.position.copy(ll2v(-data.lat, antiLon, R * 1.005));
    layer.add(anti);
    sunLight.position.copy(ll2v(data.lat, data.lon, 5));
}

function updateTerminator(data) {
    clearLayer('terminator');
    if (!data) return;
    getLayer('terminator').add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(greatCirclePoints(data.lat, data.lon, 90, 180)),
        new THREE.LineBasicMaterial({ color: 0xff6600, transparent: true, opacity: 0.2 })));
}

// ===== PLATE BOUNDARIES (GeoJSON-labeled) =====
const BOUNDARY_COLORS = {
    divergent: 0x44aaff,
    convergent: 0xff6644,
    transform: 0xffaa44,
    unknown: 0x445566,
};

async function loadPlates() {
    try {
        const resp = await fetch(`${API}/plates`);
        const geojson = await resp.json();
        clearLayer('plates');
        const layer = getLayer('plates');
        const legendEl = document.getElementById('plate-legend');
        const namesUsed = new Set();

        if (!geojson.features) {
            // Fallback: old plates.json format
            const segs = geojson;
            const mat = new THREE.LineBasicMaterial({ color: 0x445566, transparent: true, opacity: 0.35 });
            for (const seg of segs) {
                if (seg.length < 2) continue;
                const pts = [];
                for (let i = 0; i < seg.length; i++) {
                    if (i > 0 && Math.abs(seg[i][0] - seg[i - 1][0]) > 90) {
                        if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
                        pts.length = 0;
                    }
                    pts.push(ll2v(seg[i][1], seg[i][0], R * 1.0005));
                }
                if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
            }
            return;
        }

        for (const feature of geojson.features) {
            const props = feature.properties || {};
            const coords = feature.geometry?.coordinates || [];
            if (coords.length < 2) continue;

            const btype = props.boundary_type || 'unknown';
            const color = new THREE.Color(props.color || '#445566');
            const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 });

            const pts = [];
            for (let i = 0; i < coords.length; i++) {
                if (i > 0 && Math.abs(coords[i][0] - coords[i - 1][0]) > 90) {
                    if (pts.length >= 2) {
                        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
                        line.userData = { type: 'plate', name: props.name, boundary_type: btype };
                        layer.add(line);
                    }
                    pts.length = 0;
                }
                pts.push(ll2v(coords[i][1], coords[i][0], R * 1.0005));
            }
            if (pts.length >= 2) {
                const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
                line.userData = { type: 'plate', name: props.name, boundary_type: btype };
                layer.add(line);
            }

            namesUsed.add(props.name);
        }

        // Render plate legend
        if (legendEl) {
            const sorted = [...namesUsed].sort();
            legendEl.innerHTML = sorted.map(name => {
                const feat = geojson.features.find(f => f.properties.name === name);
                const color = feat?.properties.color || '#445566';
                const btype = feat?.properties.boundary_type || '';
                const symbol = btype === 'convergent' ? 'C' : btype === 'divergent' ? 'D' : 'T';
                return `<span class="plate-tag"><span class="swatch" style="background:${color}"></span>${name} (${symbol})</span>`;
            }).join('');
        }

        console.log(`Plates loaded: ${geojson.features.length} segments, ${namesUsed.size} named boundaries`);
    } catch (e) {
        console.warn('Plates:', e.message);
        // Fallback to old plates.json
        try {
            const resp = await fetch('src/plates.json');
            const segs = await resp.json();
            const mat = new THREE.LineBasicMaterial({ color: 0x445566, transparent: true, opacity: 0.35 });
            const layer = getLayer('plates');
            for (const seg of segs) {
                if (seg.length < 2) continue;
                const pts = [];
                for (let i = 0; i < seg.length; i++) {
                    if (i > 0 && Math.abs(seg[i][0] - seg[i - 1][0]) > 90) {
                        if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
                        pts.length = 0;
                    }
                    pts.push(ll2v(seg[i][1], seg[i][0], R * 1.0005));
                }
                if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
            }
        } catch (e2) { console.warn('Plates fallback failed:', e2.message); }
    }
}
loadPlates();

// ===== GEOJSON / KML LOADER =====
function renderGeoJSON(geojson, layerName = 'geojson') {
    const layer = getLayer(layerName);
    const features = geojson.features || (geojson.geometry ? [geojson] : []);

    for (const feature of features) {
        const geom = feature.geometry || feature;
        const props = feature.properties || {};
        const color = new THREE.Color(props.color || props.stroke || '#44cccc');

        if (geom.type === 'LineString') {
            renderLineString(geom.coordinates, color, layer, props);
        } else if (geom.type === 'MultiLineString') {
            for (const line of geom.coordinates) renderLineString(line, color, layer, props);
        } else if (geom.type === 'Polygon') {
            for (const ring of geom.coordinates) renderLineString(ring, color, layer, props);
        } else if (geom.type === 'MultiPolygon') {
            for (const poly of geom.coordinates) for (const ring of poly) renderLineString(ring, color, layer, props);
        } else if (geom.type === 'Point') {
            const [lon, lat] = geom.coordinates;
            const marker = new THREE.Mesh(new THREE.SphereGeometry(0.005, 8, 8), new THREE.MeshBasicMaterial({ color }));
            marker.position.copy(ll2v(lat, lon, R * 1.003));
            marker.userData = props;
            layer.add(marker);
        }
    }
}

function renderLineString(coords, color, layer, props) {
    const pts = [];
    for (let i = 0; i < coords.length; i++) {
        if (i > 0 && Math.abs(coords[i][0] - coords[i - 1][0]) > 90) {
            if (pts.length >= 2) layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 })));
            pts.length = 0;
        }
        pts.push(ll2v(coords[i][1], coords[i][0], R * 1.001));
    }
    if (pts.length >= 2) {
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
        line.userData = props;
        layer.add(line);
    }
}

function parseKML(text) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, 'text/xml');
    const features = [];
    doc.querySelectorAll('Placemark').forEach(pm => {
        const name = pm.querySelector('name')?.textContent || '';
        const coordEl = pm.querySelector('coordinates');
        if (!coordEl) return;
        const coords = coordEl.textContent.trim().split(/\s+/).map(c => {
            const [lon, lat] = c.split(',').map(Number);
            return [lon, lat];
        }).filter(c => !isNaN(c[0]) && !isNaN(c[1]));
        if (coords.length >= 2) {
            features.push({ type: 'Feature', properties: { name }, geometry: { type: 'LineString', coordinates: coords } });
        } else if (coords.length === 1) {
            features.push({ type: 'Feature', properties: { name }, geometry: { type: 'Point', coordinates: coords[0] } });
        }
    });
    return { type: 'FeatureCollection', features };
}

// Drag-and-drop handler
const dropZone = document.getElementById('drop-zone');
if (dropZone) {
    ['dragenter', 'dragover'].forEach(ev => dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => dropZone.addEventListener(ev, () => dropZone.classList.remove('over')));
    dropZone.addEventListener('drop', async e => {
        e.preventDefault();
        for (const file of e.dataTransfer.files) {
            const text = await file.text();
            const isKML = file.name.endsWith('.kml') || file.name.endsWith('.kmz');
            const geojson = isKML ? parseKML(text) : JSON.parse(text);
            clearLayer('geojson');
            renderGeoJSON(geojson, 'geojson');
            // Enable the layer checkbox
            const cb = document.querySelector('[data-layer="geojson"]');
            if (cb) cb.checked = true;
            if (layerGroups['geojson']) layerGroups['geojson'].visible = true;
            dropZone.textContent = `Loaded: ${file.name} (${geojson.features?.length || 0} features)`;
        }
    });
}

// ===== MAGNETOMETER STATIONS =====
function updateMagnetometers(data) {
    clearLayer('magnetometers');
    if (!data?.stations) return;
    const layer = getLayer('magnetometers');
    data.stations.forEach(st => {
        const pos = ll2v(st.lat, st.lon, R * 1.004);
        const mat = new THREE.MeshBasicMaterial({ color: st.network === 'USGS' ? 0xcc44cc : 0x44cccc, transparent: true, opacity: 0.8 });
        const mesh = new THREE.Mesh(new THREE.OctahedronGeometry(0.008, 0), mat);
        mesh.position.copy(pos);
        mesh.userData = { type: 'magnetometer', ...st };
        layer.add(mesh);
    });
}

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
    const mn = opts.min ?? Math.min(...vals), mx = opts.max ?? Math.max(...vals), rng = mx - mn || 1;
    ctx.strokeStyle = '#181833'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) { const y = (i / 4) * h; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    if (mn < 0 && mx > 0) {
        const zy = h - ((0 - mn) / rng) * h;
        ctx.strokeStyle = '#333366'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, zy); ctx.lineTo(w, zy); ctx.stroke();
    }
    if (opts.fillNeg) {
        const zy = h - ((0 - mn) / rng) * h;
        ctx.fillStyle = 'rgba(255,50,50,0.12)'; ctx.beginPath();
        vals.forEach((v, i) => { const x = (i / (vals.length - 1)) * w, y = h - ((v - mn) / rng) * h; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
        ctx.lineTo(w, zy); ctx.lineTo(0, zy); ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = opts.color || '#00ccff'; ctx.lineWidth = 1.5; ctx.beginPath();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1)) * w, y = Math.max(0, Math.min(h, h - ((v - mn) / rng) * h)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    ctx.fillStyle = opts.color || '#00ccff'; ctx.font = 'bold 20px monospace'; ctx.textAlign = 'right';
    ctx.fillText(vals[vals.length - 1].toFixed(opts.dec ?? 1), w - 4, 22);
}

function drawXRS(data) {
    const c = document.getElementById('xrs-chart');
    if (!c || !data?.xrs?.length) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.clientWidth * 2, h = c.height = c.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);
    const vals = data.xrs.map(e => Math.log10(e.flux));
    const mn = -8, mx = -3, rng = mx - mn;
    [[-4, 'X', '#ff4444'], [-5, 'M', '#ffaa44'], [-6, 'C', '#4488ff'], [-7, 'B', '#222244']].forEach(([lv, lb, co]) => {
        const y = h - ((lv - mn) / rng) * h;
        ctx.strokeStyle = co; ctx.lineWidth = 0.5; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = co; ctx.font = '14px monospace'; ctx.fillText(lb, 4, y - 2);
    });
    ctx.strokeStyle = '#ff8844'; ctx.lineWidth = 1.5; ctx.beginPath();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1)) * w, y = Math.max(0, Math.min(h, h - ((v - mn) / rng) * h)); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
}

// ===== SEISMOGRAM RENDERER =====
function drawSeismogram(data) {
    const c = document.getElementById('seismo-chart');
    if (!c || !data?.samples?.length) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.clientWidth * 2, h = c.height = c.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);

    const samples = data.samples;
    const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
    const centered = samples.map(v => v - mean);
    const maxAbs = Math.max(...centered.map(Math.abs)) || 1;

    // Background grid
    ctx.strokeStyle = '#181833'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

    // Waveform
    ctx.strokeStyle = '#44ff88'; ctx.lineWidth = 1.2; ctx.beginPath();
    centered.forEach((v, i) => {
        const x = (i / (centered.length - 1)) * w;
        const y = h / 2 - (v / maxAbs) * (h * 0.45);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill under waveform
    ctx.fillStyle = 'rgba(68,255,136,0.06)'; ctx.beginPath();
    centered.forEach((v, i) => {
        const x = (i / (centered.length - 1)) * w;
        const y = h / 2 - (v / maxAbs) * (h * 0.45);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h / 2); ctx.lineTo(0, h / 2); ctx.closePath(); ctx.fill();

    // Labels
    const stEl = document.getElementById('seismo-station');
    if (stEl && data.station) stEl.textContent = data.station;
    const timeEl = document.getElementById('seismo-time');
    if (timeEl) timeEl.textContent = data.start_time || '--';
    const ampEl = document.getElementById('seismo-amp');
    if (ampEl) ampEl.textContent = `pk: ${maxAbs.toFixed(0)} counts`;
}

// ===== JELLY BALL PREDICTION TRACKER =====
function updJellyBall(data) {
    if (!data || data.error) return;

    // J gauge
    const jEl = document.getElementById('jb-j');
    if (jEl) {
        jEl.textContent = data.j_current?.toFixed(3) || '--';
        jEl.style.color = data.above_critical ? '#ff4444' : data.gap_pct < 10 ? '#ffaa44' : '#44ff44';
    }
    const bar = document.getElementById('jb-bar');
    if (bar) {
        bar.style.width = Math.min(100, (data.j_current || 0) * 100) + '%';
        bar.style.background = data.above_critical ? '#ff4444' : data.gap_pct < 10 ? '#ffaa44' : '#4488ff';
    }
    const marker = document.getElementById('jb-jc-marker');
    if (marker) marker.style.width = ((data.j_critical || 0.637) * 100) + '%';
    const gapEl = document.getElementById('jb-gap');
    if (gapEl) {
        const sign = data.gap > 0 ? '-' : '+';
        gapEl.textContent = `${sign}${Math.abs(data.gap_pct || 0).toFixed(1)}%`;
    }

    // Phase badge
    const phaseEl = document.getElementById('jb-phase');
    if (phaseEl) {
        phaseEl.textContent = data.phase || '--';
        const p = (data.phase || '').toLowerCase();
        phaseEl.className = 'esc-badge ' + (
            p.includes('storm') ? 'esc-flare' : p.includes('critical') ? 'esc-active' :
            p.includes('recovery') ? 'esc-elevated' : 'esc-quiet'
        );
        phaseEl.style.fontSize = '8px'; phaseEl.style.padding = '1px 5px';
    }

    // Coupling info
    const xiEl = document.getElementById('jb-xi');
    if (xiEl) {
        const xi = data.correlation_length_km;
        xiEl.textContent = xi > 1e6 ? `${(xi / 1e6).toFixed(1)}M km` : xi > 1e3 ? `${(xi / 1e3).toFixed(0)}k` : `${xi?.toFixed(0) || '--'}`;
        xiEl.style.color = xi > 1e6 ? '#ff4444' : xi > 1e5 ? '#ffaa44' : '#44aaff';
    }
    const shieldEl = document.getElementById('jb-shield');
    if (shieldEl) {
        shieldEl.textContent = data.shield || '--';
        shieldEl.style.color = data.shield === 'ON' ? '#4f4' : data.shield === 'OFF' ? '#f44' : '#ff4';
    }

    // Three-body coupling indicators
    // Solar l=2: from Kp (higher Kp = stronger solar driving)
    const solarEl = document.getElementById('h-solar');
    if (solarEl && data.inputs) {
        const kp = data.inputs.kp || 0;
        const solarL2 = Math.min(1, kp / 9);
        solarEl.textContent = solarL2.toFixed(2);
        solarEl.style.color = solarL2 > 0.5 ? '#ff8844' : '#556';
    }
    // Lunar l=2: compute from current lunar phase (fortnightly M2)
    const lunarEl = document.getElementById('h-lunar');
    if (lunarEl) {
        // Quick lunar phase calc
        const ref = new Date('2000-01-06T00:00:00Z').getTime();
        const now = Date.now();
        const phase = ((now - ref) / 86400000 % 29.53059) / 29.53059;
        const m2 = Math.abs(Math.cos(2 * Math.PI * phase)); // 1 at new/full, 0 at quarters
        lunarEl.textContent = m2.toFixed(2);
        lunarEl.style.color = m2 > 0.7 ? '#88aaff' : '#556';
    }
    // Storm l=2: from J gap (closer to J_c = stronger ringing)
    const stormEl = document.getElementById('h-storm');
    if (stormEl && data.gap_pct != null) {
        const stormL2 = Math.max(0, 1 - Math.abs(data.gap_pct) / 30);
        stormEl.textContent = stormL2.toFixed(2);
        stormEl.style.color = stormL2 > 0.5 ? '#44ff88' : '#556';
    }

    const detailEl = document.getElementById('jb-detail');
    if (detailEl) detailEl.textContent = data.phase_detail || '--';
}

// ===== JELLYBALL NEURAL PREDICTIONS =====
let nnData = null;
let nnPhase = 'compression';

function updNeural(data) {
    if (!data || data.error) return;
    nnData = data;
    renderNeuralZones();

    // Mode amplitude bars
    const modes = data.diagnostics?.mode_amplitudes;
    if (modes) {
        for (let l = 1; l <= 6; l++) {
            const key = `l${l}`;
            const val = modes[key] ?? 0;
            const bar = document.getElementById(`mode-l${l}`);
            const score = document.getElementById(`ms-l${l}`);
            if (bar) {
                const pct = Math.min(100, Math.abs(val) / 5 * 100);
                const hue = val > 0 ? (l === 2 ? '#ff8844' : l === 3 ? '#44aaff' : '#44ff88')
                                     : (l === 2 ? '#ff4466' : '#6644aa');
                bar.style.width = pct + '%';
                bar.style.background = hue;
            }
            if (score) score.textContent = (val > 0 ? '+' : '') + val.toFixed(2);
        }
    }

    // Bivector norm
    const bivEl = document.getElementById('nn-biv');
    if (bivEl && data.diagnostics?.bivector_norm != null) {
        bivEl.textContent = data.diagnostics.bivector_norm.toFixed(1);
    }
}

function renderNeuralZones() {
    if (!nnData?.predictions?.[nnPhase]) return;
    const zones = nnData.predictions[nnPhase];
    const container = document.getElementById('nn-zones');
    if (!container) return;

    container.innerHTML = Object.entries(zones).map(([name, ratio]) => {
        const pct = Math.min(100, Math.max(0, (ratio - 0.2) / 4.8 * 100));
        const color = ratio > 1.5 ? '#ff4444' : ratio > 1.1 ? '#ffaa44' : ratio > 0.9 ? '#44ff44' : '#4488ff';
        return `<div class="det-row">
            <span class="det-label">${name}</span>
            <div class="det-bar-bg"><div class="det-bar" style="width:${pct}%;background:${color}"></div></div>
            <span class="det-score" style="color:${color}">${ratio.toFixed(2)}</span>
        </div>`;
    }).join('');
}

// Phase tab buttons
document.querySelectorAll('#nn-phase-tabs button').forEach(btn => {
    btn.addEventListener('click', () => {
        nnPhase = btn.dataset.phase;
        document.querySelectorAll('#nn-phase-tabs button').forEach(b => {
            b.style.background = b === btn ? '#0cf' : '#141430';
            b.style.color = b === btn ? '#050510' : '#aaa';
            b.style.borderColor = b === btn ? '#0cf' : '#2a2a44';
        });
        renderNeuralZones();
    });
});

// ===== FIELD STRENGTHS UPDATER =====
function updFieldStrengths(data) {
    if (!data || data.error) return;

    const fwf = data.fair_weather_ez;
    if (fwf) {
        document.getElementById('fv-fwf').textContent = `${fwf.value} V/m`;
        document.getElementById('fb-fwf').style.width = Math.min(100, (fwf.value / 250) * 100) + '%';
        const stEz = document.getElementById('st-ez');
        if (stEz) stEz.textContent = `${fwf.value} V/m`;
    }
    const tel = data.telluric_j;
    if (tel) {
        document.getElementById('fv-telluric').textContent = `${tel.value} mA/km`;
        document.getElementById('fb-telluric').style.width = Math.min(100, (tel.value / 100) * 100) + '%';
        document.getElementById('fb-telluric').style.background = tel.value > 20 ? '#ff4444' : '#ff8844';
    }
    const man = data.mansurov_dbdt;
    if (man) {
        document.getElementById('fv-mansurov').textContent = `${man.value} nT/hr`;
        document.getElementById('fb-mansurov').style.width = Math.min(100, (man.value / 50) * 100) + '%';
    }
    const sch = data.schumann_f1;
    if (sch) {
        document.getElementById('fv-schumann').textContent = `${sch.value} Hz`;
        // Normalize around 7.83 baseline
        document.getElementById('fb-schumann').style.width = Math.min(100, (sch.value / 8.5) * 100) + '%';
    }
    const gic = data.gic_risk;
    if (gic) {
        document.getElementById('fv-gic').textContent = gic.label;
        document.getElementById('fb-gic').style.width = (gic.score * 100) + '%';
        document.getElementById('fb-gic').style.background = gic.score > 0.5 ? '#ff4444' : gic.score > 0.2 ? '#ffaa44' : '#44ff44';
        document.getElementById('fv-gic').style.color = gic.score > 0.5 ? '#ff4444' : gic.score > 0.2 ? '#ffaa44' : '#44ff44';
    }
}

// ===== Status updaters =====
function updKp(d) {
    if (!d?.current) return;
    const k = d.current, el = document.getElementById('kp-metric');
    el.textContent = `Kp ${k.toFixed(0)}`;
    el.className = 'm ' + (k < 4 ? 'q' : k < 6 ? 'a' : 's');
    const s = document.getElementById('st-kp');
    s.textContent = k.toFixed(0); s.className = 'v ' + (k < 4 ? 'g' : k < 6 ? '' : 'w');
}

function updSW(d) {
    if (!d) return;
    if (d.current_bz != null) {
        const e = document.getElementById('st-bz');
        e.textContent = `${d.current_bz.toFixed(1)}`;
        e.className = 'v ' + (d.current_bz < -10 ? 'w' : 'g');
    }
    if (d.current_speed != null) {
        const e = document.getElementById('st-vsw');
        e.textContent = `${d.current_speed.toFixed(0)}`;
        e.className = 'v ' + (d.current_speed > 600 ? 'w' : 'g');
    }
    drawChart('sw-chart', d.bz, { color: '#ff6666', fillNeg: true, dec: 1 });
}

function updXRS(d) {
    if (!d) return;
    if (d.current_flux) {
        const f = d.current_flux;
        const cl = f >= 1e-4 ? `X${(f / 1e-4).toFixed(1)}` : f >= 1e-5 ? `M${(f / 1e-5).toFixed(1)}` : f >= 1e-6 ? `C${(f / 1e-6).toFixed(1)}` : 'B';
        document.getElementById('st-xrs').textContent = cl;
    }
    drawXRS(d);
    const el = document.getElementById('op-state');
    el.textContent = d.state || '?';
    el.className = 'state ' + (d.state === 'FALLING' ? 'falling' : d.state === 'RISING' ? 'rising' : 'stable');
}

function updSun(d) {
    if (!d?.images) return;
    window._si = d.images;
    const img = document.getElementById('sun-image');
    if (!img.dataset.loaded) { img.src = d.images.eit_195 || Object.values(d.images)[0]; img.dataset.loaded = '1'; }
}

function updLunar(d) {
    if (!d) return;
    document.getElementById('lunar-metric').textContent = `${d.name}`;
    document.getElementById('lunar-detail').textContent = `${d.illumination}% | F:${d.tidal_force.toFixed(2)} | dF:${d.tidal_rate.toFixed(2)}`;
    document.getElementById('st-moon').textContent = `${d.illumination}%`;
}

function updCR(d) {
    if (!d?.stations) return;
    const ks = Object.keys(d.stations);
    if (!ks.length) return;
    const avg = ks.reduce((s, k) => s + d.stations[k].deviation_pct, 0) / ks.length;
    const el = document.getElementById('cr-metric');
    el.textContent = `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%`;
    el.className = 'm ' + (d.forbush_detected ? 's' : 'q');
    document.getElementById('cr-detail').textContent = d.forbush_detected ? 'FORBUSH DECREASE' : `${ks.length} stations nominal`;
    document.getElementById('st-cr').textContent = `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%`;
    document.getElementById('st-cr').className = 'v ' + (d.forbush_detected ? 'w' : 'g');
}

function updGlobalCR(d) {
    if (!d || d.error) return;
    const stEl = document.getElementById('cr-stations');
    if (stEl) stEl.textContent = `${d.n_stations || 0} stations`;
    if (d.global_mean != null) {
        const el = document.getElementById('cr-metric');
        if (el) {
            el.textContent = `${d.global_mean > 0 ? '+' : ''}${d.global_mean.toFixed(1)}%`;
            el.className = 'm ' + (d.forbush ? 's' : 'q');
        }
        const fb = document.getElementById('cr-forbush');
        if (fb) {
            fb.textContent = d.forbush ? 'FORBUSH DECREASE' : 'nominal';
            fb.style.color = d.forbush ? '#f44' : '#4f4';
        }
    }
}

function updTEC(d) {
    if (!d) return;
    const el = document.getElementById('tec-metric');
    const det = document.getElementById('tec-detail');
    if (d.available) {
        if (el) { el.textContent = 'LIVE'; el.className = 'm q'; }
        if (det) det.textContent = d.dataset || 'USTEC';
    } else {
        if (el) { el.textContent = 'N/A'; el.className = 'm'; }
        if (det) det.textContent = d.note?.substring(0, 30) || 'unavailable';
    }
}

function updPrecip(d) {
    if (!d) return;
    const el = document.getElementById('precip-metric');
    const det = document.getElementById('precip-detail');
    if (el) {
        el.textContent = `${d.global_precip_72h || 0} mm`;
        el.className = 'm ' + (d.global_precip_72h > 100 ? 'a' : 'q');
    }
    if (det) {
        const thunder = d.global_thunder_hours || 0;
        det.textContent = `${d.n_stations || 0} sites | ${thunder} storm-hrs`;
    }
    // Render precipitation + thunderstorm markers on globe
    renderWeatherMarkers(d);
}

function updLightning(d) {
    if (!d) return;
    const el = document.getElementById('lightning-metric');
    const det = document.getElementById('lightning-detail');
    if (el) {
        const clim = d.climatology;
        if (clim) {
            el.textContent = clim.month;
            el.style.color = '#ffaa44';
        }
        const rt = d.realtime_thunder_hours || 0;
        if (rt > 0 && el) el.textContent += ` (${rt}h)`;
    }
    if (det && d.climatology?.hotspots) {
        const top = d.climatology.hotspots.sort((a, b) => b.mean_density - a.mean_density)[0];
        det.textContent = top ? `peak: ${top.name}` : 'WWLLN climatology';
    }
}

// --- WEATHER INDICATORS ON GLOBE ---
function renderWeatherMarkers(precipData) {
    clearLayer('weather');
    if (!precipData?.stations) return;
    const layer = getLayer('weather');

    for (const st of precipData.stations) {
        const pos = ll2v(st.lat, st.lon, R * 1.012);
        const isThunder = st.thunder_hours > 0;
        const hasRain = st.total_72h_mm > 5;

        if (!hasRain && !isThunder) continue;

        // Rain: blue droplet column, height = precipitation amount
        if (hasRain) {
            const rainHeight = Math.min(0.12, st.total_72h_mm / 300);
            const topPos = ll2v(st.lat, st.lon, R * 1.012 + rainHeight);
            const rainGeo = new THREE.BufferGeometry().setFromPoints([pos, topPos]);
            const rainMat = new THREE.LineBasicMaterial({
                color: 0x4488ff, transparent: true,
                opacity: Math.min(0.7, 0.2 + st.total_72h_mm / 100),
                blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const line = new THREE.Line(rainGeo, rainMat);
            line.userData = { type: 'precip', ...st };
            layer.add(line);

            // Small glow at base
            const baseGeo = new THREE.SphereGeometry(0.006, 6, 6);
            const baseMat = new THREE.MeshBasicMaterial({
                color: 0x4488ff, transparent: true, opacity: 0.4, depthWrite: false,
            });
            const base = new THREE.Mesh(baseGeo, baseMat);
            base.position.copy(pos);
            layer.add(base);
        }

        // Thunderstorm: yellow-orange flash marker
        if (isThunder) {
            // Lightning bolt (small zig-zag line)
            const boltBase = ll2v(st.lat, st.lon, R * 1.015);
            const boltTop = ll2v(st.lat, st.lon, R * 1.035 + st.thunder_hours * 0.003);
            const boltMid1 = boltBase.clone().lerp(boltTop, 0.33);
            boltMid1.x += 0.008; boltMid1.z += 0.005;
            const boltMid2 = boltBase.clone().lerp(boltTop, 0.66);
            boltMid2.x -= 0.006; boltMid2.z -= 0.004;
            const boltGeo = new THREE.BufferGeometry().setFromPoints([boltBase, boltMid1, boltMid2, boltTop]);
            const boltMat = new THREE.LineBasicMaterial({
                color: 0xffcc44, transparent: true, opacity: 0.8,
                blending: THREE.AdditiveBlending, depthWrite: false,
            });
            const bolt = new THREE.Line(boltGeo, boltMat);
            bolt.name = 'thunder-bolt';
            bolt.userData = { type: 'thunder', ...st };
            layer.add(bolt);

            // Glow halo
            const glowGeo = new THREE.SphereGeometry(0.01 + st.thunder_hours * 0.002, 8, 8);
            const glowMat = new THREE.MeshBasicMaterial({
                color: 0xffaa22, transparent: true, opacity: 0.15,
                depthWrite: false, blending: THREE.AdditiveBlending,
            });
            const glow = new THREE.Mesh(glowGeo, glowMat);
            glow.position.copy(ll2v(st.lat, st.lon, R * 1.02));
            glow.name = 'thunder-glow';
            layer.add(glow);
        }
    }
}

// Animate thunder flicker
function animateWeather(frame) {
    const layer = layerGroups['weather'];
    if (!layer) return;
    layer.children.forEach(c => {
        if (c.name === 'thunder-bolt') {
            // Random flicker
            c.material.opacity = 0.3 + 0.5 * (Math.sin(frame * 0.5 + Math.random() * 10) > 0.3 ? 1 : 0);
        }
        if (c.name === 'thunder-glow') {
            c.material.opacity = 0.08 + 0.12 * Math.sin(frame * 0.15 + c.position.x * 10);
        }
    });
}

function updDst(data) {
    if (!data) return;
    const el = document.getElementById('dst-metric'), st = document.getElementById('st-dst');
    if (data.current != null) {
        el.textContent = `${data.current} nT`;
        el.className = 'm ' + (data.current > -30 ? 'q' : data.current > -50 ? 'a' : 's');
        st.textContent = `${data.current}`;
        st.className = 'v ' + (data.current > -30 ? 'g' : data.current > -50 ? '' : 'w');
    }
}

// Sun image selector
document.querySelectorAll('#sun-selector button').forEach(b => {
    b.addEventListener('click', () => {
        document.querySelectorAll('#sun-selector button').forEach(x => x.classList.remove('on'));
        b.classList.add('on');
        if (window._si?.[b.dataset.img])
            document.getElementById('sun-image').src = window._si[b.dataset.img] + '?t=' + Date.now();
    });
});

// ===== Tooltip =====
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tip = document.createElement('div');
tip.style.cssText = 'position:fixed;background:rgba(5,5,16,0.95);color:#ccc;font:11px monospace;padding:6px 10px;border:1px solid #00ccff;border-radius:4px;pointer-events:none;display:none;z-index:1000;max-width:280px;';
document.body.appendChild(tip);

box.addEventListener('mousemove', e => {
    const r = box.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(mouse, camera);

    // Check earthquakes
    const eqLayer = layerGroups['earthquakes'];
    if (eqLayer) {
        const hits = ray.intersectObjects(eqLayer.children);
        const hit = hits.find(h => h.object.userData?.mag);
        if (hit) {
            const eq = hit.object.userData, ageH = (Date.now() - eq.time) / 3600000;
            const zc = { eye: '#44f', inner: '#66c', transition: '#4a4', wavefront: '#f44', 'wavefront-tail': '#f84', neutral: '#884', 'far-suppress': '#468', 'far-neutral': '#666', 'pre-antipodal': '#868', antipodal: '#c8c' };
            tip.innerHTML = `<b style="color:#ff6644">M${eq.mag.toFixed(1)}</b> ${eq.place}<br>Depth: ${eq.depth?.toFixed(0) || '?'}km | ${ageH.toFixed(1)}h ago<br>${eq.ang_dist}deg | <span style="color:${zc[eq.zone] || '#888'}">${eq.zone}</span>`;
            tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY - 10) + 'px';
            box.style.cursor = 'pointer';
            return;
        }
    }

    // Check plates
    const plateLayer = layerGroups['plates'];
    if (plateLayer?.visible) {
        const hits = ray.intersectObjects(plateLayer.children);
        const hit = hits.find(h => h.object.userData?.type === 'plate');
        if (hit) {
            const p = hit.object.userData;
            tip.innerHTML = `<b style="color:#4488ff">${p.name}</b><br><span style="color:#889">${p.boundary_type}</span>`;
            tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY - 10) + 'px';
            box.style.cursor = 'pointer';
            return;
        }
    }

    // Check weather markers
    const wxLayer = layerGroups['weather'];
    if (wxLayer?.visible) {
        const hits = ray.intersectObjects(wxLayer.children);
        const hit = hits.find(h => h.object.userData?.type === 'precip' || h.object.userData?.type === 'thunder');
        if (hit) {
            const w = hit.object.userData;
            if (w.type === 'thunder') {
                tip.innerHTML = `<b style="color:#ffcc44">${w.name}</b><br>${w.thunder_hours}h thunderstorm<br>${w.total_72h_mm}mm rain (72h)`;
            } else {
                tip.innerHTML = `<b style="color:#4488ff">${w.name}</b><br>${w.total_72h_mm}mm rain (72h)<br>${w.current_mm}mm/hr now`;
            }
            tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY - 10) + 'px';
            box.style.cursor = 'pointer';
            return;
        }
    }

    tip.style.display = 'none';
    box.style.cursor = 'grab';
});

// Click handler
box.addEventListener('click', e => {
    const r = box.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(mouse, camera);

    const eqLayer = layerGroups['earthquakes'];
    if (eqLayer) {
        const hits = ray.intersectObjects(eqLayer.children);
        const hit = hits.find(h => h.object.userData?.mag);
        if (hit) { showDetail(hit.object.userData); return; }
    }
    const magLayer = layerGroups['magnetometers'];
    if (magLayer?.visible) {
        const hits = ray.intersectObjects(magLayer.children);
        const hit = hits.find(h => h.object.userData?.type === 'magnetometer');
        if (hit) { showMagDetail(hit.object.userData); return; }
    }
    document.getElementById('detail').style.display = 'none';
});

function showDetail(eq) {
    const panel = document.getElementById('detail'), content = document.getElementById('detail-content');
    const ageH = (Date.now() - eq.time) / 3600000, dt = new Date(eq.time);
    const zc = { eye: '#44f', inner: '#66c', transition: '#4a4', wavefront: '#f44', 'wavefront-tail': '#f84', neutral: '#884', 'far-suppress': '#468', 'far-neutral': '#666', 'pre-antipodal': '#868', antipodal: '#c8c' };
    const zr = { eye: '0.85x', inner: '0.92x', transition: '0.98x', wavefront: '1.36x', 'wavefront-tail': '1.09x', neutral: '0.95x', 'far-suppress': '0.82x', 'far-neutral': '0.90x', 'pre-antipodal': '1.00x', antipodal: '1.16x' };
    content.innerHTML = `<h3>M${eq.mag.toFixed(1)} ${eq.place || 'Unknown'}</h3>
        <div class="row"><span class="k">Time</span><span class="val">${dt.toISOString().replace('T', ' ').substring(0, 19)} UTC</span></div>
        <div class="row"><span class="k">Age</span><span class="val">${ageH < 1 ? (ageH * 60).toFixed(0) + ' min' : ageH.toFixed(1) + ' hours'} ago</span></div>
        <div class="row"><span class="k">Location</span><span class="val">${eq.lat.toFixed(3)}N, ${eq.lon.toFixed(3)}E</span></div>
        <div class="row"><span class="k">Depth</span><span class="val">${eq.depth?.toFixed(1) || '?'} km</span></div>
        <div class="row"><span class="k">Subsolar dist</span><span class="val">${eq.ang_dist} deg</span></div>
        <div class="row"><span class="k">Jelly Ball zone</span><span class="zone-badge" style="background:${zc[eq.zone] || '#444'};color:#fff">${eq.zone} (${zr[eq.zone] || '?'})</span></div>
        <div style="margin-top:8px;border-top:1px solid #222;padding-top:6px;"><a href="https://earthquake.usgs.gov/earthquakes/eventpage/${eq.id || ''}" target="_blank">USGS Event Page &rarr;</a></div>`;
    panel.style.display = 'block';
}

function showMagDetail(st) {
    const panel = document.getElementById('detail'), content = document.getElementById('detail-content');
    content.innerHTML = `<h3 style="color:#cc44cc">${st.code} - ${st.name}</h3>
        <div class="row"><span class="k">Network</span><span class="val">${st.network}</span></div>
        <div class="row"><span class="k">Location</span><span class="val">${st.lat.toFixed(2)}N, ${st.lon.toFixed(2)}E</span></div>
        ${st.live ? `<div style="margin-top:6px;border-top:1px solid #222;padding-top:6px;">
            <div class="row"><span class="k">B_X</span><span class="val">${st.live.X?.toFixed(1) || '?'} nT</span></div>
            <div class="row"><span class="k">B_Y</span><span class="val">${st.live.Y?.toFixed(1) || '?'} nT</span></div>
            <div class="row"><span class="k">B_Z</span><span class="val">${st.live.Z?.toFixed(1) || '?'} nT</span></div>
        </div>` : '<div class="d" style="margin-top:6px">No live data</div>'}`;
    panel.style.display = 'block';
}

// ===== PALEOMAG =====
let palemagData = null;
async function loadPaleomag() { palemagData = await fetchJSON('/paleomag'); if (palemagData?.sites) drawPaleomagChart(); }
function drawPaleomagChart() {
    const c = document.getElementById('paleomag-chart');
    if (!c || !palemagData?.sites) return;
    const ctx = c.getContext('2d');
    const w = c.width = c.clientWidth * 2, h = c.height = c.clientHeight * 2;
    ctx.clearRect(0, 0, w, h);
    const times = palemagData.times, tMin = -500, tMax = 2500, fMin = 30, fMax = 80;
    ctx.strokeStyle = '#181833'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) { const y = (i / 4) * h; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    const x1200 = ((1200 - tMin) / (tMax - tMin)) * w;
    ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x1200, 0); ctx.lineTo(x1200, h); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#ff4444'; ctx.font = '14px monospace'; ctx.fillText('1200', x1200 + 2, 14);
    const drawSite = (name, color, thick) => {
        const vals = palemagData.sites[name]?.values; if (!vals) return;
        ctx.strokeStyle = color; ctx.lineWidth = thick ? 2.5 : 1; ctx.globalAlpha = thick ? 1.0 : 0.5; ctx.beginPath();
        for (let i = 0; i < times.length; i++) { const x = ((times[i] - tMin) / (tMax - tMin)) * w, y = h - ((vals[i] - fMin) / (fMax - fMin)) * h; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
        ctx.stroke(); ctx.globalAlpha = 1.0;
    };
    drawSite('Greece', '#44aaff', false); drawSite('Anatolia', '#ffaa44', false); drawSite('Egypt', '#44ff44', false);
    drawSite('Levant', '#ff4444', true); drawSite('China', '#ffff44', true);
    ctx.font = '12px monospace'; ctx.fillStyle = '#ff4444'; ctx.fillText('Levant', 4, h - 8); ctx.fillStyle = '#ffff44'; ctx.fillText('China', 70, h - 8);
}
document.getElementById('paleomag-toggle')?.addEventListener('change', e => {
    const panel = document.getElementById('paleomag-panel');
    if (panel) panel.style.display = e.target.checked ? 'block' : 'none';
    if (e.target.checked && !palemagData) loadPaleomag();
});

// ===== TIME SLIDER =====
const timeSlider = document.getElementById('time-slider');
const timeVal = document.getElementById('time-val');
const timeLive = document.getElementById('time-live');
let isLive = true, historyHoursBack = 0;
if (timeSlider) {
    document.getElementById('time-control').classList.add('visible');
    timeSlider.addEventListener('input', () => {
        historyHoursBack = parseInt(timeSlider.value);
        isLive = historyHoursBack === 0;
        timeVal.textContent = isLive ? 'LIVE' : `-${historyHoursBack}h`;
        timeLive.classList.toggle('on', isLive);
    });
    timeLive.addEventListener('click', () => { timeSlider.value = 0; historyHoursBack = 0; isLive = true; timeVal.textContent = 'LIVE'; timeLive.classList.add('on'); });
}

// Clock
setInterval(() => {
    const now = isLive ? new Date() : new Date(Date.now() - historyHoursBack * 3600000);
    document.getElementById('clock').textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC' + (isLive ? '' : ` (-${historyHoursBack}h)`);
}, 1000);

// ===== Layer toggles =====
document.querySelectorAll('[data-layer]').forEach(inp => {
    inp.addEventListener('change', () => { const g = layerGroups[inp.dataset.layer]; if (g) g.visible = inp.checked; });
});
setTimeout(() => {
    document.querySelectorAll('[data-layer]').forEach(inp => {
        if (!inp.checked && layerGroups[inp.dataset.layer]) layerGroups[inp.dataset.layer].visible = false;
    });
}, 200);

// ===== MAIN POLL =====
async function poll() {
    const results = await Promise.allSettled([
        fetchJSON('/earthquakes'),     // 0
        fetchJSON('/subsolar'),        // 1
        fetchJSON('/kp'),              // 2
        fetchJSON('/solar_wind'),      // 3
        fetchJSON('/xrs'),             // 4
        fetchJSON('/sun'),             // 5
        fetchJSON('/lunar'),           // 6
        fetchJSON('/cosmic_rays'),     // 7
        fetchJSON('/dst'),             // 8
        fetchJSON('/magnetometers'),   // 9
        fetchJSON('/field_strengths'),       // 10
        fetchJSON('/seismic/waveform'),      // 11
        fetchJSON('/jellyball/prediction'),  // 12
        fetchJSON('/jellyball/neural'),      // 13
        fetchJSON('/cosmic_rays_global'),    // 14
        fetchJSON('/tec'),                   // 15
        fetchJSON('/precipitation'),         // 16
        fetchJSON('/lightning'),             // 17
    ]);
    const v = i => results[i]?.value;
    if (v(0)) updateEarthquakes(v(0));
    if (v(1)) { updateSubsolar(v(1)); updateJellyBall(v(1)); updateTerminator(v(1)); }
    if (v(2)) updKp(v(2));
    if (v(3)) {
        updSW(v(3));
        if (v(3).current_speed) swSpeed = v(3).current_speed;
        if (v(3).current_density) swDensity = v(3).current_density;
        if (v(3).current_bz != null) updateMagnetosphereCompression(v(3).current_bz);
    }
    if (v(4)) updXRS(v(4));
    if (v(5)) updSun(v(5));
    if (v(6)) updLunar(v(6));
    if (v(7)) updCR(v(7));
    if (v(8)) updDst(v(8));
    if (v(9)) updateMagnetometers(v(9));
    if (v(10)) updFieldStrengths(v(10));
    if (v(11)) drawSeismogram(v(11));
    if (v(12)) updJellyBall(v(12));
    if (v(13)) updNeural(v(13));
    if (v(14)) { updGlobalCR(v(14)); if (v(14).global_mean != null) updateCRRate(v(14).global_mean); }
    if (v(15)) updTEC(v(15));
    if (v(16)) updPrecip(v(16));
    if (v(17)) updLightning(v(17));
    // Update magnetosphere storm visualization from Kp + Dst
    const kpVal = v(2)?.current ?? currentKp;
    const dstVal = v(8)?.current ?? currentDst;
    updateStormLevel(kpVal, dstVal);
}
poll();
setInterval(poll, POLL);

// ============================================================
// SOLAR MONITOR INTEGRATION
// ============================================================

const DET_NAMES = ['zscore', 'cusum', 'hardness', 'rate', 'multichannel', 'proton', 'criticality'];
function scoreColor(score) { return score < 0.3 ? '#44ff44' : score < 0.5 ? '#aaff44' : score < 0.7 ? '#ffaa44' : '#ff4444'; }

function updDetectors(data) {
    if (!data || data.error) return;

    // Solar monitor returns: { fused_score, alert, raw_scores: [{name, raw_score, percentile_rank}], detector_agreement }
    // OR from /status: { fusion_diagnostics: { fused_score, raw_scores, detector_agreement } }
    const diag = data.fusion_diagnostics || data;
    const detectors = diag.raw_scores || diag.detectors || [];

    if (Array.isArray(detectors)) {
        detectors.forEach(d => {
            const name = (d.name || '').toLowerCase().replace(/[_\s-]/g, '');
            const matchName = DET_NAMES.find(n => name.includes(n)) || name;
            // Use percentile_rank for bar (0-1 scale), raw_score for tooltip
            const val = d.percentile_rank ?? d.score ?? d.raw_score ?? null;
            const bar = document.getElementById(`det-${matchName}`);
            const scoreEl = document.getElementById(`ds-${matchName}`);
            if (bar && val != null) {
                const pct = Math.min(100, val * 100);
                bar.style.width = pct + '%';
                bar.style.background = scoreColor(val);
            }
            if (scoreEl && val != null) scoreEl.textContent = val.toFixed(2);
        });
    }

    const fused = diag.fused_score ?? data.fused_score ?? data.fused_flare_score ?? data.fused ?? null;
    if (fused != null) {
        const fill = document.getElementById('fused-fill'), label = document.getElementById('fused-label');
        if (fill) { fill.style.width = Math.min(100, fused * 100) + '%'; fill.style.background = scoreColor(fused); }
        if (label) label.textContent = `FUSED: ${fused.toFixed(3)}`;
        const stF = document.getElementById('st-fused');
        if (stF) { stF.textContent = fused.toFixed(2); stF.className = 'v ' + (fused < 0.3 ? 'g' : fused < 0.7 ? '' : 'w'); }
    }

    // detector_agreement can be at top level (/status) or inside diagnostics (/detectors)
    const agree = data.detector_agreement ?? diag.detector_agreement ?? data.agreement ?? null;
    if (agree != null) { const el = document.getElementById('det-agreement'); if (el) el.textContent = agree; }
}

function updEscalation(data) {
    if (!data || data.error) return;
    // Solar monitor returns: { level: "Quiet", level_label: "QUIET", peak_fused, hardness_spikes_in_window }
    const level = (data.level_label || data.level || data.state || 'quiet').toLowerCase();
    const el = document.getElementById('esc-state');
    if (el) { el.textContent = level.toUpperCase(); el.className = 'esc-badge esc-' + level; }
    const stE = document.getElementById('st-esc');
    if (stE) { stE.textContent = level.toUpperCase(); stE.className = 'v ' + (level === 'quiet' ? 'g' : level === 'flare' ? 'w' : ''); }
    const detail = document.getElementById('esc-detail');
    if (detail) {
        const spikes = data.hardness_spikes_in_window ?? data.hardness_spike_count ?? '--';
        const peak = data.peak_fused ?? data.peak ?? '--';
        detail.textContent = `Spikes: ${spikes} | Peak: ${typeof peak === 'number' ? peak.toFixed(2) : peak}`;
    }
}

const PW_NAMES = ['forbush', 'heep', 'ssc', 'mansurov', 'lunar'];
const PW_COLORS = { forbush: '#6644ff', heep: '#44ffaa', ssc: '#ff44aa', mansurov: '#ffff44', lunar: '#888' };

function updPathways(data) {
    if (!data || data.error) return;
    // Solar monitor returns: array of {name, score, effect, active, details}
    // OR from /status: { stressor: { pathways: [...], total } }
    const stressor = data.stressor || data;
    const pathways = stressor.pathways || (Array.isArray(data) ? data : []);
    let totalStress = 0;

    const pwList = Array.isArray(pathways) ? pathways : Object.values(pathways);
    pwList.forEach(pw => {
        // Match "Forbush Chain" -> forbush, "SSC Telluric" -> ssc, "Lunar Tidal" -> lunar
        const name = (pw.name || pw.pathway || '').toLowerCase().replace(/[_\s-]/g, '');
        const matchName = PW_NAMES.find(n => name.includes(n));
        if (!matchName) return;
        const score = pw.score ?? pw.value ?? 0;
        const effect = (pw.effect || pw.direction || '').toLowerCase();
        const isSuppression = effect.includes('suppress');
        const bar = document.getElementById(`pw-${matchName}`);
        const dirEl = document.getElementById(`pwd-${matchName}`);
        const scoreEl = document.getElementById(`pws-${matchName}`);
        if (bar) {
            bar.style.width = Math.min(100, Math.abs(score) * 100) + '%';
            bar.style.background = pw.active ? PW_COLORS[matchName] || '#888' : '#333';
        }
        if (dirEl) {
            dirEl.textContent = isSuppression ? '-' : '+';
            dirEl.style.color = isSuppression ? '#f44' : '#4f4';
        }
        if (scoreEl) scoreEl.textContent = Math.abs(score).toFixed(2);
        totalStress += isSuppression ? -Math.abs(score) : Math.abs(score);
    });

    const stressIdx = stressor.total ?? data.total_stress ?? data.stressor_index ?? totalStress;
    const stressEl = document.getElementById('stress-val');
    if (stressEl) {
        const val = typeof stressIdx === 'number' ? stressIdx : totalStress;
        stressEl.textContent = (val >= 0 ? '+' : '') + val.toFixed(2);
        stressEl.style.color = val > 0.3 ? '#f44' : val > 0 ? '#ff4' : '#4f4';
    }
}

// SSE Streams
let solarConnected = false;
function connectSSE() {
    try {
        const sse = new EventSource(`${SOLAR_API}/metrics`);
        sse.onopen = () => {
            solarConnected = true;
            const dot = document.getElementById('solar-conn'); if (dot) dot.className = 'conn-dot live';
            const st = document.getElementById('solar-status'); if (st) st.textContent = '(LIVE)';
        };
        sse.onmessage = e => {
            try {
                const d = JSON.parse(e.data);
                // Metrics SSE can send full status snapshots
                if (d.fusion_diagnostics || d.fused_score != null || d.raw_scores) updDetectors(d);
                if (d.escalation) updEscalation(d.escalation);
                else if (d.level || d.level_label) updEscalation(d);
                if (d.stressor?.pathways) updPathways(d.stressor);
                else if (d.pathways || Array.isArray(d)) updPathways(d);
                if (d.feeds?.imf_bz != null) updateMagnetosphereCompression(d.feeds.imf_bz);
            } catch (_) { }
        };
        sse.onerror = () => {
            solarConnected = false;
            const dot = document.getElementById('solar-conn'); if (dot) dot.className = 'conn-dot dead';
            const st = document.getElementById('solar-status'); if (st) st.textContent = '(polling)';
        };
    } catch (_) { }
    try {
        const alerts = new EventSource(`${SOLAR_API}/alerts`);
        alerts.onmessage = e => {
            try {
                const a = JSON.parse(e.data);
                const banner = document.getElementById('alert-banner');
                if (!banner) return;
                const type = a.type || a.kind || '', msg = a.message || a.msg || JSON.stringify(a);
                banner.textContent = `${type.toUpperCase()}: ${msg}`;
                banner.style.display = 'block';
                banner.style.background = type.includes('flare') ? 'rgba(255,50,50,0.95)' : 'rgba(40,160,255,0.95)';
                setTimeout(() => { banner.style.display = 'none'; }, 15000);
            } catch (_) { }
        };
    } catch (_) { }
}
connectSSE();

// Solar monitor polling fallback — uses /status for all-in-one, falls back to individual endpoints
async function pollSolar() {
    try {
        // Try /status first (single request with everything)
        const [statusResp, feedsResp] = await Promise.all([
            fetch(`${SOLAR_API}/status`),
            fetch(`${SOLAR_API}/feeds`),
        ]);
        const status = await statusResp.json();
        const feeds = await feedsResp.json();
        if (status && !status.error) {
            updDetectors(status);
            if (status.escalation) updEscalation(status.escalation);
            if (status.stressor) updPathways(status.stressor);
            // Feed live particle data to solar wind visualization
            if (feeds && !feeds.error) updateSolarWindData(feeds);
            // Extract proton detector score for SEP particles
            const protonDet = status.fusion_diagnostics?.raw_scores?.find(d => d.name === 'proton');
            if (protonDet) swProtonScore = protonDet.percentile_rank;
            const dot = document.getElementById('solar-conn');
            if (dot) dot.className = 'conn-dot live';
            const st = document.getElementById('solar-status');
            if (st) st.textContent = solarConnected ? '(LIVE)' : '(polling)';
            return;
        }
    } catch (_) { }

    // Fallback: individual endpoints
    try {
        const [detR, escR, pwR] = await Promise.allSettled([
            fetch(`${SOLAR_API}/detectors`).then(r => r.json()),
            fetch(`${SOLAR_API}/escalation`).then(r => r.json()),
            fetch(`${SOLAR_API}/pathways`).then(r => r.json()),
        ]);
        if (detR.value && !detR.value.error) updDetectors(detR.value);
        if (escR.value && !escR.value.error) updEscalation(escR.value);
        if (pwR.value && !pwR.value.error) updPathways(pwR.value);
    } catch (_) { }
}
pollSolar();
setInterval(pollSolar, POLL);

// ============================================================
// THREE.JS PHYSICS VISUALIZATIONS
// ============================================================

// ===== SUN OBJECT =====
const SUN_X = 10; // far enough to be clearly separate from Earth
fixedLayers.add('sun-object');
(function buildSun() {
    const layer = getLayer('sun-object');
    // Core — bright enough to trigger bloom
    const sunMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 32, 32),
        new THREE.MeshBasicMaterial({ color: 0xffdd55 })
    );
    sunMesh.position.set(SUN_X, 0, 0);
    layer.add(sunMesh);
    // Corona glow
    const corona = new THREE.Mesh(
        new THREE.SphereGeometry(0.9, 32, 32),
        new THREE.MeshBasicMaterial({ color: 0xffaa22, transparent: true, opacity: 0.15, side: THREE.BackSide, depthWrite: false })
    );
    corona.position.set(SUN_X, 0, 0);
    layer.add(corona);
})();

// ===== MAGNETOSPHERE =====
let magnetoCompression = 1.0;
let stormLevel = 0;
let currentBz = 0;
let currentKp = 2;
let currentDst = 0;
let reconnectionPositions = null, reconnectionPts = null;

// Bow shock standoff — particles deflect here, visuals drawn here
const BOW_STANDOFF = () => 1.6 * magnetoCompression + 0.2;

function buildMagnetosphere() {
    clearLayer('magnetosphere');
    const layer = getLayer('magnetosphere');
    const comp = magnetoCompression;
    const storm = stormLevel;
    const S = 0.5; // scale factor for field lines (keeps them close to globe)

    // --- DIPOLE FIELD LINES (bright cyan, 2 meridional planes) ---
    // Like the reference: clean arcs from pole to pole, compressed sunward
    const cyan = new THREE.Color(0x00ccff);
    const cyanDim = new THREE.Color(0x2288aa);

    for (let p = 0; p < 4; p++) {
        const phi = (p / 4) * Math.PI; // 4 half-planes = 8 visual planes
        for (let s = 0; s < 5; s++) {
            const L = 2.0 + s * 0.7;
            const pts = [];
            for (let j = 0; j <= 80; j++) {
                const theta = (j / 80) * Math.PI;
                const r = L * Math.sin(theta) * Math.sin(theta);
                let x = r * Math.sin(theta) * Math.cos(phi);
                let y = r * Math.cos(theta);
                let z = r * Math.sin(theta) * Math.sin(phi);

                // Compress sunward side
                if (x > 0) x *= comp * 0.7;
                // Stretch nightside into tail
                if (x < 0) {
                    x *= 1 + (1 - comp) * 0.8 + s * 0.15;
                    y *= 1 - s * 0.04 * Math.min(1, Math.abs(x * S));
                }
                pts.push(new THREE.Vector3(x * S, y * S, z * S));
            }
            const color = s < 2 ? cyan : cyanDim;
            const opacity = s < 2 ? 0.5 : 0.25;
            layer.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(pts),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity })
            ));
        }
    }

    // --- BOW SHOCK (bright line arc at standoff distance, not a surface) ---
    const bowR = BOW_STANDOFF();
    const bowColor = storm > 0.5 ? 0xff6644 : 0x44ddff;
    // Main arc — parabolic cross-section in 4 meridional planes
    for (let m = 0; m < 4; m++) {
        const angle = (m / 4) * Math.PI;
        const pts = [];
        for (let i = 0; i <= 40; i++) {
            const t = (i / 40) * Math.PI * 0.5; // 0 to 90 deg from nose
            const rCross = bowR * Math.sin(t);
            const xBow = bowR * Math.cos(t);
            pts.push(new THREE.Vector3(xBow, rCross * Math.sin(angle), rCross * Math.cos(angle)));
        }
        layer.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: bowColor, transparent: true, opacity: 0.4 })
        ));
    }
    // Cross-ring at nose
    const noseRingPts = [];
    for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        const rr = bowR * 0.3;
        noseRingPts.push(new THREE.Vector3(bowR * 0.95, rr * Math.sin(a), rr * Math.cos(a)));
    }
    layer.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(noseRingPts),
        new THREE.LineBasicMaterial({ color: bowColor, transparent: true, opacity: 0.35 })
    ));

    // --- AURORA OVALS ---
    const auroraLat = 70 - storm * 12;
    for (const isNorth of [true, false]) {
        const pts = [];
        const lat = isNorth ? auroraLat : -auroraLat;
        for (let i = 0; i <= 80; i++) pts.push(ll2v(lat, (i / 80) * 360 - 180, R * 1.008));
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: 0x44ff88, transparent: true, opacity: 0.15 + storm * 0.5 })
        );
        line.name = isNorth ? 'aurora-n-1' : 'aurora-s-1';
        layer.add(line);
    }

    // --- RING CURRENT (torus) ---
    const rcIntensity = Math.min(1, Math.abs(currentDst) / 100);
    const rcMesh = new THREE.Mesh(
        new THREE.TorusGeometry(0.25, 0.02 + storm * 0.02, 12, 48),
        new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(0.08, 0.9, 0.4 + rcIntensity * 0.3), transparent: true, opacity: 0.06 + rcIntensity * 0.15, depthWrite: false })
    );
    rcMesh.rotation.x = Math.PI / 2;
    layer.add(rcMesh);

    // --- RECONNECTION (Bz southward) ---
    if (currentBz < -3) {
        const nParts = 50;
        reconnectionPositions = new Float32Array(nParts * 3);
        for (let i = 0; i < nParts; i++) {
            reconnectionPositions[i * 3] = bowR * 0.9 + Math.random() * 0.1;
            reconnectionPositions[i * 3 + 1] = (Math.random() - 0.5) * 0.15;
            reconnectionPositions[i * 3 + 2] = (Math.random() - 0.5) * 0.06;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(reconnectionPositions, 3));
        reconnectionPts = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xff4488, size: 0.008, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false }));
        layer.add(reconnectionPts);
    } else { reconnectionPositions = null; reconnectionPts = null; }
}

function animateReconnection() {
    if (!reconnectionPositions) return;
    for (let i = 0; i < reconnectionPositions.length / 3; i++) {
        const ix = i * 3;
        const dir = i % 2 === 0 ? 1 : -1;
        reconnectionPositions[ix] -= 0.003;
        reconnectionPositions[ix + 1] += dir * 0.004;
        if (Math.abs(reconnectionPositions[ix + 1]) > 0.5 || reconnectionPositions[ix] < -0.4) {
            reconnectionPositions[ix] = 0.5 + Math.random() * 0.15;
            reconnectionPositions[ix + 1] = (Math.random() - 0.5) * 0.1;
            reconnectionPositions[ix + 2] = (Math.random() - 0.5) * 0.08;
        }
    }
    if (reconnectionPts) reconnectionPts.geometry.attributes.position.needsUpdate = true;
}

function updateMagnetosphereCompression(bz) {
    currentBz = bz;
    const c = bz < 0 ? Math.max(0.4, 1 + bz / 30) : 1.0;
    if (Math.abs(c - magnetoCompression) > 0.02) { magnetoCompression = c; buildMagnetosphere(); }
}

function updateStormLevel(kp, dst) {
    currentKp = kp;
    currentDst = dst;
    const kpLevel = Math.max(0, (kp - 3) / 6);
    const dstLevel = Math.min(1, Math.max(0, Math.abs(dst) / 150));
    const newLevel = Math.max(kpLevel, dstLevel);
    if (Math.abs(newLevel - stormLevel) > 0.05) {
        stormLevel = newLevel;
        buildMagnetosphere();
        const si = document.getElementById('storm-indicator');
        if (si) {
            si.textContent = stormLevel > 0.7 ? 'EXTREME' : stormLevel > 0.5 ? 'MAJOR' : stormLevel > 0.3 ? 'MODERATE' : stormLevel > 0.1 ? 'MINOR' : 'QUIET';
            si.style.color = stormLevel > 0.5 ? '#f44' : stormLevel > 0.2 ? '#ff4' : '#4f4';
        }
        const al = document.getElementById('aurora-lat');
        if (al) { al.textContent = Math.round(70 - stormLevel * 12) + '\u00b0'; al.style.color = stormLevel > 0.3 ? '#88ffaa' : '#44ff88'; }
        const rc = document.getElementById('ring-current-val');
        if (rc) { rc.textContent = `${dst} nT`; rc.style.color = dst < -100 ? '#f44' : dst < -50 ? '#ffaa44' : '#4f4'; }
    }
}

buildMagnetosphere();

// ===== SOLAR WIND PARTICLES (data-driven) =====
// Three populations: protons (yellow), electrons (cyan), SEP high-energy (red)
// Density, speed, color, and count respond to live solar monitor data
const SW_MAX = 800;
let swParticles = null, swPositions = null, swVelocities = null, swColors = null;
let swSpeed = 400;        // km/s from live data
let swDensity = 5;        // /cc from live data
let swElectronFlux = 100; // pfu from live data
let swProtonScore = 0;    // 0-1 from detector

function initParticle(i, type) {
    // type: 0=proton (bulk), 1=electron (fast), 2=SEP (high energy)
    const ix = i * 3;
    // Spread across the sun-Earth corridor
    const spread = type === 2 ? 0.3 : 0.7;
    swPositions[ix] = 2 + Math.random() * 7;
    swPositions[ix + 1] = (Math.random() - 0.5) * spread;
    swPositions[ix + 2] = (Math.random() - 0.5) * spread;
    // Speed: protons=nominal, electrons=1.5x, SEP=2-3x
    const baseSpeed = 0.01 + Math.random() * 0.005;
    const speedMult = type === 2 ? 2.5 : type === 1 ? 1.5 : 1.0;
    swVelocities[ix] = -baseSpeed * speedMult;
    swVelocities[ix + 1] = (Math.random() - 0.5) * 0.001;
    swVelocities[ix + 2] = (Math.random() - 0.5) * 0.001;
    // Color by type
    if (type === 2) {
        // SEP: hot red-pink
        swColors[ix] = 1.0; swColors[ix + 1] = 0.2; swColors[ix + 2] = 0.15;
    } else if (type === 1) {
        // Electron: bright cyan
        swColors[ix] = 0.3; swColors[ix + 1] = 0.85; swColors[ix + 2] = 1.0;
    } else {
        // Proton: bright yellow-white
        swColors[ix] = 1.0; swColors[ix + 1] = 0.85 + Math.random() * 0.15; swColors[ix + 2] = 0.4 + Math.random() * 0.3;
    }
}

function buildSolarWind() {
    clearLayer('solar-wind');
    const layer = getLayer('solar-wind');
    const geo = new THREE.BufferGeometry();
    swPositions = new Float32Array(SW_MAX * 3);
    swVelocities = new Float32Array(SW_MAX * 3);
    swColors = new Float32Array(SW_MAX * 3);

    // Population split based on current data
    const electronFrac = Math.min(0.4, swElectronFlux / 5000);  // up to 40% at 5000 pfu
    const sepFrac = Math.min(0.2, swProtonScore * 0.2);          // up to 20% when proton detector fires
    const protonFrac = 1 - electronFrac - sepFrac;

    // Active particle count: always visible base, scales up with density
    const activeCount = Math.min(SW_MAX, Math.floor(400 + swDensity * 50));  // 400 base, +50 per /cc

    for (let i = 0; i < SW_MAX; i++) {
        const t = i / SW_MAX;
        const type = t < protonFrac ? 0 : t < protonFrac + electronFrac ? 1 : 2;
        initParticle(i, type);
        // Hide excess particles far away
        if (i >= activeCount) {
            swPositions[i * 3] = 99;
            swPositions[i * 3 + 1] = 99;
            swPositions[i * 3 + 2] = 99;
        }
    }

    geo.setAttribute('position', new THREE.BufferAttribute(swPositions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(swColors, 3));
    swParticles = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.018, vertexColors: true, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false,
        sizeAttenuation: true,
    }));
    layer.add(swParticles);
}

function animateSolarWind() {
    if (!swPositions) return;
    const sf = swSpeed / 400;
    const activeCount = Math.min(SW_MAX, Math.floor(400 + swDensity * 50));
    const electronFrac = Math.min(0.4, swElectronFlux / 5000);
    const sepFrac = Math.min(0.2, swProtonScore * 0.2);
    const protonFrac = 1 - electronFrac - sepFrac;

    for (let i = 0; i < SW_MAX; i++) {
        if (i >= activeCount) continue;  // skip hidden particles
        const ix = i * 3;
        const t = i / SW_MAX;
        const type = t < protonFrac ? 0 : t < protonFrac + electronFrac ? 1 : 2;
        const speedMult = type === 2 ? 2.5 : type === 1 ? 1.5 : 1.0;

        swPositions[ix] += swVelocities[ix] * sf;
        swPositions[ix + 1] += swVelocities[ix + 1];
        swPositions[ix + 2] += swVelocities[ix + 2];

        // Deflect around magnetosphere
        const dist = Math.sqrt(swPositions[ix] ** 2 + swPositions[ix + 1] ** 2 + swPositions[ix + 2] ** 2);
        const bowDist = BOW_STANDOFF();
        if (dist < bowDist) {
            const nx = swPositions[ix] / dist, ny = swPositions[ix + 1] / dist, nz = swPositions[ix + 2] / dist;
            swVelocities[ix] += nx * 0.003;
            swVelocities[ix + 1] += ny * 0.003;
            swVelocities[ix + 2] += nz * 0.003;
        }

        // Reset when past Earth or too far — respawn from sun side
        if (swPositions[ix] < -2 || dist > 12) {
            const spread = type === 2 ? 0.3 : 0.7;
            swPositions[ix] = 6 + Math.random() * 3;
            swPositions[ix + 1] = (Math.random() - 0.5) * spread;
            swPositions[ix + 2] = (Math.random() - 0.5) * spread;
            const baseSpeed = 0.01 + Math.random() * 0.005;
            const sm = type === 2 ? 2.5 : type === 1 ? 1.5 : 1.0;
            swVelocities[ix] = -baseSpeed * sm;
            swVelocities[ix + 1] = (Math.random() - 0.5) * 0.001;
            swVelocities[ix + 2] = (Math.random() - 0.5) * 0.001;
        }
    }

    // Update colors in real-time (population fractions shift with data)
    if (swParticles) {
        swParticles.geometry.attributes.position.needsUpdate = true;
        swParticles.geometry.attributes.color.needsUpdate = true;
    }
}

// Update particle properties from live solar monitor feeds
function updateSolarWindData(feeds) {
    if (!feeds) return;
    const sw = feeds.solar_wind_latest || feeds;
    if (sw.speed != null) swSpeed = sw.speed;
    if (sw.density != null) swDensity = sw.density;
    const el = feeds.electron_latest || {};
    if (el.flux != null) swElectronFlux = el.flux;
}

buildSolarWind();

// Comet
const cometGroup = getLayer('comet');
let cometMesh = null, cometTail = null, cometAngle = 0;
function buildComet() {
    clearLayer('comet'); const layer = getLayer('comet');
    cometMesh = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), new THREE.MeshBasicMaterial({ color: 0x88ffff }));
    layer.add(cometMesh);
    cometMesh.add(new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 16), new THREE.MeshBasicMaterial({ color: 0x44ddff, transparent: true, opacity: 0.2, depthWrite: false })));
    const tailPts = []; for (let i = 0; i < 60; i++) tailPts.push(new THREE.Vector3(0, 0, 0));
    cometTail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(tailPts), new THREE.LineBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.4 }));
    layer.add(cometTail);
    const ionPts = []; for (let i = 0; i < 40; i++) ionPts.push(new THREE.Vector3(0, 0, 0));
    const ionTail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ionPts), new THREE.LineBasicMaterial({ color: 0x4466ff, transparent: true, opacity: 0.25 }));
    ionTail.name = 'ion-tail'; layer.add(ionTail);
}
function animateComet() {
    if (!cometMesh) return;
    cometAngle += 0.0008;
    const orbitR = 3.5 - 1.0 * Math.sin(cometAngle * 0.5);
    cometMesh.position.set(orbitR * Math.cos(cometAngle) + 4, 0.5 * Math.sin(cometAngle * 0.7), orbitR * Math.sin(cometAngle) * 0.3);
    const sunDir = new THREE.Vector3(4, 0, 0).sub(cometMesh.position).normalize();
    const tp = cometTail.geometry.attributes.position;
    for (let i = 0; i < 60; i++) {
        const t = i / 60, len = t * 1.5;
        tp.setXYZ(i, cometMesh.position.x - sunDir.x * len + Math.sin(t * 2) * t * 0.15, cometMesh.position.y - sunDir.y * len + Math.cos(t * 3) * t * 0.08, cometMesh.position.z - sunDir.z * len + t * 0.1);
    }
    tp.needsUpdate = true;
    const ion = cometGroup.getObjectByName('ion-tail');
    if (ion) { const ip = ion.geometry.attributes.position; for (let i = 0; i < 40; i++) { const t = i / 40, len = t * 2; ip.setXYZ(i, cometMesh.position.x - sunDir.x * len, cometMesh.position.y - sunDir.y * len, cometMesh.position.z - sunDir.z * len); } ip.needsUpdate = true; }
}
buildComet();

// ============================================================
// PHYSICAL SIMULATION LAYERS
// ============================================================

// --- COSMIC RAY (single particle, data-driven rate) ---
// One GCR at a time. Rate = measured neutron monitor count rate.
// During Forbush decrease: rate drops, fewer events.
let crActive = false;
let crProgress = 0;
let crCooldown = 0;
let crRate = 120; // frames between events (driven by live CR data)
const crTrailPts = 20;
const crPositions = new Float32Array(crTrailPts * 3);
const crGeo = new THREE.BufferGeometry();
crGeo.setAttribute('position', new THREE.BufferAttribute(crPositions, 3));
const crLine = new THREE.Line(crGeo, new THREE.LineBasicMaterial({
    color: 0xaaddff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
}));
scene.add(crLine);

// GCR trajectory: isotropic entry, curved by geomagnetic field
let crEntry = new THREE.Vector3(), crDir = new THREE.Vector3(), crCharge = 1;

function spawnGCR() {
    // Enter from random direction on a sphere of radius ~5
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    crEntry.set(5 * Math.sin(phi) * Math.cos(theta), 5 * Math.cos(phi), 5 * Math.sin(phi) * Math.sin(theta));
    // Aim roughly toward Earth with some spread
    crDir.copy(crEntry).negate().normalize();
    crDir.x += (Math.random() - 0.5) * 0.4;
    crDir.y += (Math.random() - 0.5) * 0.4;
    crDir.z += (Math.random() - 0.5) * 0.4;
    crDir.normalize();
    crCharge = Math.random() > 0.5 ? 1 : -1;
    crProgress = 0;
    crActive = true;
    crLine.material.opacity = 0.6;
}

function animateGCR() {
    if (!crActive) {
        crCooldown--;
        if (crCooldown <= 0) {
            spawnGCR();
            crCooldown = crRate;
        }
        crLine.material.opacity *= 0.93; // fade out trail
        crGeo.attributes.position.needsUpdate = true;
        return;
    }

    crProgress++;
    const pos = crEntry.clone().addScaledVector(crDir, crProgress * 0.08);

    // Geomagnetic deflection: Lorentz force F = qv x B
    // Simplified: deflect perpendicular to both velocity and radial
    const radial = pos.clone().normalize();
    const lorentz = new THREE.Vector3().crossVectors(crDir, radial).multiplyScalar(0.003 * crCharge / (pos.length() + 0.5));
    crDir.add(lorentz).normalize();

    // Update trail (shift positions back, add new head)
    for (let i = crTrailPts - 1; i > 0; i--) {
        crPositions[i * 3] = crPositions[(i - 1) * 3];
        crPositions[i * 3 + 1] = crPositions[(i - 1) * 3 + 1];
        crPositions[i * 3 + 2] = crPositions[(i - 1) * 3 + 2];
    }
    crPositions[0] = pos.x; crPositions[1] = pos.y; crPositions[2] = pos.z;
    crGeo.attributes.position.needsUpdate = true;

    // Terminate if hits atmosphere or escapes
    if (pos.length() < R * 1.05) {
        crActive = false; // absorbed by atmosphere -> shower
        crLine.material.color.set(0xffffff); // bright flash on impact
        setTimeout(() => crLine.material.color.set(0xaaddff), 200);
    }
    if (pos.length() > 8 || crProgress > 200) {
        crActive = false; // escaped or deflected away
    }
}

// Update CR rate from live data: higher count = more frequent events
function updateCRRate(crDeviation) {
    // Baseline: ~one every 2 seconds (120 frames at 60fps)
    // Forbush decrease (negative deviation): slower rate
    // Enhanced flux: faster rate
    crRate = Math.max(30, Math.floor(120 * (1 - crDeviation / 100 * 2)));
}

// --- IONOSPHERIC SHELL (Schumann cavity + telluric currents) ---
// At ~100km altitude (R * 1.016), not on the surface
const IONO_R = R * 1.016; // 100km scale height

// Schumann cavity shell with current-flow shader
const ionoGeo = new THREE.SphereGeometry(IONO_R, 64, 64);
const ionoMat = new THREE.ShaderMaterial({
    uniforms: {
        uTime: { value: 0 },
        uStorm: { value: 0 },
        uSchumannAmp: { value: 0.5 },
    },
    vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPos;
        varying vec2 vUv;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            vPos = position;
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
    fragmentShader: `
        uniform float uTime;
        uniform float uStorm;
        uniform float uSchumannAmp;
        varying vec3 vNormal;
        varying vec3 vPos;
        varying vec2 vUv;
        void main() {
            // Fresnel edge glow (ionospheric limb)
            float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 3.0);

            // Telluric current flow: horizontal waves at ionospheric altitude
            // Two perpendicular current systems (Sq + disturbed)
            float lat = asin(vPos.y / length(vPos));
            float lon = atan(vPos.z, vPos.x);
            float sqCurrent = sin(lat * 6.0 + uTime * 0.5) * cos(lon * 4.0 - uTime * 0.3);
            float distCurrent = sin(lat * 3.0 - uTime * 1.5) * uStorm;

            // Schumann resonance pulse (7.83 Hz scaled)
            float schumann = sin(uTime * 0.82) * 0.5 + 0.5; // ~7.8 visual Hz
            float f2 = sin(uTime * 1.50) * 0.3 + 0.3; // 2nd harmonic ~14.3 Hz
            float schumannTotal = (schumann + f2 * 0.5) * uSchumannAmp;

            // Combine: edge glow + current pattern + Schumann pulse
            vec3 quietColor = vec3(0.15, 0.6, 0.4);   // green ionosphere
            vec3 stormColor = vec3(0.8, 0.3, 0.15);    // orange during storms
            vec3 baseColor = mix(quietColor, stormColor, uStorm);

            float alpha = fresnel * 0.08
                        + abs(sqCurrent) * 0.015 * (1.0 + uStorm)
                        + abs(distCurrent) * 0.03
                        + schumannTotal * 0.01;
            alpha = clamp(alpha, 0.0, 0.15);

            gl_FragColor = vec4(baseColor + vec3(schumannTotal * 0.2), alpha);
        }`,
    transparent: true, side: THREE.FrontSide, depthWrite: false, blending: THREE.AdditiveBlending,
});
const ionoShell = new THREE.Mesh(ionoGeo, ionoMat);
ionoShell.name = 'ionosphere';
scene.add(ionoShell);
// Ionosphere rotates with Earth
rotatingLayers.add('ionosphere-dummy'); // placeholder so it's tracked

function animateIonosphere(frame) {
    ionoShell.rotation.y = earth.rotation.y; // sync with Earth
    ionoMat.uniforms.uTime.value = frame * 0.016;
    ionoMat.uniforms.uStorm.value = stormLevel;
}

// --- FIELD-ALIGNED CURRENT DOTS ---
// Charged particles flowing along field lines (region 1 + region 2 FAC)
// Rate scales with Kp — more FAC during storms
const FAC_COUNT = 30;
const facPositions = new Float32Array(FAC_COUNT * 3);
const facColors = new Float32Array(FAC_COUNT * 3);
const facGeo = new THREE.BufferGeometry();
facGeo.setAttribute('position', new THREE.BufferAttribute(facPositions, 3));
facGeo.setAttribute('color', new THREE.BufferAttribute(facColors, 3));
const facPts = new THREE.Points(facGeo, new THREE.PointsMaterial({
    size: 0.015, vertexColors: true, transparent: true, opacity: 0.8,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
}));
scene.add(facPts);

const facState = [];
for (let i = 0; i < FAC_COUNT; i++) {
    facState.push({
        phi: Math.random() * Math.PI * 2,
        L: 1.8 + Math.floor(Math.random() * 4) * 0.6,
        theta: Math.random() * Math.PI,
        speed: 0.015 + Math.random() * 0.025,
        dir: Math.random() > 0.5 ? 1 : -1,
        active: Math.random() < 0.3 + stormLevel * 0.5, // more active during storms
    });
}

function animateFAC(frame) {
    const S = 0.5;
    const activeThreshold = 0.3 + stormLevel * 0.5;
    for (let i = 0; i < FAC_COUNT; i++) {
        const f = facState[i];
        if (!f.active) {
            // Randomly activate based on storm level
            if (Math.random() < 0.002 * (1 + stormLevel * 5)) {
                f.active = true;
                f.theta = Math.random() * Math.PI;
                f.phi = Math.random() * Math.PI * 2;
            }
            facPositions[i * 3] = 99; facPositions[i * 3 + 1] = 99; facPositions[i * 3 + 2] = 99;
            continue;
        }

        f.theta += f.speed * f.dir;
        if (f.theta > Math.PI || f.theta < 0) {
            f.dir *= -1;
            f.theta = Math.max(0.01, Math.min(Math.PI - 0.01, f.theta));
            if (Math.random() > activeThreshold) f.active = false; // deactivate sometimes
        }

        const r = f.L * Math.sin(f.theta) * Math.sin(f.theta);
        let x = r * Math.sin(f.theta) * Math.cos(f.phi);
        let y = r * Math.cos(f.theta);
        let z = r * Math.sin(f.theta) * Math.sin(f.phi);
        if (x > 0) x *= magnetoCompression * 0.7;
        if (x < 0) x *= 1 + (1 - magnetoCompression) * 0.8;

        const ix = i * 3;
        facPositions[ix] = x * S;
        facPositions[ix + 1] = y * S;
        facPositions[ix + 2] = z * S;

        // Color: cyan flowing toward equator, magenta flowing toward poles
        const towardEquator = (f.dir > 0 && f.theta < Math.PI / 2) || (f.dir < 0 && f.theta > Math.PI / 2);
        facColors[ix] = towardEquator ? 0.3 : 0.8;
        facColors[ix + 1] = towardEquator ? 0.9 : 0.3;
        facColors[ix + 2] = towardEquator ? 1.0 : 0.9;
    }
    facGeo.attributes.position.needsUpdate = true;
    facGeo.attributes.color.needsUpdate = true;
}

// ===== ANIMATE =====
let frame = 0;
function animate() {
    requestAnimationFrame(animate);
    ctrl.update();
    frame++;
    const rot = 0.00015;
    earth.rotation.y += rot; wireframe.rotation.y += rot;
    for (const name of rotatingLayers) { if (layerGroups[name]) layerGroups[name].rotation.y += rot; }

    // Earthquake wave propagation
    if (frame % 30 === 0 && eqWaves.length > 0) animateWaves();

    // Subsolar pulse
    const ss = layerGroups['subsolar']?.getObjectByName('subsolar-pulse');
    if (ss) ss.scale.setScalar(1 + 0.3 * Math.sin(frame * 0.05));

    // Solar wind particles
    animateSolarWind();

    // Comet
    animateComet();

    // Reconnection jets
    if (currentBz < -3) animateReconnection();

    // Aurora ovals — multi-frequency pulse
    if (stormLevel > 0.05) {
        const ml = layerGroups['magnetosphere'];
        if (ml) ml.children.forEach(c => {
            if (c.name?.includes('aurora-')) {
                // Pulse with multiple frequencies (substorm-like)
                const base = 0.1 + stormLevel * 0.6;
                const pulse1 = Math.sin(frame * 0.03) * 0.15;       // slow breathing
                const pulse2 = Math.sin(frame * 0.11) * 0.08;       // faster flicker
                const pulse3 = Math.sin(frame * 0.31) * 0.04;       // rapid shimmer
                c.material.opacity = Math.max(0, base + pulse1 + pulse2 + pulse3);
            }
        });
    }

    // Galactic cosmic ray (single, data-driven rate)
    animateGCR();

    // Ionospheric shell (Schumann + telluric currents)
    animateIonosphere(frame);

    // Field-aligned currents (Kp-driven)
    animateFAC(frame);

    // Weather markers (thunder flicker)
    animateWeather(frame);

    composer.render();
}
animate();

// Resize
window.addEventListener('resize', () => {
    camera.aspect = box.clientWidth / box.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(box.clientWidth, box.clientHeight);
    composer.setSize(box.clientWidth, box.clientHeight);
});
