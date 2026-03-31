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
const ambientLight = new THREE.AmbientLight(0x444466, 0.8);
scene.add(ambientLight);
const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
sunLight.position.set(5, 3, 5);
scene.add(sunLight);

// ===== Earth globe =====
const earthGeom = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
const textureLoader = new THREE.TextureLoader();

// Use a simple color until texture loads
const earthMat = new THREE.MeshPhongMaterial({
    color: 0x1a3355,
    emissive: 0x0a1122,
    specular: 0x333333,
    shininess: 15,
});
const earth = new THREE.Mesh(earthGeom, earthMat);
scene.add(earth);

// Load earth texture
textureLoader.load(
    'https://unpkg.com/three-globe@2.31.1/example/img/earth-blue-marble.jpg',
    (texture) => {
        earthMat.map = texture;
        earthMat.color.set(0xffffff);
        earthMat.emissive.set(0x000000);
        earthMat.needsUpdate = true;
    },
    undefined,
    () => console.log('Earth texture failed to load, using fallback color')
);

// Atmosphere glow
const atmosGeom = new THREE.SphereGeometry(EARTH_RADIUS * 1.015, 64, 64);
const atmosMat = new THREE.MeshPhongMaterial({
    color: 0x4488ff,
    transparent: true,
    opacity: 0.08,
    side: THREE.BackSide,
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

// ===== Helper: lat/lon to 3D position =====
function latLonToVec3(lat, lon, radius = EARTH_RADIUS * 1.001) {
    const phi = (90 - lat) * Math.PI / 180;
    const theta = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(
        -radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta)
    );
}

// ===== Zone colors =====
const ZONE_COLORS = {
    eye: 0x4444ff,
    inner: 0x44ff44,
    wavefront: 0xff4444,
    outer: 0xffff44,
    far: 0x888888,
    antipodal: 0xcc88cc,
};

// ===== Data update functions =====

async function fetchJSON(endpoint) {
    try {
        const resp = await fetch(`${API_BASE}${endpoint}`);
        return await resp.json();
    } catch (e) {
        console.warn(`Fetch ${endpoint} failed:`, e);
        return null;
    }
}

function clearGroup(group) {
    while (group.children.length > 0) {
        const child = group.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        group.remove(child);
    }
}

function updateEarthquakes(data) {
    clearGroup(layers.earthquakes);
    if (!data?.earthquakes) return;

    data.earthquakes.forEach(eq => {
        const size = Math.max(0.005, (eq.mag - 3.5) * 0.004);
        const geom = new THREE.SphereGeometry(size, 8, 8);
        const color = ZONE_COLORS[eq.zone] || 0xffffff;
        const mat = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.8,
        });
        const mesh = new THREE.Mesh(geom, mat);
        const pos = latLonToVec3(eq.lat, eq.lon, EARTH_RADIUS * 1.003);
        mesh.position.copy(pos);
        mesh.userData = eq;
        layers.earthquakes.add(mesh);
    });

    document.getElementById('st-eqs').textContent = data.earthquakes.length;
}

function updateSubsolar(data) {
    clearGroup(layers.subsolar);
    clearGroup(layers['jelly-ball']);
    if (!data) return;

    // Subsolar point marker
    const ssGeom = new THREE.SphereGeometry(0.015, 12, 12);
    const ssMat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    const ssMesh = new THREE.Mesh(ssGeom, ssMat);
    ssMesh.position.copy(latLonToVec3(data.lat, data.lon, EARTH_RADIUS * 1.005));
    layers.subsolar.add(ssMesh);

    // Jelly Ball zone rings
    if (data.zones) {
        data.zones.forEach(zone => {
            const ringRadius = Math.sin(zone.radius_deg * Math.PI / 180) * EARTH_RADIUS;
            const ringGeom = new THREE.RingGeometry(ringRadius - 0.002, ringRadius + 0.002, 64);
            const ringMat = new THREE.MeshBasicMaterial({
                color: parseInt(zone.color.replace('#', '0x')),
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide,
            });
            const ring = new THREE.Mesh(ringGeom, ringMat);

            // Orient ring around subsolar point
            const ssPos = latLonToVec3(data.lat, data.lon);
            ring.position.copy(ssPos.clone().multiplyScalar(1.002));
            ring.lookAt(0, 0, 0);

            layers['jelly-ball'].add(ring);
        });
    }
}

function updateKp(data) {
    if (!data?.current) return;
    const el = document.getElementById('kp-metric');
    const kp = data.current;
    el.textContent = `Kp ${kp.toFixed(0)}`;
    el.className = 'metric ' + (kp < 4 ? 'quiet' : kp < 6 ? 'active' : 'storm');
    document.getElementById('st-kp').textContent = kp.toFixed(0);
    document.getElementById('st-kp').className = 'value ' + (kp < 4 ? 'ok' : kp < 6 ? '' : 'warn');
}

function updateSolarWind(data) {
    if (!data) return;
    if (data.current_bz != null) {
        const el = document.getElementById('st-bz');
        el.textContent = `${data.current_bz.toFixed(1)} nT`;
        el.className = 'value ' + (data.current_bz < -10 ? 'warn' : 'ok');
    }
    if (data.current_speed != null) {
        const el = document.getElementById('st-vsw');
        el.textContent = `${data.current_speed.toFixed(0)} km/s`;
        el.className = 'value ' + (data.current_speed > 600 ? 'warn' : 'ok');
    }
}

function updateXRS(data) {
    if (!data) return;
    if (data.current_flux) {
        const f = data.current_flux;
        let cls = f >= 1e-4 ? `X${(f/1e-4).toFixed(1)}` :
                  f >= 1e-5 ? `M${(f/1e-5).toFixed(1)}` :
                  f >= 1e-6 ? `C${(f/1e-6).toFixed(1)}` : `B`;
        document.getElementById('st-xrs').textContent = cls;
    }

    // Order parameter
    const stateEl = document.getElementById('op-state');
    stateEl.textContent = data.state || 'UNKNOWN';
    stateEl.className = 'state ' + (data.state === 'FALLING' ? 'falling' :
                                     data.state === 'RISING' ? 'rising' : 'stable');
}

function updateSun(data) {
    if (!data?.images) return;
    window._sunImages = data.images;
    // Default to AIA 193
    const img = document.getElementById('sun-image');
    if (!img.src || img.src === window.location.href) {
        img.src = data.images.aia_193;
    }
}

function updateLunar(data) {
    if (!data) return;
    document.getElementById('lunar-metric').textContent = `${data.name} (${data.illumination}%)`;
    document.getElementById('lunar-detail').textContent =
        `Tidal force: ${data.tidal_force.toFixed(3)} | dF/dt: ${data.tidal_rate.toFixed(3)} | Full in ${data.days_to_full}d`;
    document.getElementById('st-moon').textContent = `${data.illumination}%`;
}

function updateCosmicRays(data) {
    if (!data?.stations) return;
    const stations = Object.keys(data.stations);
    if (stations.length === 0) return;
    const avg_dev = stations.reduce((s, k) => s + data.stations[k].deviation_pct, 0) / stations.length;
    const el = document.getElementById('cr-metric');
    el.textContent = `${avg_dev > 0 ? '+' : ''}${avg_dev.toFixed(1)}%`;
    el.className = 'metric ' + (data.forbush_detected ? 'storm' : 'quiet');
    document.getElementById('cr-detail').textContent =
        data.forbush_detected ? 'FORBUSH DECREASE DETECTED' : `${stations.length} stations, no Forbush`;
}

// ===== Sun image selector =====
document.querySelectorAll('#sun-selector button').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#sun-selector button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const key = btn.dataset.img;
        if (window._sunImages?.[key]) {
            document.getElementById('sun-image').src = window._sunImages[key];
        }
    });
});

// ===== Layer toggle handlers =====
document.querySelectorAll('.layer-toggle input').forEach(input => {
    input.addEventListener('change', () => {
        const layer = layers[input.dataset.layer];
        if (layer) layer.visible = input.checked;
    });
});

// ===== Clock =====
function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent =
        now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

// ===== Main data poll =====
async function pollData() {
    const [eqs, ss, kp, sw, xrs, sun, lunar, cr] = await Promise.all([
        fetchJSON('/earthquakes'),
        fetchJSON('/subsolar'),
        fetchJSON('/kp'),
        fetchJSON('/solar_wind'),
        fetchJSON('/xrs'),
        fetchJSON('/sun'),
        fetchJSON('/lunar'),
        fetchJSON('/cosmic_rays'),
    ]);

    updateEarthquakes(eqs);
    updateSubsolar(ss);
    updateKp(kp);
    updateSolarWind(sw);
    updateXRS(xrs);
    updateSun(sun);
    updateLunar(lunar);
    updateCosmicRays(cr);
}

pollData();
setInterval(pollData, POLL_INTERVAL);

// ===== Animation loop =====
function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Slow earth rotation
    earth.rotation.y += 0.0001;
    layers.earthquakes.rotation.y += 0.0001;
    layers.subsolar.rotation.y += 0.0001;
    layers['jelly-ball'].rotation.y += 0.0001;

    renderer.render(scene, camera);
}
animate();

// ===== Resize handler =====
window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});
