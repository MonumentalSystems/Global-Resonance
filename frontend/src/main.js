// ============================================================
// GLOBAL RESONANCE — CesiumJS Globe + Three.js Space Physics
// ============================================================
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
// Three.js kept for future magnetosphere/solar wind overlay
import * as THREE from 'three';

// Same-origin '/api' works everywhere: the Vite dev server proxies /api -> :8000
// (see vite.config.js), and in production FastAPI serves the API and this
// frontend from the same origin. Override with window.API_BASE if ever needed.
const API = window.API_BASE || '/api';
const SOLAR_API = window.SOLAR_MONITOR_URL || API + '/solar';
const POLL = 30_000;

// ============================================================
// MATERIALS: ANIMATED FLOW LINES (wind, currents)
// ============================================================
const PolylineTrailMaterialProperty = (function () {
    function PolylineTrailMaterialProperty(color, duration) {
        this._definitionChanged = new Cesium.Event();
        this._color = undefined;
        this._colorSubscription = undefined;
        this.color = color;
        this.duration = duration || 2000;
        this._time = Date.now();
    }

    Object.defineProperties(PolylineTrailMaterialProperty.prototype, {
        isConstant: { get: () => false },
        definitionChanged: { get: function () { return this._definitionChanged; } },
        color: Cesium.createPropertyDescriptor('color'),
    });

    PolylineTrailMaterialProperty.prototype.getType = function () {
        return 'PolylineTrail';
    };

    PolylineTrailMaterialProperty.prototype.getValue = function (time, result) {
        if (!Cesium.defined(result)) result = {};
        result.color = Cesium.Property.getValueOrClonedDefault(
            this._color, time, Cesium.Color.WHITE, result.color
        );
        result.time = ((Date.now() - this._time) % this.duration) / this.duration;
        return result;
    };

    PolylineTrailMaterialProperty.prototype.equals = function (other) {
        return this === other ||
            (other instanceof PolylineTrailMaterialProperty &&
                Cesium.Property.equals(this._color, other._color) &&
                this.duration === other.duration);
    };

    return PolylineTrailMaterialProperty;
})();

function ensurePolylineTrailMaterial() {
    if (Cesium.Material.PolylineTrailType) return;
    Cesium.Material.PolylineTrailType = 'PolylineTrail';
    Cesium.Material.PolylineTrailSource = `
        czm_material czm_getMaterial(czm_materialInput materialInput)
        {
            czm_material material = czm_getDefaultMaterial(materialInput);
            vec2 st = materialInput.st;
            float t = fract(st.s - time);
            float head = smoothstep(0.0, 0.15, t);
            float tail = smoothstep(1.0, 0.85, t);
            float alpha = head * tail;
            material.alpha = alpha * color.a;
            material.diffuse = color.rgb;
            return material;
        }
    `;
    Cesium.Material._materialCache.addMaterial(Cesium.Material.PolylineTrailType, {
        fabric: {
            type: Cesium.Material.PolylineTrailType,
            uniforms: {
                color: new Cesium.Color(1, 1, 1, 1),
                time: 0,
            },
            source: Cesium.Material.PolylineTrailSource,
        },
        translucent: function () { return true; },
    });
}

ensurePolylineTrailMaterial();

// ============================================================
// CESIUM GLOBE SETUP
// ============================================================
const box = document.getElementById('canvas-container');

const viewer = new Cesium.Viewer(box, {
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    animation: false,
    timeline: false,
    fullscreenButton: false,
    vrButton: false,
    infoBox: false,
    selectionIndicator: false,
    creditContainer: document.createElement('div'),
    skyAtmosphere: new Cesium.SkyAtmosphere(),
    orderIndependentTranslucency: true,
    shadows: false,
    requestRenderMode: false,
    // No imageryProvider — we add our own below
    imageryProvider: false,
});

// Add imagery: try ArcGIS first, fallback to bundled Natural Earth
(async () => {
    try {
        const arcgis = await Cesium.ArcGisMapServerImageryProvider.fromUrl(
            'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
        );
        viewer.imageryLayers.addImageryProvider(arcgis);
        console.log('ArcGIS World Imagery loaded');
    } catch (e) {
        console.warn('ArcGIS unavailable, trying Natural Earth:', e.message);
        try {
            const tms = await Cesium.TileMapServiceImageryProvider.fromUrl(
                Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII')
            );
            viewer.imageryLayers.addImageryProvider(tms);
            console.log('Natural Earth imagery loaded');
        } catch (e2) {
            console.warn('Natural Earth also failed:', e2.message);
            // Last resort: OpenStreetMap
            viewer.imageryLayers.addImageryProvider(
                new Cesium.OpenStreetMapImageryProvider({
                    url: 'https://tile.openstreetmap.org/',
                })
            );
            console.log('OpenStreetMap imagery loaded');
        }
    }
})();

// Configure atmosphere
viewer.scene.skyAtmosphere.brightnessShift = 0.0;
viewer.scene.skyAtmosphere.hueShift = -0.05;
viewer.scene.skyAtmosphere.saturationShift = 0.1;
viewer.scene.globe.enableLighting = true;
viewer.scene.globe.dynamicAtmosphereLighting = true;
viewer.scene.globe.dynamicAtmosphereLightingFromSun = true;
viewer.scene.globe.showGroundAtmosphere = true;
viewer.scene.globe.atmosphereBrightnessShift = -0.1;

// Occlude entities behind the globe. Without this, markers/labels on the far
// side render through the Earth and appear over the wrong continents, shifting
// as the camera rotates (looks like they "slide" with the viewer).
viewer.scene.globe.depthTestAgainstTerrain = true;

// Markers within this camera distance (m) skip depth-testing so near-surface
// points/labels stay crisp without z-fighting; beyond it (e.g. the far side of
// the globe) the depth buffer occludes them. ~2 Earth radii.
const DEPTH_TEST_NEAR = 1.3e7;

// Cesium renders the globe, sky, stars — Three.js overlays space physics on top
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#030308');

// Expose the viewer + Cesium namespace for console debugging and external tools.
if (typeof window !== 'undefined') { window.viewer = viewer; window.Cesium = Cesium; }
viewer.scene.sun.show = true;
viewer.scene.moon.show = true;
viewer.scene.skyBox.show = true;

// Initial camera: view Earth from space
viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(0, 20, 25000000),
    orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-90),
        roll: 0,
    },
});

// Cesium canvas stays at default z-index; Three.js overlay goes on top with pointer-events:none

// Remove default Cesium UI chrome
const toolbar = box.querySelector('.cesium-viewer-toolbar');
if (toolbar) toolbar.style.display = 'none';
const bottomContainer = box.querySelector('.cesium-viewer-bottom');
if (bottomContainer) bottomContainer.style.display = 'none';

// ============================================================
// THREE.JS SPACE PHYSICS OVERLAY
// ============================================================
// Three.js renders magnetosphere, solar wind, comet, cosmic rays
// on a transparent canvas overlaid on the Cesium globe.
// We position the overlay BEHIND Cesium's canvas and use Cesium
// primitives for near-Earth features instead.

// Three.js overlay disabled — Cesium handles all rendering.
// Magnetosphere/solar wind will be added as Cesium primitives later.
const threeScene = new THREE.Scene();
const threeCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);

// Three.js helpers
const R = 1;
function ll2v(lat, lon, r = R * 1.001) {
    const p = (90 - lat) * Math.PI / 180, t = (lon + 180) * Math.PI / 180;
    return new THREE.Vector3(-r * Math.sin(p) * Math.cos(t), r * Math.cos(p), r * Math.sin(p) * Math.sin(t));
}

// Camera sync: map Cesium ECEF camera to Three.js scene
const EARTH_RADIUS = 6371000;
const SCALE = R / EARTH_RADIUS;

function syncThreeCamera() {
    const cam = viewer.camera;
    const pos = cam.positionWC;
    const dir = cam.directionWC;
    const up = cam.upWC;

    // Cesium uses ECEF (X=equator/prime meridian, Y=equator/90E, Z=north pole)
    // Three.js: X=right, Y=up, Z=toward viewer
    // Map: Cesium X -> Three X, Cesium Z -> Three Y, Cesium -Y -> Three Z
    threeCamera.position.set(pos.x * SCALE, pos.z * SCALE, -pos.y * SCALE);

    const lookTarget = new THREE.Vector3(
        (pos.x + dir.x * 10000) * SCALE,
        (pos.z + dir.z * 10000) * SCALE,
        -(pos.y + dir.y * 10000) * SCALE
    );
    threeCamera.up.set(up.x, up.z, -up.y);
    threeCamera.lookAt(lookTarget);

    // Match FOV
    if (cam.frustum.fovy) {
        threeCamera.fov = Cesium.Math.toDegrees(cam.frustum.fovy);
    }
    threeCamera.updateProjectionMatrix();
}

// Three.js layer groups (for space physics only)
const threeLayerGroups = {};
function getThreeLayer(name) {
    if (!threeLayerGroups[name]) {
        threeLayerGroups[name] = new THREE.Group();
        threeScene.add(threeLayerGroups[name]);
    }
    return threeLayerGroups[name];
}
function clearThreeLayer(name) {
    const g = threeLayerGroups[name];
    if (!g) return;
    while (g.children.length) {
        const c = g.children[0];
        if (c.geometry) c.geometry.dispose();
        if (c.material) { if (Array.isArray(c.material)) c.material.forEach(m => m.dispose()); else c.material.dispose(); }
        g.remove(c);
    }
}

// ============================================================
// CESIUM DATA LAYERS
// ============================================================

// --- Layer visibility state ---
const layerVisible = {
    earthquakes: true, 'eq-waves': true, 'jelly-ball': true,
    subsolar: true, terminator: true, plates: true,
    magnetometers: false, weather: true, 'wind-field': true, 'ocean-currents': true, clouds: true, geojson: false,
    'magnetic-field': true, 'solar-wind': true, telluric: true,
    'magnetic-anomalies': true,
    'ocean-lights': true,
};

// --- Live state variables (updated by poll loop) ---
let magnetoCompression = 1.0, stormLevel = 0, currentBz = 0, currentKp = 2, currentDst = 0;
let swSpeed = 400, swDensity = 5, swElectronFlux = 100, swProtonScore = 0;

// --- Data sources for each Cesium layer ---
const dataSources = {};

async function getDataSource(name) {
    if (!dataSources[name]) {
        const ds = new Cesium.CustomDataSource(name);
        dataSources[name] = ds;
        await viewer.dataSources.add(ds);
    }
    return dataSources[name];
}

function clearDataSource(name) {
    if (dataSources[name]) dataSources[name].entities.removeAll();
}

function setLayerVisible(name, visible) {
    layerVisible[name] = visible;
    if (dataSources[name]) dataSources[name].show = visible;
    if (threeLayerGroups[name]) threeLayerGroups[name].visible = visible;
    if (name === 'wind-field') setWindFieldVisible(visible);
    if (name === 'ocean-currents') setOceanFieldVisible(visible);
}

// ============================================================
// EARTHQUAKE LAYER
// ============================================================
const eqWaves = [];

function recencyColorCss(ageH) {
    if (ageH < 1) return Cesium.Color.WHITE;
    if (ageH < 6) return Cesium.Color.fromCssColorString(`rgb(255,${Math.floor(76 + 179 * (1 - ageH / 6))},25)`);
    if (ageH < 24) return Cesium.Color.fromCssColorString(`rgb(255,${Math.floor(76 * (1 - (ageH - 6) / 18))},0)`);
    if (ageH < 48) return new Cesium.Color(0.5, 0.2, 0.5, 0.8);
    return new Cesium.Color(0.2, 0.2, 0.5, 0.6);
}

async function updateEarthquakes(data) {
    const ds = await getDataSource('earthquakes');
    ds.entities.removeAll();
    eqWaves.length = 0;
    updateCompoundFaultContext(data?.compound_fault_advisories || []);
    if (!data?.earthquakes) return;
    const now = Date.now();

    for (const eq of data.earthquakes) {
        const ageH = (now - eq.time) / 3600000;
        const color = recencyColorCss(ageH);
        const size = Math.max(3, Math.pow(eq.mag - 3.5, 1.3) * 4);

        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(eq.lon, eq.lat, (eq.depth || 33) * 100),
            point: {
                pixelSize: size,
                color: color,
                outlineColor: Cesium.Color.BLACK.withAlpha(0.3),
                outlineWidth: 1,
                disableDepthTestDistance: DEPTH_TEST_NEAR,
                heightReference: Cesium.HeightReference.NONE,
            },
            properties: { ...eq, _type: 'earthquake' },
        });

        // Glow for M6+
        if (eq.mag >= 6.0) {
            ds.entities.add({
                position: Cesium.Cartesian3.fromDegrees(eq.lon, eq.lat, (eq.depth || 33) * 100),
                point: {
                    pixelSize: size * 4,
                    color: color.withAlpha(0.1),
                    disableDepthTestDistance: DEPTH_TEST_NEAR,
                },
            });
        }

        // Seismic wave rings for recent events
        if (ageH < 24) {
            const waveSpeed = 0.3 + (eq.mag - 4) * 0.15;
            const maxRad = 2 + Math.pow(eq.mag - 4, 2) * 1.5;
            const currentRad = Math.min(ageH * waveSpeed, maxRad);
            const opacity = Math.max(0, 0.5 * (1 - currentRad / maxRad));
            if (currentRad > 0.2 && opacity > 0.02) {
                const positions = greatCircleDegrees(eq.lat, eq.lon, currentRad);
                ds.entities.add({
                    polyline: {
                        positions: Cesium.Cartesian3.fromDegreesArray(positions),
                        width: 1.5,
                        material: new Cesium.ColorMaterialProperty(color.withAlpha(opacity)),
                        clampToGround: true,
                    },
                });
            }
        }
    }
    document.getElementById('st-eqs').textContent = data.earthquakes.length;
}

function updateCompoundFaultContext(advisories) {
    const panel = document.getElementById('compound-fault-context');
    const copy = document.getElementById('compound-fault-copy');
    const source = document.getElementById('compound-fault-source');
    if (!panel || !copy || !source) return;

    const advisory = advisories.find(item => item?.active);
    if (!advisory) {
        panel.style.display = 'none';
        copy.textContent = '';
        source.removeAttribute('href');
        return;
    }

    const additional = Math.max(0, advisories.filter(item => item?.active).length - 1);
    const suffix = additional ? ` (+${additional} additional candidate${additional === 1 ? '' : 's'})` : '';
    copy.textContent = `Candidate ${advisory.trigger_candidate}; monitor ${advisory.target} after authoritative fault attribution${suffix}.`;
    source.href = advisory.source;
    panel.style.display = 'block';
}

function greatCircleDegrees(lat, lon, radiusDeg, nPts = 120) {
    const slat = lat * Math.PI / 180, slon = lon * Math.PI / 180, rd = radiusDeg * Math.PI / 180;
    const pts = [];
    for (let i = 0; i <= nPts; i++) {
        const a = (i / nPts) * 2 * Math.PI;
        const lat2 = Math.asin(Math.sin(slat) * Math.cos(rd) + Math.cos(slat) * Math.sin(rd) * Math.cos(a));
        const lon2 = slon + Math.atan2(Math.sin(a) * Math.sin(rd) * Math.cos(slat), Math.cos(rd) - Math.sin(slat) * Math.sin(lat2));
        pts.push(lon2 * 180 / Math.PI, lat2 * 180 / Math.PI);
    }
    return pts;
}

// ============================================================
// JELLY BALL ZONES
// ============================================================
async function updateJellyBall(data) {
    const ds = await getDataSource('jelly-ball');
    ds.entities.removeAll();
    if (!data?.zones) return;

    // Only draw key zones (eye, wavefront, antipodal) to reduce clutter
    const keyZones = ['eye', 'wavefront', 'antipodal'];
    for (const zone of data.zones) {
        if (!keyZones.includes(zone.name)) continue;
        const positions = greatCircleDegrees(data.lat, data.lon, zone.radius_deg);
        const color = Cesium.Color.fromCssColorString(zone.color);
        ds.entities.add({
            polyline: {
                positions: Cesium.Cartesian3.fromDegreesArray(positions),
                width: zone.name === 'wavefront' ? 2.5 : 1.5,
                material: new Cesium.ColorMaterialProperty(color.withAlpha(0.4)),
                clampToGround: true,
            },
        });
        // Label at top of ring
        const labelPt = greatCircleDegrees(data.lat, data.lon, zone.radius_deg, 4);
        // Pick the northernmost point (index 2,3 = lon,lat of second point)
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(labelPt[2], labelPt[3], 30000),
            label: {
                text: `${zone.name} ${zone.ratio}x`,
                font: '9px monospace',
                fillColor: color,
                pixelOffset: new Cesium.Cartesian2(0, -6),
                disableDepthTestDistance: DEPTH_TEST_NEAR,
                scale: 0.7,
            },
        });
    }
}

// ============================================================
// SUBSOLAR POINT + TERMINATOR
// ============================================================
let subsolarEntity = null;
let antipodalEntity = null;

async function updateSubsolar(data) {
    const ds = await getDataSource('subsolar');
    ds.entities.removeAll();
    if (!data) return;

    // Subsolar point marker
    subsolarEntity = ds.entities.add({
        position: Cesium.Cartesian3.fromDegrees(data.lon, data.lat, 50000),
        point: {
            pixelSize: 18,
            color: Cesium.Color.YELLOW,
            outlineColor: Cesium.Color.ORANGE,
            outlineWidth: 2,
            disableDepthTestDistance: DEPTH_TEST_NEAR,
        },
        label: {
            text: 'SUBSOLAR',
            font: '10px monospace',
            fillColor: Cesium.Color.YELLOW,
            style: Cesium.LabelStyle.FILL,
            pixelOffset: new Cesium.Cartesian2(0, -18),
            disableDepthTestDistance: DEPTH_TEST_NEAR,
        },
    });

    // Vertical ray from subsolar point
    ds.entities.add({
        polyline: {
            positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                data.lon, data.lat, 0,
                data.lon, data.lat, 500000,
            ]),
            width: 1.5,
            material: new Cesium.ColorMaterialProperty(Cesium.Color.YELLOW.withAlpha(0.3)),
        },
    });

    // Antipodal point
    const antiLon = data.lon > 0 ? data.lon - 180 : data.lon + 180;
    antipodalEntity = ds.entities.add({
        position: Cesium.Cartesian3.fromDegrees(antiLon, -data.lat, 10000),
        point: {
            pixelSize: 10,
            color: new Cesium.Color(0.8, 0.5, 0.8, 0.5),
            disableDepthTestDistance: DEPTH_TEST_NEAR,
        },
    });

    // Update sun position for lighting
    const sunPos = Cesium.Cartesian3.fromDegrees(data.lon, data.lat, 149597870700);
    viewer.scene.light = new Cesium.DirectionalLight({
        direction: Cesium.Cartesian3.normalize(
            Cesium.Cartesian3.negate(sunPos, new Cesium.Cartesian3()),
            new Cesium.Cartesian3()
        ),
        intensity: 2.0,
    });
}

async function updateTerminator(data) {
    const ds = await getDataSource('terminator');
    ds.entities.removeAll();
    if (!data) return;

    const positions = greatCircleDegrees(data.lat, data.lon, 90, 180);
    ds.entities.add({
        polyline: {
            positions: Cesium.Cartesian3.fromDegreesArray(positions),
            width: 2,
            material: new Cesium.ColorMaterialProperty(
                Cesium.Color.fromCssColorString('#ff6600').withAlpha(0.35)
            ),
            clampToGround: true,
        },
    });
}

// ============================================================
// GEOMAGNETIC DIPOLE FIELD LINES
// ============================================================
// WMM 2025 geomagnetic pole: 80.6°N, 72.7°W (north dip pole)
// Dipole tilt = 9.7° from rotation axis
const DIPOLE_NORTH = { lat: 80.6, lon: -72.7 };
const DIPOLE_SOUTH = { lat: -80.6, lon: 107.3 };
const DEG = Math.PI / 180;

// Rotate a point from dipole coordinates (colat θ_d, lon φ_d) to geographic
// Dipole axis points toward (poleLat, poleLon)
function dipoleToGeo(thetaD, phiD, poleLat, poleLon) {
    const cosP = Math.cos(poleLat * DEG), sinP = Math.sin(poleLat * DEG);
    // Direction cosines in dipole frame
    const x = Math.sin(thetaD) * Math.cos(phiD);
    const y = Math.sin(thetaD) * Math.sin(phiD);
    const z = Math.cos(thetaD);
    // Rotate around Y by (90° - poleLat) to tilt dipole axis to geographic
    const tilt = (90 - poleLat) * DEG;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    const xg = x * ct + z * st;
    const yg = y;
    const zg = -x * st + z * ct;
    // Convert to geographic lat/lon, then add pole longitude offset
    const geoLat = Math.asin(zg) / DEG;
    const geoLon = poleLon + Math.atan2(yg, xg) / DEG;
    return { lat: geoLat, lon: ((geoLon + 540) % 360) - 180 };
}

async function buildMagneticField(subsolarData) {
    const ds = await getDataSource('magnetic-field');
    ds.entities.removeAll();

    const Bz = currentBz;
    // Magnetopause standoff: Shue et al. 1998 model
    // r0 = 11.4 * (Bz_nT)^(1/6.6) Re for Dp ~ 2 nPa; simplified:
    const Dp = (swDensity || 5) * 1.67e-27 * ((swSpeed || 400) * 1e3) ** 2 * 1e9; // nPa
    const r0_Re = 11.4 * Math.pow(Math.max(0.5, Dp), -1 / 6.6);
    const compRatio = Math.min(1.0, Math.max(0.4, r0_Re / 11.4));
    const RE = 6371000; // meters

    // Field line colors
    const innerColor = Cesium.Color.fromCssColorString('#00ccff'); // L=2-3
    const outerColor = Cesium.Color.fromCssColorString('#336699'); // L=4-6

    // Draw dipole field lines: r = L sin²θ in dipole coordinates
    // 6 meridional planes (φ_d = 0, 60, 120, ...) × 4 L-shells
    const nPlanes = 6;
    const Lshells = [2.0, 3.0, 4.5, 6.0];

    for (let p = 0; p < nPlanes; p++) {
        const phiD = (p / nPlanes) * 2 * Math.PI;
        for (let si = 0; si < Lshells.length; si++) {
            const L = Lshells[si];
            const pts = []; // [lon, lat, alt, ...]
            // Trace from north foot to south foot
            // Foot colatitude: sin²θ_foot = 1/L → θ_foot = arcsin(1/√L)
            const thetaFoot = Math.asin(1 / Math.sqrt(L));
            const nPts = 80;
            for (let j = 0; j <= nPts; j++) {
                const theta = thetaFoot + (Math.PI - 2 * thetaFoot) * (j / nPts);
                const sinT = Math.sin(theta);
                const r = L * sinT * sinT; // dipole equation
                if (r < 1.0) continue;
                const alt = (r - 1.0) * RE;

                // Get geographic position
                const geo = dipoleToGeo(theta, phiD, DIPOLE_NORTH.lat, DIPOLE_NORTH.lon);

                // Solar wind compression: compress dayside, stretch nightside
                let adjAlt = alt;
                if (subsolarData) {
                    const dLon = ((geo.lon - subsolarData.lon + 540) % 360) - 180;
                    const dayFrac = Math.cos(dLon * DEG); // +1 = subsolar, -1 = antisolar
                    if (dayFrac > 0) {
                        // Dayside: compress by magnetopause standoff
                        adjAlt *= compRatio;
                        // Cap at magnetopause
                        const rMpause = r0_Re * RE * (1 - dayFrac * 0.3);
                        if (adjAlt + RE > rMpause) adjAlt = rMpause - RE;
                    } else {
                        // Nightside: stretch into magnetotail
                        adjAlt *= 1 + (1 - compRatio) * 0.6 * (-dayFrac);
                    }
                }

                if (adjAlt > 0 && adjAlt < 80000000) {
                    pts.push(geo.lon, geo.lat, adjAlt);
                }
            }

            if (pts.length >= 9) {
                const color = si < 2 ? innerColor : outerColor;
                const alpha = si < 2 ? 0.5 : 0.25;
                ds.entities.add({
                    polyline: {
                        positions: Cesium.Cartesian3.fromDegreesArrayHeights(pts),
                        width: si < 2 ? 2.0 : 1.2,
                        material: new Cesium.ColorMaterialProperty(color.withAlpha(alpha)),
                    },
                });
            }
        }
    }

    // Geomagnetic pole markers
    for (const pole of [DIPOLE_NORTH, DIPOLE_SOUTH]) {
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(pole.lon, pole.lat, 5000),
            point: { pixelSize: 8, color: Cesium.Color.CYAN, disableDepthTestDistance: DEPTH_TEST_NEAR },
            label: {
                text: pole.lat > 0 ? 'N mag' : 'S mag',
                font: '9px monospace', fillColor: Cesium.Color.CYAN,
                pixelOffset: new Cesium.Cartesian2(12, 0),
                disableDepthTestDistance: DEPTH_TEST_NEAR,
            },
        });
    }

    // ---- AURORA OVALS ----
    // Feldstein model: oval centered on geomagnetic pole, equatorward boundary
    // depends on Kp. Oval is NOT a circle — wider on nightside (magnetic midnight).
    // Equatorward boundary: Λ = 67° - 3° × Kp  (Starkov 1994)
    const kp = currentKp || 2;
    const auroraEqward = 67 - 3 * Math.min(kp, 9); // geomagnetic colatitude
    const auroraPoleward = auroraEqward + 5 + kp * 0.5; // ~5° wide band

    for (const isNorth of [true, false]) {
        const pole = isNorth ? DIPOLE_NORTH : DIPOLE_SOUTH;
        const sign = isNorth ? 1 : -1;

        for (const [colatBand, alpha, width] of [[auroraEqward, 0.5, 3], [auroraPoleward, 0.2, 1.5]]) {
            const pts = [];
            for (let i = 0; i <= 120; i++) {
                const phiD = (i / 120) * 2 * Math.PI; // magnetic local time
                // Nightside (φ_d ~ π) oval extends 3-5° more equatorward
                const nightShift = 3 * Math.max(0, -Math.cos(phiD));
                const colat = (colatBand + nightShift) * DEG;
                const geo = dipoleToGeo(
                    isNorth ? colat : Math.PI - colat,
                    phiD, DIPOLE_NORTH.lat, DIPOLE_NORTH.lon
                );
                pts.push(geo.lon, geo.lat);
            }
            ds.entities.add({
                polyline: {
                    positions: Cesium.Cartesian3.fromDegreesArray(pts),
                    width: width,
                    material: new Cesium.PolylineGlowMaterialProperty({
                        glowPower: 0.25,
                        color: Cesium.Color.fromCssColorString('#44ff88').withAlpha(alpha * (0.3 + stormLevel * 0.7)),
                    }),
                    clampToGround: true,
                },
            });
        }
    }
}

// ============================================================
// ANIMATED SOLAR WIND — smooth flowing streams around magnetosphere
// ============================================================
// Visual-scale compression: real magnetopause is ~10 Re but we draw
// at ~3-5 Re so Earth stays prominent. Physics is still correct
// relative to itself; only the absolute scale is compressed.

let swAnimRunning = false;
let swSubsolar = null;
let swPhase = 0; // global animation phase

async function buildSolarWindFlow(subsolarData) {
    const ds = await getDataSource('solar-wind');
    ds.entities.removeAll();
    if (!subsolarData) return;
    swSubsolar = subsolarData;

    const ssLon = subsolarData.lon;
    const ssLat = subsolarData.lat;
    const speed = swSpeed || 400;
    const density = swDensity || 5;
    const Bz = currentBz || 0;
    const RE = 6371000; // meters

    // Visual scale: magnetopause at ~2.5 Re altitude for good proportions
    const mpAlt = RE * 2.5;  // ~16,000 km altitude
    const bowAlt = mpAlt * 1.3;

    // Speed-based color
    const speedFrac = Math.min(1, Math.max(0, (speed - 300) / 400));
    const streamColor = Cesium.Color.fromHsl(0.10 - speedFrac * 0.06, 0.9, 0.55, 0.45);

    // --- Smooth stream lines ---
    // 10 streams spread in latitude, each a smooth arc from upstream to downstream
    const nStreams = 10;
    for (let i = 0; i < nStreams; i++) {
        const latOffset = (i / (nStreams - 1) - 0.5) * 70; // ±35° spread from subsolar

        // Build a smooth stream path: approach from far, curve at bow shock, flow to tail
        const pts = [];
        const nPts = 40;
        for (let j = 0; j < nPts; j++) {
            const t = j / (nPts - 1); // 0=far upstream, 1=far downstream

            // Longitude arc: upstream (+80°) → subsolar (0°) → downstream (-120°)
            const lonArc = ssLon + 80 - t * 200;

            // Altitude: high upstream, dips toward bow shock, rises at flanks
            // Smooth parabolic profile peaking at bow shock distance
            const approachT = Math.max(0, 1 - t * 2); // 1 at upstream, 0 at subsolar
            const flankT = Math.max(0, t * 2 - 1);     // 0 at subsolar, 1 at tail
            const coreT = 1 - Math.abs(t - 0.4) * 2.5; // peaks near bow shock

            // Streams far from center pass high, center streams dip close
            const centerDist = Math.abs(latOffset) / 35; // 0=center, 1=edge
            const baseAlt = bowAlt * (0.8 + centerDist * 0.6);
            const dip = (1 - centerDist) * bowAlt * 0.3 * Math.max(0, coreT);
            const alt = baseAlt + approachT * bowAlt * 0.5 - dip + flankT * bowAlt * 0.2;

            // Latitude: slight curve toward equator at flanks
            const lat = ssLat + latOffset * (1 + flankT * 0.3);

            pts.push(lonArc, lat, Math.max(RE * 0.5, alt));
        }

        const width = 2.0 - Math.abs(latOffset) / 35 * 0.8; // thicker near center
        ds.entities.add({
            polyline: {
                positions: Cesium.Cartesian3.fromDegreesArrayHeights(pts),
                width: width,
                material: new Cesium.PolylineGlowMaterialProperty({
                    glowPower: 0.15,
                    color: streamColor,
                }),
            },
        });
    }

    // --- Bow shock arc ---
    const bowColor = stormLevel > 0.5
        ? Cesium.Color.fromCssColorString('#ff5533').withAlpha(0.5)
        : Cesium.Color.fromCssColorString('#ff8844').withAlpha(0.4);

    // Horizontal arc
    const bowPts = [];
    for (let i = -70; i <= 70; i += 3) {
        const angle = i * DEG;
        const alt = bowAlt / Math.max(0.3, Math.cos(angle * 0.7));
        const lat = ssLat + Math.sin(angle) * 45;
        bowPts.push(ssLon + Math.cos(angle) * 15, lat, alt);
    }
    ds.entities.add({
        polyline: {
            positions: Cesium.Cartesian3.fromDegreesArrayHeights(bowPts),
            width: 2.5,
            material: new Cesium.PolylineGlowMaterialProperty({ glowPower: 0.25, color: bowColor }),
        },
    });

    // Vertical arc (rotated 90°)
    const bowPts2 = [];
    for (let i = -70; i <= 70; i += 3) {
        const angle = i * DEG;
        const alt = bowAlt / Math.max(0.3, Math.cos(angle * 0.7));
        const lat = ssLat + Math.cos(angle) * 5;
        const extraAlt = Math.sin(angle) * bowAlt * 0.8;
        bowPts2.push(ssLon, lat, alt + extraAlt);
    }
    ds.entities.add({
        polyline: {
            positions: Cesium.Cartesian3.fromDegreesArrayHeights(bowPts2),
            width: 2,
            material: new Cesium.PolylineGlowMaterialProperty({
                glowPower: 0.2, color: bowColor.withAlpha(0.3),
            }),
        },
    });

    // --- IMF Bz + solar wind data label ---
    const bzColor = Bz < 0
        ? Cesium.Color.fromCssColorString('#ff4466')
        : Cesium.Color.fromCssColorString('#4488ff');
    ds.entities.add({
        position: Cesium.Cartesian3.fromDegrees(ssLon + 20, ssLat, bowAlt * 1.1),
        label: {
            text: `SW: ${speed.toFixed(0)} km/s  n=${density.toFixed(1)}/cc\nBz ${Bz > 0 ? '+' : ''}${Bz.toFixed(1)} nT  ${Bz < -5 ? 'RECONNECTING' : Bz < 0 ? 'southward' : 'northward'}`,
            font: '10px monospace',
            fillColor: bzColor,
            pixelOffset: new Cesium.Cartesian2(0, 0),
            disableDepthTestDistance: DEPTH_TEST_NEAR,
            showBackground: true,
            backgroundColor: Cesium.Color.BLACK.withAlpha(0.6),
        },
    });

    // --- Magnetotail field lines (stretched on nightside) ---
    const tailColor = Cesium.Color.fromCssColorString('#224488').withAlpha(0.2);
    for (let i = -2; i <= 2; i++) {
        const latOff = i * 8;
        const tailPts = [];
        for (let j = 0; j <= 20; j++) {
            const t = j / 20;
            const lon = ssLon - 180 + (1 - t) * 80; // antisolar, stretching away
            const alt = RE * (0.5 + t * 2.5) + Math.abs(latOff) * RE * 0.05;
            tailPts.push(((lon + 540) % 360) - 180, ssLat * -0.2 + latOff, alt);
        }
        ds.entities.add({
            polyline: {
                positions: Cesium.Cartesian3.fromDegreesArrayHeights(tailPts),
                width: 1,
                material: new Cesium.ColorMaterialProperty(tailColor),
            },
        });
    }
}

// ============================================================
// TELLURIC CURRENTS (ocean + fault, Kp-scaled)
// ============================================================
async function buildTelluricCurrents(fieldData) {
    const ds = await getDataSource('telluric');
    ds.entities.removeAll();

    const kp = fieldData?.inputs?.kp || currentKp || 2;
    const jLive = fieldData?.telluric_j?.value || 2.0;

    // Ocean v×B telluric currents: detailed paths from physical oceanography
    // Intensities are baseline quiet-time J (mA/km) from Malin & Barraclough 1991
    // Coordinates trace actual current axes from satellite altimetry + drifter data
    const telluricPaths = [
        // ---- ATLANTIC GYRE ----
        // Gulf Stream: Florida Strait → Cape Hatteras → meander zone → Grand Banks separation
        // Peak transport ~30 Sv at Straits, v×B generates ~270 mA/km
        { name: 'Gulf Stream', baseline: 270, color: '#ff4444', path: [
            [-80.0,23.5],[-80.2,24.5],[-80.1,25.8],[-79.8,27.0],  // Florida Strait
            [-79.5,28.5],[-79.7,29.8],[-80.0,30.5],[-79.8,31.2],  // hugs Florida coast
            [-79.2,31.8],[-78.5,32.5],[-77.8,33.2],[-76.5,34.0],  // Georgia Bight
            [-75.5,34.8],[-74.8,35.5],[-74.0,36.0],[-73.0,36.8],  // Cape Hatteras separation
            [-71.5,37.5],[-69.5,38.5],[-67.0,39.5],[-64.0,40.0],  // meander zone (eddies)
            [-60.0,40.5],[-56.0,41.5],[-52.0,42.5],[-49.0,43.5],  // Grand Banks
            [-45.0,44.5],[-42.0,46.0],[-38.0,47.5],[-34.0,48.5],  // mid-Atlantic transition
        ]},
        // North Atlantic Drift → Norwegian Current
        { name: 'N. Atlantic Drift', baseline: 80, color: '#ff8844', path: [
            [-34.0,48.5],[-28.0,50.0],[-22.0,51.5],[-18.0,52.5],  // from Gulf Stream terminus
            [-14.0,53.5],[-10.0,55.0],[-8.0,56.5],[-5.0,58.0],    // west of Ireland/Scotland
            [-2.0,59.5],[2.0,61.0],[5.0,62.5],[7.0,64.0],          // Norwegian Sea
            [10.0,66.0],[13.0,68.0],[15.0,70.0],                    // toward Barents Sea
        ]},
        // Canary Current (southward return, eastern boundary)
        { name: 'Canary Current', baseline: 25, color: '#ff9966', path: [
            [-10.0,43.0],[-11.0,40.0],[-12.0,37.0],[-13.0,34.0],
            [-14.5,31.0],[-16.0,28.0],[-17.5,25.0],[-18.0,22.0],
            [-18.5,19.0],[-18.0,16.0],[-17.0,13.5],
        ]},
        // North Equatorial Current (westward)
        { name: 'N. Equatorial (Atl)', baseline: 20, color: '#ffaa66', path: [
            [-17.0,13.5],[-22.0,12.0],[-28.0,11.0],[-35.0,10.5],
            [-42.0,10.0],[-48.0,10.5],[-54.0,11.5],[-58.0,12.5],
        ]},
        // Brazil Current (western boundary, southward)
        { name: 'Brazil Current', baseline: 35, color: '#ff8844', path: [
            [-35.0,-5.0],[-36.5,-8.0],[-37.5,-11.0],[-38.0,-14.0],
            [-38.5,-17.0],[-39.5,-20.0],[-41.0,-23.0],[-43.0,-25.0],
            [-46.0,-28.0],[-49.0,-31.0],[-51.0,-33.5],[-52.0,-36.0],
        ]},
        // ---- PACIFIC GYRE ----
        // Kuroshio: Taiwan → Japan coast → Kuroshio Extension
        // Peak transport ~55 Sv, v×B ~ 60 mA/km
        { name: 'Kuroshio', baseline: 60, color: '#ff8844', path: [
            [121.5,18.0],[122.0,20.0],[122.5,22.0],[123.0,24.0],   // east of Taiwan
            [124.5,25.5],[126.0,27.0],[128.0,28.5],[130.0,30.0],   // Ryukyu Islands
            [131.5,31.0],[133.0,32.0],[134.5,33.0],[136.0,33.5],   // south of Japan
            [137.5,34.0],[139.5,34.5],[141.0,35.0],[142.5,35.5],   // Enshu-nada
            [144.0,36.0],[146.0,36.0],[148.0,35.5],[150.0,35.0],   // Kuroshio Extension
            [153.0,35.0],[156.0,35.5],[160.0,36.0],[165.0,37.0],   // meander/eddy zone
            [170.0,38.0],[175.0,39.0],[180.0,40.0],
        ]},
        // California Current (southward, eastern boundary)
        { name: 'California Current', baseline: 15, color: '#ffaa66', path: [
            [-126.0,48.0],[-126.0,45.0],[-125.5,42.0],[-124.5,39.0],
            [-123.0,36.5],[-121.5,34.5],[-120.0,32.5],[-118.5,30.5],
            [-117.0,28.0],[-116.0,25.0],[-115.0,22.0],
        ]},
        // North Equatorial Current (Pacific, westward)
        { name: 'N. Equatorial (Pac)', baseline: 18, color: '#ffaa66', path: [
            [-115.0,22.0],[-120.0,18.0],[-130.0,15.0],[-140.0,13.0],
            [-150.0,12.0],[-160.0,11.5],[-170.0,11.0],[180.0,10.5],
            [170.0,10.0],[160.0,10.0],[150.0,11.0],[140.0,12.0],
            [130.0,13.5],[125.0,15.0],
        ]},
        // Humboldt/Peru Current (cold, upwelling, northward along S. America)
        { name: 'Humboldt Current', baseline: 30, color: '#ff9966', path: [
            [-75.0,-42.0],[-74.0,-38.0],[-73.5,-34.0],[-73.0,-30.0],
            [-72.0,-26.0],[-71.5,-22.0],[-71.5,-18.0],[-72.0,-14.0],
            [-76.0,-10.0],[-80.0,-6.0],[-82.0,-3.0],[-82.5,0.0],
        ]},
        // East Australian Current
        { name: 'E. Australian', baseline: 35, color: '#ff8844', path: [
            [155.0,-15.0],[154.5,-18.0],[154.0,-20.0],[153.5,-23.0],
            [153.5,-25.0],[153.5,-27.5],[154.0,-30.0],[154.5,-32.0],
            [155.0,-34.0],[156.0,-36.0],[157.5,-37.5],[160.0,-38.5],
        ]},
        // ---- INDIAN OCEAN ----
        // Agulhas Current: Mozambique Channel → south Africa → retroflection
        // Peak ~70 Sv, one of the strongest western boundary currents
        { name: 'Agulhas', baseline: 50, color: '#ff8844', path: [
            [40.5,-11.0],[41.0,-15.0],[40.5,-18.0],[39.0,-21.0],   // Mozambique Channel
            [37.0,-24.0],[35.5,-26.5],[34.0,-28.0],[32.0,-30.0],   // hugs coast
            [30.5,-31.5],[29.0,-33.0],[27.0,-34.0],[25.0,-34.5],   // southeast SA
            [23.0,-35.0],[21.0,-35.5],[19.5,-36.0],[18.5,-36.5],   // Cape of Good Hope
            [19.0,-37.5],[20.5,-38.5],[23.0,-39.0],[26.0,-39.0],   // retroflection loop
            [29.0,-38.0],[31.0,-36.5],                               // back east
        ]},
        // Indonesia Throughflow: Pacific→Indian through narrow straits
        // Only ~15 Sv but through narrow channels = high v = high J
        { name: 'Indonesia Throughflow', baseline: 45, color: '#ff6644', path: [
            [127.0,4.0],[126.5,2.5],[126.0,1.5],[125.5,0.5],       // Molucca Sea
            [124.5,-0.5],[123.5,-1.5],[122.0,-2.5],[121.0,-3.5],   // Banda Sea
            [119.5,-5.0],[118.0,-6.5],[116.5,-8.0],[115.0,-8.5],   // Lombok/Savu straits
            [113.0,-9.0],[111.0,-10.0],[109.0,-11.0],[107.0,-11.5], // into Indian Ocean
        ]},
        // Somali Current (seasonal, monsoon-driven)
        { name: 'Somali Current', baseline: 40, color: '#ff8844', path: [
            [43.0,-2.0],[44.0,0.0],[45.5,3.0],[47.0,5.5],
            [49.0,8.0],[50.5,10.0],[51.5,11.5],[51.0,13.0],
        ]},
        // ---- SOUTHERN OCEAN ----
        // Antarctic Circumpolar Current: follows actual meandering path
        // Largest current on Earth by volume (~130 Sv), v×B ~ 40 mA/km
        { name: 'Antarctic Circumpolar', baseline: 40, color: '#ff6644', path: [
            [-70.0,-56.0],[-65.0,-56.5],[-60.0,-57.5],             // Drake Passage
            [-55.0,-54.0],[-50.0,-50.0],[-45.0,-48.0],             // Falkland/Malvinas
            [-40.0,-47.0],[-35.0,-46.5],[-30.0,-46.0],             // mid-Atlantic Ridge
            [-25.0,-46.5],[-20.0,-47.0],[-15.0,-47.5],
            [-10.0,-48.0],[-5.0,-48.5],[0.0,-49.0],[5.0,-49.5],
            [10.0,-49.0],[15.0,-48.0],[20.0,-47.0],[25.0,-46.0],   // south of Africa
            [30.0,-46.5],[40.0,-47.0],[50.0,-48.0],[60.0,-49.0],
            [70.0,-50.0],[80.0,-50.5],[90.0,-51.0],[100.0,-52.0],  // south Indian Ocean
            [110.0,-53.0],[120.0,-54.0],[130.0,-55.0],
            [140.0,-55.5],[145.0,-56.0],[150.0,-57.0],             // south of Australia
            [155.0,-58.0],[160.0,-59.0],[165.0,-60.0],
            [170.0,-61.0],[175.0,-62.0],[180.0,-62.5],             // south Pacific
            [-175.0,-63.0],[-170.0,-63.5],[-165.0,-63.0],
            [-160.0,-62.5],[-155.0,-62.0],[-150.0,-61.5],
            [-140.0,-61.0],[-130.0,-60.5],[-120.0,-60.0],
            [-110.0,-59.5],[-100.0,-59.0],[-90.0,-58.5],
            [-80.0,-58.0],[-75.0,-57.5],[-70.0,-56.0],             // back to Drake
        ]},
        // ---- FAULT ZONE CONDUCTORS ----
        // East African Rift: fault gouge + hydrothermal fluids = conductor
        { name: 'E. African Rift', baseline: 15, color: '#ffaa44', path: [
            [36.5,12.0],[36.8,10.0],[37.0,8.0],[36.8,6.0],[36.2,4.0],
            [35.5,2.0],[35.0,0.0],[34.5,-1.5],[33.5,-4.0],[32.5,-6.5],
            [31.5,-9.0],[30.5,-11.0],[29.5,-13.0],[29.0,-15.0],
        ]},
        // San Andreas Fault
        { name: 'San Andreas', baseline: 12, color: '#ffaa44', path: [
            [-115.5,32.5],[-116.0,33.0],[-117.0,34.0],[-117.8,34.5],
            [-118.5,34.8],[-119.5,35.2],[-120.5,35.8],[-121.0,36.2],
            [-121.5,36.8],[-122.0,37.5],[-122.5,38.0],[-123.0,38.5],
            [-123.3,39.0],[-123.5,40.0],
        ]},
    ];

    // Storm scaling: telluric J ~ exp(0.4 × Kp) from the backend model
    const stormScale = Math.exp(0.4 * kp) / Math.exp(0.4 * 2); // ratio vs Kp=2 baseline

    for (const path of telluricPaths) {
        const J = path.baseline * stormScale;
        const width = Math.max(1.5, Math.min(6, J / 50));
        const alpha = Math.min(0.8, 0.2 + J / 300);
        const color = Cesium.Color.fromCssColorString(path.color);

        const flat = path.path.flatMap(([lon, lat]) => [lon, lat]);
        ds.entities.add({
            polyline: {
                positions: Cesium.Cartesian3.fromDegreesArray(flat),
                width, clampToGround: true,
                material: new PolylineTrailMaterialProperty(
                    color.withAlpha(alpha),
                    Math.max(900, 3200 - J * 6)
                ),
            },
            properties: { _type: 'telluric', name: path.name, j_mA_km: J.toFixed(1) },
        });

        const mid = path.path[Math.floor(path.path.length / 2)];
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(mid[0], mid[1], 20000),
            label: {
                text: `${path.name}\n${J.toFixed(0)} mA/km`,
                font: '9px monospace', fillColor: color,
                pixelOffset: new Cesium.Cartesian2(0, -8),
                disableDepthTestDistance: DEPTH_TEST_NEAR, scale: 0.8,
                showBackground: true, backgroundColor: Cesium.Color.BLACK.withAlpha(0.5),
            },
        });
    }
}

// Build initial layers (will update with real data on first poll)
buildMagneticField(null);
buildTelluricCurrents(null);

// ============================================================
// PLATE BOUNDARIES (GeoJSON)
// ============================================================
async function loadPlates() {
    try {
        const resp = await fetch(`${API}/plates`);
        const geojson = await resp.json();
        const ds = await getDataSource('plates');
        ds.entities.removeAll();
        const legendEl = document.getElementById('plate-legend');
        const namesUsed = new Set();

        if (!geojson.features) return;

        for (const feature of geojson.features) {
            const props = feature.properties || {};
            const coords = feature.geometry?.coordinates || [];
            if (coords.length < 2) continue;

            const color = Cesium.Color.fromCssColorString(props.color || '#445566');
            // Split at antimeridian crossings
            let segment = [];
            for (let i = 0; i < coords.length; i++) {
                if (i > 0 && Math.abs(coords[i][0] - coords[i - 1][0]) > 90) {
                    if (segment.length >= 2) {
                        const flat = segment.flatMap(c => [c[0], c[1]]);
                        ds.entities.add({
                            polyline: {
                                positions: Cesium.Cartesian3.fromDegreesArray(flat),
                                width: 1.5,
                                material: new Cesium.ColorMaterialProperty(color.withAlpha(0.5)),
                                clampToGround: true,
                            },
                            properties: { _type: 'plate', name: props.name, boundary_type: props.boundary_type },
                        });
                    }
                    segment = [];
                }
                segment.push(coords[i]);
            }
            if (segment.length >= 2) {
                const flat = segment.flatMap(c => [c[0], c[1]]);
                ds.entities.add({
                    polyline: {
                        positions: Cesium.Cartesian3.fromDegreesArray(flat),
                        width: 1.5,
                        material: new Cesium.ColorMaterialProperty(color.withAlpha(0.5)),
                        clampToGround: true,
                    },
                    properties: { _type: 'plate', name: props.name, boundary_type: props.boundary_type },
                });
            }
            namesUsed.add(props.name);
        }

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
    } catch (e) { console.warn('Plates:', e.message); }
}
loadPlates();

// ============================================================
// MAGNETOMETER STATIONS
// ============================================================
async function updateMagnetometers(data) {
    const ds = await getDataSource('magnetometers');
    ds.entities.removeAll();
    if (!data?.stations) return;

    for (const st of data.stations) {
        const color = st.network === 'USGS'
            ? new Cesium.Color(0.8, 0.27, 0.8, 0.9)
            : new Cesium.Color(0.27, 0.8, 0.8, 0.9);
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(st.lon, st.lat, 5000),
            point: {
                pixelSize: 8,
                color: color,
                outlineColor: Cesium.Color.WHITE.withAlpha(0.3),
                outlineWidth: 1,
                disableDepthTestDistance: DEPTH_TEST_NEAR,
            },
            label: {
                text: st.code || '',
                font: '9px monospace',
                fillColor: color,
                pixelOffset: new Cesium.Cartesian2(0, -12),
                disableDepthTestDistance: DEPTH_TEST_NEAR,
                scale: 0.8,
            },
            properties: { _type: 'magnetometer', ...st },
        });
    }
}

// ============================================================
// MAGNETIC ANOMALIES (ore deposits, BIFs)
// ============================================================
async function updateMagneticAnomalies(data) {
    const ds = await getDataSource('magnetic-anomalies');
    ds.entities.removeAll();
    if (!data?.anomalies) return;

    for (const anom of data.anomalies) {
        const strength = Math.abs(anom.strength_nT);
        const maxS = 5000;
        const norm = Math.min(1, strength / maxS);

        // Radius proportional to sqrt(area) but min 30km for visibility
        const bodyRadius = Math.max(30000, Math.sqrt(anom.area_km2) * 1000);
        // Anomaly halo: detectability ring scaled by strength
        const haloRadius = bodyRadius * (1 + norm * 3);

        // Color by Schumann interaction regime
        let color;
        if (anom.schumann_regime === 'scatterer') {
            color = new Cesium.Color(1.0, 0.27, 1.0, 0.5);  // magenta = scatterer
        } else if (anom.schumann_regime === 'absorber') {
            color = new Cesium.Color(1.0, 0.6, 0.27, 0.4);  // orange = absorber
        } else {
            color = new Cesium.Color(0.6, 0.6, 1.0, 0.3);   // pale blue = transparent
        }

        // Outer halo (anomaly extent)
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(anom.lon, anom.lat, 500),
            ellipsoid: {
                radii: new Cesium.Cartesian3(haloRadius, haloRadius, 3000),
                material: color.withAlpha(0.12 + norm * 0.15),
            },
            properties: { _type: 'anomaly', ...anom },
        });

        // Inner core (ore body)
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(anom.lon, anom.lat, 2000),
            ellipsoid: {
                radii: new Cesium.Cartesian3(bodyRadius, bodyRadius, 4000),
                material: color.withAlpha(0.35 + norm * 0.25),
            },
            properties: { _type: 'anomaly', ...anom },
        });

        // Label
        const labelText = `${anom.name}\n${anom.strength_nT > 0 ? '+' : ''}${anom.strength_nT} nT`;
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(anom.lon, anom.lat, 8000),
            label: {
                text: labelText,
                font: '10px monospace',
                fillColor: color.withAlpha(0.9),
                outlineColor: Cesium.Color.BLACK.withAlpha(0.6),
                outlineWidth: 2,
                style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                pixelOffset: new Cesium.Cartesian2(0, -18),
                disableDepthTestDistance: DEPTH_TEST_NEAR,
                scale: 0.85,
            },
            properties: { _type: 'anomaly', ...anom },
        });
    }
}

// ============================================================
// OCEAN LIGHT PHENOMENA (te lapa, St. Elmo's fire, EQ lights)
// ============================================================
async function updateOceanLightPhenomena(data) {
    const ds = await getDataSource('ocean-lights');
    ds.entities.removeAll();
    if (!data) return;

    // Draw ocean current paths
    if (data.currents) {
        for (const c of data.currents) {
            if (!c.path || c.path.length < 2) continue;
            const flat = c.path.flatMap(p => [p[0], p[1]]);
            ds.entities.add({
                polyline: {
                    positions: Cesium.Cartesian3.fromDegreesArray(flat),
                    width: 2.5,
                    material: new Cesium.ColorMaterialProperty(
                        Cesium.Color.fromCssColorString(c.color || '#4488ff').withAlpha(0.35)
                    ),
                    clampToGround: false,
                },
                properties: { _type: 'ocean_current', name: c.name },
            });
        }
    }

    // Draw report markers
    if (data.reports) {
        for (const r of data.reports) {
            let color, size, symbol;
            if (r.type === 'te_lapa') {
                color = new Cesium.Color(0.2, 1.0, 0.8, 0.9);  // cyan-green
                size = 10;
                symbol = '\u2726';  // star
            } else if (r.type === 'st_elmo') {
                color = new Cesium.Color(0.5, 0.4, 1.0, 0.9);  // violet
                size = 9;
                symbol = '\u26A1';  // lightning
            } else {
                color = new Cesium.Color(1.0, 0.5, 0.2, 0.9);  // orange
                size = 8;
                symbol = '\u25C6';  // diamond
            }

            // Marker point
            ds.entities.add({
                position: Cesium.Cartesian3.fromDegrees(r.lon, r.lat, 3000),
                point: {
                    pixelSize: size,
                    color: color,
                    outlineColor: Cesium.Color.WHITE.withAlpha(0.4),
                    outlineWidth: 1,
                    disableDepthTestDistance: DEPTH_TEST_NEAR,
                },
                label: {
                    text: r.name,
                    font: '9px monospace',
                    fillColor: color,
                    pixelOffset: new Cesium.Cartesian2(0, -14),
                    disableDepthTestDistance: DEPTH_TEST_NEAR,
                    scale: 0.75,
                },
                properties: { _type: 'ocean_light', ...r },
            });

            // Te lapa: add a radial glow to represent the island E-field perturbation
            if (r.type === 'te_lapa') {
                ds.entities.add({
                    position: Cesium.Cartesian3.fromDegrees(r.lon, r.lat, 500),
                    ellipsoid: {
                        radii: new Cesium.Cartesian3(130000, 130000, 2000),  // 130 km radius
                        material: color.withAlpha(0.06),
                    },
                    properties: { _type: 'ocean_light', ...r },
                });
            }
        }
    }
}

// ============================================================
// WEATHER MARKERS (precipitation + lightning)
// ============================================================
async function renderWeatherMarkers(precipData) {
    const ds = await getDataSource('weather');
    ds.entities.removeAll();
    if (!precipData?.stations) return;

    for (const st of precipData.stations) {
        const isThunder = st.thunder_hours > 0;
        const hasRain = st.total_72h_mm > 5;
        if (!hasRain && !isThunder) continue;

        if (hasRain) {
            const rainHeight = Math.min(200000, st.total_72h_mm * 600);
            ds.entities.add({
                position: Cesium.Cartesian3.fromDegrees(st.lon, st.lat, rainHeight / 2),
                cylinder: {
                    length: rainHeight,
                    topRadius: 30000,
                    bottomRadius: 30000,
                    material: Cesium.Color.fromCssColorString('#4488ff').withAlpha(
                        Math.min(0.5, 0.15 + st.total_72h_mm / 200)
                    ),
                    outline: false,
                },
                properties: { _type: 'precip', ...st },
            });
        }

        if (isThunder) {
            // Lightning bolt as a zigzag polyline
            const baseH = 20000;
            const topH = 50000 + st.thunder_hours * 5000;
            const phase = Math.random() * Math.PI * 2;
            const boltColor = new Cesium.CallbackProperty(() => {
                const pulse = 0.3 + 0.7 * Math.abs(Math.sin(Date.now() / 180 + phase));
                return Cesium.Color.fromCssColorString('#ffcc44').withAlpha(pulse);
            }, false);
            ds.entities.add({
                polyline: {
                    positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                        st.lon, st.lat, baseH,
                        st.lon + 0.05, st.lat + 0.03, (baseH + topH) / 3,
                        st.lon - 0.04, st.lat - 0.02, (baseH + topH) * 2 / 3,
                        st.lon, st.lat, topH,
                    ]),
                    width: 2.5,
                    material: new Cesium.ColorMaterialProperty(boltColor),
                },
                properties: { _type: 'thunder', ...st },
            });

            // Glow halo
            ds.entities.add({
                position: Cesium.Cartesian3.fromDegrees(st.lon, st.lat, 30000),
                ellipsoid: {
                    radii: new Cesium.Cartesian3(50000, 50000, 30000),
                    material: Cesium.Color.fromCssColorString('#ffaa22').withAlpha(0.1),
                },
            });
        }
    }

    // Wind streaks (station-based, animated)
    for (const st of precipData.stations) {
        if (st.wind_speed_kmh == null || st.wind_dir_deg == null) continue;
        const speed = Math.max(0, st.wind_speed_kmh);
        const dir = (st.wind_dir_deg + 180) % 360; // convert "from" to "to"
        const dirRad = Cesium.Math.toRadians(dir);
        const lenKm = Math.max(40, Math.min(220, 40 + speed * 3));
        const dLat = (lenKm / 111) * Math.cos(dirRad);
        const dLon = (lenKm / (111 * Math.max(0.2, Math.cos(Cesium.Math.toRadians(st.lat))))) * Math.sin(dirRad);
        const nStreaks = 6;
        for (let i = 0; i < nStreaks; i++) {
            const jitterLon = (Math.random() - 0.5) * 0.8;
            const jitterLat = (Math.random() - 0.5) * 0.8;
            const startLon = st.lon + jitterLon;
            const startLat = st.lat + jitterLat;
            const endLon = startLon + dLon * 0.35;
            const endLat = startLat + dLat * 0.35;
            const alpha = Math.min(0.9, 0.25 + speed / 80);
            const color = Cesium.Color.fromCssColorString('#66ccff').withAlpha(alpha);
            ds.entities.add({
                polyline: {
                    positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                        startLon, startLat, 12000,
                        endLon, endLat, 14000,
                    ]),
                    width: 2.0,
                    material: new PolylineTrailMaterialProperty(color, Math.max(900, 2800 - speed * 20)),
                },
                properties: { _type: 'wind', ...st },
            });
        }
    }
}

// ============================================================
// CLOUD CHARGE LAYER
// ============================================================
async function renderCloudLayer(cloudData) {
    const ds = await getDataSource('clouds');
    ds.entities.removeAll();
    if (!cloudData?.stations) return;

    for (const st of cloudData.stations) {
        const cc = st.cloud_cover?.total || 0;
        if (cc < 20) continue;

        let color, alpha;
        if (st.charge_type === 'Cb dipole') { color = '#ffcc44'; alpha = 0.2; }
        else if (st.charge_type === 'convective') { color = '#ff8844'; alpha = 0.12; }
        else if (st.charge_type === 'stratiform') { color = '#88aacc'; alpha = 0.06; }
        else { color = '#6688aa'; alpha = 0.03 + cc / 100 * 0.04; }

        const size = 60000 + cc / 100 * 80000;
        ds.entities.add({
            position: Cesium.Cartesian3.fromDegrees(st.lon, st.lat, 8000),
            ellipsoid: {
                radii: new Cesium.Cartesian3(size, size, 5000),
                material: Cesium.Color.fromCssColorString(color).withAlpha(alpha),
            },
            properties: { _type: 'cloud', ...st },
        });

        // Charge gradient arrow for charged regions
        if (st.charge_c > 1) {
            ds.entities.add({
                polyline: {
                    positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                        st.lon, st.lat, 3000,
                        st.lon, st.lat, 12000 + st.charge_c * 500,
                    ]),
                    width: 2,
                    material: new Cesium.PolylineGlowMaterialProperty({
                        glowPower: 0.3,
                        color: Cesium.Color.fromCssColorString('#ff4488').withAlpha(0.5),
                    }),
                },
            });
        }
    }
}

// ============================================================
// GLOBAL WIND FIELD (particle flow)
// ============================================================
const windCanvas = document.getElementById('wind-canvas');
const windCtx = windCanvas?.getContext('2d');
const oceanCanvas = document.getElementById('ocean-canvas');
const oceanCtx = oceanCanvas?.getContext('2d');
let windFieldVisible = true;

function setWindFieldVisible(visible) {
    windFieldVisible = visible;
    if (windCanvas) windCanvas.style.display = visible ? 'block' : 'none';
}

let oceanFieldVisible = true;

function setOceanFieldVisible(visible) {
    oceanFieldVisible = visible;
    if (oceanCanvas) oceanCanvas.style.display = visible ? 'block' : 'none';
}

function resizeOverlayCanvas(canvas, ctx) {
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeWindCanvas);
function resizeWindCanvas() {
    resizeOverlayCanvas(windCanvas, windCtx);
    resizeOverlayCanvas(oceanCanvas, oceanCtx);
}
resizeWindCanvas();

const windColorStops = [
    { s: 0, c: [90, 160, 255] },
    { s: 5, c: [70, 210, 200] },
    { s: 10, c: [120, 255, 140] },
    { s: 15, c: [255, 225, 90] },
    { s: 20, c: [255, 140, 70] },
    { s: 30, c: [255, 60, 60] },
];

function windColor(speed) {
    const s = Math.max(0, Math.min(30, speed || 0));
    for (let i = 0; i < windColorStops.length - 1; i++) {
        const a = windColorStops[i], b = windColorStops[i + 1];
        if (s >= a.s && s <= b.s) {
            const t = (s - a.s) / (b.s - a.s || 1);
            const r = Math.round(a.c[0] + (b.c[0] - a.c[0]) * t);
            const g = Math.round(a.c[1] + (b.c[1] - a.c[1]) * t);
            const bch = Math.round(a.c[2] + (b.c[2] - a.c[2]) * t);
            return `rgb(${r},${g},${bch})`;
        }
    }
    return 'rgb(255,60,60)';
}

const windField = {
    grid: null,
    particles: [],
    lastFrame: performance.now(),
    maxAge: 70,
    seedCount: 1200,
    setGrid(grid) {
        this.grid = grid;
        this.resetParticles();
    },
    resetParticles() {
        if (!this.grid) return;
        const w = windCanvas?.clientWidth || 800;
        const h = windCanvas?.clientHeight || 600;
        this.seedCount = Math.max(800, Math.min(1800, Math.floor((w * h) / 1200)));
        this.particles = [];
        for (let i = 0; i < this.seedCount; i++) this.particles.push(this.randomParticle());
    },
    randomParticle() {
        const g = this.grid;
        const lat = g.minLat + Math.random() * (g.maxLat - g.minLat);
        const lon = g.minLon + Math.random() * (g.maxLon - g.minLon);
        return { lat, lon, age: Math.random() * this.maxAge };
    },
    sample(lat, lon) {
        const g = this.grid;
        if (!g) return null;
        if (lat < g.minLat || lat > g.maxLat) return null;
        let lonWrap = lon;
        while (lonWrap < g.minLon) lonWrap += 360;
        while (lonWrap > g.maxLon) lonWrap -= 360;

        const i = (lat - g.minLat) / g.dLat;
        const j = (lonWrap - g.minLon) / g.dLon;
        const i0 = Math.floor(i);
        let j0 = Math.floor(j);
        const i1 = Math.min(g.nLat - 1, i0 + 1);
        let j1 = j0 + 1;
        if (j1 >= g.nLon) j1 = 0;
        if (i0 < 0 || i1 >= g.nLat) return null;

        const fi = i - i0;
        const fj = j - j0;

        const idx = (ii, jj) => ii * g.nLon + jj;
        const u00 = g.u[idx(i0, j0)], v00 = g.v[idx(i0, j0)];
        const u10 = g.u[idx(i1, j0)], v10 = g.v[idx(i1, j0)];
        const u01 = g.u[idx(i0, j1)], v01 = g.v[idx(i0, j1)];
        const u11 = g.u[idx(i1, j1)], v11 = g.v[idx(i1, j1)];
        if ([u00, v00, u10, v10, u01, v01, u11, v11].some(x => x == null)) return null;

        const u0 = u00 * (1 - fi) + u10 * fi;
        const u1 = u01 * (1 - fi) + u11 * fi;
        const v0 = v00 * (1 - fi) + v10 * fi;
        const v1 = v01 * (1 - fi) + v11 * fi;
        return { u: u0 * (1 - fj) + u1 * fj, v: v0 * (1 - fj) + v1 * fj };
    },
};

const WIND_VISUAL_SPEEDUP = 28000;
const OCEAN_VISUAL_SPEEDUP = 42000;

function isLikelyLandRegion(lat, lon) {
    return (
        (lon >= -170 && lon <= -50 && lat >= 10 && lat <= 75) ||   // North America
        (lon >= -85 && lon <= -35 && lat >= -55 && lat <= 15) ||   // South America
        (lon >= -20 && lon <= 55 && lat >= -35 && lat <= 70) ||    // Africa + Europe
        (lon >= 55 && lon <= 150 && lat >= 5 && lat <= 70) ||      // Asia
        (lon >= 110 && lon <= 155 && lat >= -45 && lat <= -10)     // Australia
    );
}

async function renderWindFieldVectors(grid) {
    const ds = await getDataSource('wind-field');
    ds.entities.removeAll();
    if (!grid?.lats?.length || !grid?.lons?.length) return;

    const renderWindLattice = (latOffsetFrac, lonOffsetFrac, latticeId) => {
        for (let i = 0; i < grid.nLat; i += 1) {
            for (let j = 0; j < grid.nLon; j += 1) {
                const idx = i * grid.nLon + j;
                const u = grid.u[idx];
                const v = grid.v[idx];
                if (u == null || v == null) continue;

                const lat = grid.lats[i];
                const lon = grid.lons[j];
                const speed = Math.hypot(u, v);
                if (speed < 0.6) continue;

                const isLikelyLand = isLikelyLandRegion(lat, lon);
                const isAsiaInterior = lon >= 60 && lon <= 140 && lat >= 5 && lat <= 60;
                const lowSpeedBoost = speed < 5
                    ? (isAsiaInterior ? 1.45 : isLikelyLand ? 1.32 : 1.18)
                    : (isLikelyLand ? 1.08 : 1.0);
                const latRad = Cesium.Math.toRadians(lat);
                const baseColor = Cesium.Color.fromCssColorString(windColor(speed));
                const color = Cesium.Color.lerp(
                    baseColor,
                    Cesium.Color.WHITE,
                    isLikelyLand ? 0.35 : 0.08,
                    new Cesium.Color()
                ).withAlpha(Math.min(0.78, (0.18 + speed / 28) * lowSpeedBoost));
                const streamlineCount = speed > 8 ? 2 : speed > 5 ? 2 : (isAsiaInterior ? 2 : 1);
                const vectorScale = (18500 + speed * 1100) * lowSpeedBoost;
                const baseDLon = (u * vectorScale) / (111320 * Math.max(0.25, Math.cos(latRad)));
                const baseDLat = (v * vectorScale) / 111320;
                const basePerpLon = (-v * vectorScale * 0.1) / (111320 * Math.max(0.25, Math.cos(latRad)));
                const basePerpLat = (u * vectorScale * 0.1) / 111320;
                const cellLatSpread = Math.max(0.18, grid.dLat * (isAsiaInterior ? 0.28 : 0.34));
                const cellLonSpread = Math.max(0.18, grid.dLon * (isAsiaInterior ? 0.28 : 0.34));

                for (let k = 0; k < streamlineCount; k++) {
                    const hashA = Math.sin((i + 1) * 127.1 + (j + 1) * 311.7 + (k + 1) * 74.7 + latticeId * 51.2);
                    const hashB = Math.sin((i + 1) * 269.5 + (j + 1) * 183.3 + (k + 1) * 246.1 + latticeId * 93.7);
                    const hashC = Math.sin((i + 1) * 419.2 + (j + 1) * 371.9 + (k + 1) * 11.3 + latticeId * 17.4);
                    const latJitter = (((hashA + 1) * 0.5 - 0.5) * cellLatSpread) + (grid.dLat * latOffsetFrac);
                    const lonJitter = (((hashB + 1) * 0.5 - 0.5) * cellLonSpread) + (grid.dLon * lonOffsetFrac);
                    const offsetScale = ((hashC + 1) * 0.5 - 0.5) * 1.1;

                    const startLat = Math.max(-85, Math.min(85, lat + latJitter + basePerpLat * offsetScale));
                    let startLon = lon + lonJitter + basePerpLon * offsetScale;
                    if (startLon > 180) startLon -= 360;
                    if (startLon < -180) startLon += 360;

                    const wiggle = ((Math.sin((i + 1) * 0.9 + (j + 1) * 1.7 + (k + 1) * 2.3 + latticeId) + 1) * 0.5 - 0.5) * 0.32;
                    const midLat = Math.max(-85, Math.min(85, startLat + baseDLat * 0.55 + basePerpLat * wiggle));
                    let midLon = startLon + baseDLon * 0.55 + basePerpLon * wiggle;
                    if (midLon > 180) midLon -= 360;
                    if (midLon < -180) midLon += 360;

                    const endLat = Math.max(-85, Math.min(85, startLat + baseDLat));
                    let endLon = startLon + baseDLon;
                    if (endLon > 180) endLon -= 360;
                    if (endLon < -180) endLon += 360;

                    ds.entities.add({
                        polyline: {
                            positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                                startLon, startLat, isLikelyLand ? 24000 : 17000,
                                midLon, midLat, isLikelyLand ? 28500 : 22000,
                                endLon, endLat, isLikelyLand ? 31500 : 25000,
                            ]),
                            width: Math.max(0.9, Math.min(2.0, (0.82 + speed / 22) * (isAsiaInterior ? 1.12 : isLikelyLand ? 1.08 : 1.0))),
                            material: new PolylineTrailMaterialProperty(
                                color.withAlpha(Math.max(0.14, color.alpha - Math.abs(offsetScale) * 0.05)),
                                Math.max(720, 1900 - speed * 65 + k * 110 + latticeId * 90 - (isAsiaInterior ? 120 : 0))
                            ),
                        },
                        properties: { _type: 'wind-field-vector', speed_mps: speed.toFixed(1) },
                    });
                }
            }
        }
    };

    renderWindLattice(0.0, 0.0, 0);
    renderWindLattice(0.35, 0.5, 1);
}

async function refreshWindField() {
    const data = await fetchJSON('/wind_field');
    if (!data?.grid?.lats || !data?.grid?.lons) return;
    const lats = data.grid.lats;
    const lons = data.grid.lons;
    const grid = {
        lats,
        lons,
        u: data.grid.u,
        v: data.grid.v,
        nLat: lats.length,
        nLon: lons.length,
        dLat: lats.length > 1 ? (lats[1] - lats[0]) : 1,
        dLon: lons.length > 1 ? (lons[1] - lons[0]) : 1,
        minLat: lats[0],
        maxLat: lats[lats.length - 1],
        minLon: lons[0],
        maxLon: lons[lons.length - 1],
    };
    windField.setGrid(grid);
    renderWindFieldVectors(grid);
}

function stepWindField() {
    if (!windCanvas || !windCtx || !windField.grid || !windFieldVisible) {
        requestAnimationFrame(stepWindField);
        return;
    }

    const now = performance.now();
    const dt = Math.min(0.05, Math.max(0.012, (now - windField.lastFrame) / 1000));
    windField.lastFrame = now;

    const dpr = window.devicePixelRatio || 1;
    const w = windCanvas.width / dpr;
    const h = windCanvas.height / dpr;

    windCtx.fillStyle = 'rgba(3,3,8,0.08)';
    windCtx.fillRect(0, 0, w, h);
    windCtx.globalCompositeOperation = 'lighter';
    windCtx.lineWidth = 1.1;

    for (let p of windField.particles) {
        if (p.age++ > windField.maxAge) {
            Object.assign(p, windField.randomParticle());
            continue;
        }
        const sample = windField.sample(p.lat, p.lon);
        if (!sample) {
            Object.assign(p, windField.randomParticle());
            continue;
        }

        const speed = Math.hypot(sample.u, sample.v);
        const latRad = Cesium.Math.toRadians(p.lat);
        const dLat = (sample.v * dt * WIND_VISUAL_SPEEDUP) / 111320;
        const dLon = (sample.u * dt * WIND_VISUAL_SPEEDUP) / (111320 * Math.max(0.25, Math.cos(latRad)));

        const prevLat = p.lat;
        const prevLon = p.lon;
        p.lat += dLat;
        p.lon += dLon;
        if (p.lon > 180) p.lon -= 360;
        if (p.lon < -180) p.lon += 360;
        if (p.lat > windField.grid.maxLat || p.lat < windField.grid.minLat) {
            Object.assign(p, windField.randomParticle());
            continue;
        }

        const prevCart = Cesium.Cartesian3.fromDegrees(prevLon, prevLat, 12000);
        const nextCart = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 12000);
        const prevPos = Cesium.SceneTransforms.wgs84ToWindowCoordinates(viewer.scene, prevCart);
        const nextPos = Cesium.SceneTransforms.wgs84ToWindowCoordinates(viewer.scene, nextCart);
        if (!prevPos || !nextPos) continue;
        if (nextPos.x < 0 || nextPos.y < 0 || nextPos.x > w || nextPos.y > h) continue;

        windCtx.strokeStyle = windColor(speed);
        windCtx.beginPath();
        windCtx.moveTo(prevPos.x, prevPos.y);
        windCtx.lineTo(nextPos.x, nextPos.y);
        windCtx.stroke();
    }

    windCtx.globalCompositeOperation = 'source-over';
    requestAnimationFrame(stepWindField);
}

refreshWindField();
setInterval(refreshWindField, 15 * 60 * 1000);
stepWindField();

// ============================================================
// OCEAN CURRENTS FIELD (particle flow)
// ============================================================
const oceanColorStops = [
    { s: 0, c: [70, 120, 200] },
    { s: 0.3, c: [60, 170, 210] },
    { s: 0.6, c: [80, 210, 200] },
    { s: 1.0, c: [120, 240, 180] },
    { s: 1.5, c: [255, 210, 120] },
    { s: 2.0, c: [255, 140, 90] },
];

function oceanColor(speed) {
    const s = Math.max(0, Math.min(2.5, speed || 0));
    for (let i = 0; i < oceanColorStops.length - 1; i++) {
        const a = oceanColorStops[i], b = oceanColorStops[i + 1];
        if (s >= a.s && s <= b.s) {
            const t = (s - a.s) / (b.s - a.s || 1);
            const r = Math.round(a.c[0] + (b.c[0] - a.c[0]) * t);
            const g = Math.round(a.c[1] + (b.c[1] - a.c[1]) * t);
            const bch = Math.round(a.c[2] + (b.c[2] - a.c[2]) * t);
            return `rgb(${r},${g},${bch})`;
        }
    }
    return 'rgb(255,140,90)';
}

const oceanField = {
    grid: null,
    particles: [],
    lastFrame: performance.now(),
    maxAge: 90,
    seedCount: 900,
    setGrid(grid) {
        this.grid = grid;
        this.resetParticles();
    },
    resetParticles() {
        if (!this.grid) return;
        const w = oceanCanvas?.clientWidth || 800;
        const h = oceanCanvas?.clientHeight || 600;
        this.seedCount = Math.max(600, Math.min(1400, Math.floor((w * h) / 1400)));
        this.particles = [];
        for (let i = 0; i < this.seedCount; i++) this.particles.push(this.randomParticle());
    },
    randomParticle() {
        const g = this.grid;
        const lat = g.minLat + Math.random() * (g.maxLat - g.minLat);
        const lon = g.minLon + Math.random() * (g.maxLon - g.minLon);
        return { lat, lon, age: Math.random() * this.maxAge };
    },
    sample(lat, lon) {
        const g = this.grid;
        if (!g) return null;
        if (lat < g.minLat || lat > g.maxLat) return null;
        let lonWrap = lon;
        while (lonWrap < g.minLon) lonWrap += 360;
        while (lonWrap > g.maxLon) lonWrap -= 360;

        const i = (lat - g.minLat) / g.dLat;
        const j = (lonWrap - g.minLon) / g.dLon;
        const i0 = Math.floor(i);
        let j0 = Math.floor(j);
        const i1 = Math.min(g.nLat - 1, i0 + 1);
        let j1 = j0 + 1;
        if (j1 >= g.nLon) j1 = 0;
        if (i0 < 0 || i1 >= g.nLat) return null;

        const fi = i - i0;
        const fj = j - j0;

        const idx = (ii, jj) => ii * g.nLon + jj;
        const u00 = g.u[idx(i0, j0)], v00 = g.v[idx(i0, j0)];
        const u10 = g.u[idx(i1, j0)], v10 = g.v[idx(i1, j0)];
        const u01 = g.u[idx(i0, j1)], v01 = g.v[idx(i0, j1)];
        const u11 = g.u[idx(i1, j1)], v11 = g.v[idx(i1, j1)];
        if ([u00, v00, u10, v10, u01, v01, u11, v11].some(x => x == null)) return null;

        const u0 = u00 * (1 - fi) + u10 * fi;
        const u1 = u01 * (1 - fi) + u11 * fi;
        const v0 = v00 * (1 - fi) + v10 * fi;
        const v1 = v01 * (1 - fi) + v11 * fi;
        return { u: u0 * (1 - fj) + u1 * fj, v: v0 * (1 - fj) + v1 * fj };
    },
};

async function refreshOceanField() {
    const data = await fetchJSON('/ocean_currents');
    if (!data?.grid?.lats || !data?.grid?.lons) return;
    const lats = data.grid.lats;
    const lons = data.grid.lons;
    const grid = {
        lats,
        lons,
        u: data.grid.u,
        v: data.grid.v,
        nLat: lats.length,
        nLon: lons.length,
        dLat: lats.length > 1 ? (lats[1] - lats[0]) : 1,
        dLon: lons.length > 1 ? (lons[1] - lons[0]) : 1,
        minLat: lats[0],
        maxLat: lats[lats.length - 1],
        minLon: lons[0],
        maxLon: lons[lons.length - 1],
    };
    oceanField.setGrid(grid);
}

function stepOceanField() {
    if (!oceanCanvas || !oceanCtx || !oceanField.grid || !oceanFieldVisible) {
        requestAnimationFrame(stepOceanField);
        return;
    }

    const now = performance.now();
    const dt = Math.min(0.05, Math.max(0.012, (now - oceanField.lastFrame) / 1000));
    oceanField.lastFrame = now;

    const dpr = window.devicePixelRatio || 1;
    const w = oceanCanvas.width / dpr;
    const h = oceanCanvas.height / dpr;

    oceanCtx.fillStyle = 'rgba(2,4,10,0.08)';
    oceanCtx.fillRect(0, 0, w, h);
    oceanCtx.globalCompositeOperation = 'lighter';
    oceanCtx.lineWidth = 1.0;

    for (let p of oceanField.particles) {
        if (p.age++ > oceanField.maxAge) {
            Object.assign(p, oceanField.randomParticle());
            continue;
        }
        const sample = oceanField.sample(p.lat, p.lon);
        if (!sample) {
            Object.assign(p, oceanField.randomParticle());
            continue;
        }

        const speed = Math.hypot(sample.u, sample.v);
        const latRad = Cesium.Math.toRadians(p.lat);
        const dLat = (sample.v * dt * OCEAN_VISUAL_SPEEDUP) / 111320;
        const dLon = (sample.u * dt * OCEAN_VISUAL_SPEEDUP) / (111320 * Math.max(0.25, Math.cos(latRad)));

        const prevLat = p.lat;
        const prevLon = p.lon;
        p.lat += dLat;
        p.lon += dLon;
        if (p.lon > 180) p.lon -= 360;
        if (p.lon < -180) p.lon += 360;
        if (p.lat > oceanField.grid.maxLat || p.lat < oceanField.grid.minLat) {
            Object.assign(p, oceanField.randomParticle());
            continue;
        }

        const prevCart = Cesium.Cartesian3.fromDegrees(prevLon, prevLat, 7000);
        const nextCart = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 7000);
        const prevPos = Cesium.SceneTransforms.wgs84ToWindowCoordinates(viewer.scene, prevCart);
        const nextPos = Cesium.SceneTransforms.wgs84ToWindowCoordinates(viewer.scene, nextCart);
        if (!prevPos || !nextPos) continue;
        if (nextPos.x < 0 || nextPos.y < 0 || nextPos.x > w || nextPos.y > h) continue;

        oceanCtx.strokeStyle = oceanColor(speed);
        oceanCtx.beginPath();
        oceanCtx.moveTo(prevPos.x, prevPos.y);
        oceanCtx.lineTo(nextPos.x, nextPos.y);
        oceanCtx.stroke();
    }

    oceanCtx.globalCompositeOperation = 'source-over';
    requestAnimationFrame(stepOceanField);
}

refreshOceanField();
setInterval(refreshOceanField, 6 * 60 * 60 * 1000);
stepOceanField();

// ============================================================
// GEOJSON / KML LOADER
// ============================================================
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

async function renderGeoJSON(geojson) {
    const ds = await getDataSource('geojson');
    ds.entities.removeAll();
    const features = geojson.features || (geojson.geometry ? [geojson] : []);
    for (const feature of features) {
        const geom = feature.geometry || feature;
        const props = feature.properties || {};
        const color = Cesium.Color.fromCssColorString(props.color || props.stroke || '#44cccc');
        if (geom.type === 'LineString' || geom.type === 'MultiLineString') {
            const lines = geom.type === 'MultiLineString' ? geom.coordinates : [geom.coordinates];
            for (const line of lines) {
                ds.entities.add({
                    polyline: {
                        positions: Cesium.Cartesian3.fromDegreesArray(line.flatMap(c => [c[0], c[1]])),
                        width: 2,
                        material: new Cesium.ColorMaterialProperty(color.withAlpha(0.6)),
                        clampToGround: true,
                    },
                });
            }
        } else if (geom.type === 'Point') {
            ds.entities.add({
                position: Cesium.Cartesian3.fromDegrees(geom.coordinates[0], geom.coordinates[1]),
                point: { pixelSize: 8, color, disableDepthTestDistance: DEPTH_TEST_NEAR },
            });
        } else if (geom.type === 'Polygon' || geom.type === 'MultiPolygon') {
            const polys = geom.type === 'MultiPolygon' ? geom.coordinates : [geom.coordinates];
            for (const poly of polys) {
                for (const ring of poly) {
                    ds.entities.add({
                        polyline: {
                            positions: Cesium.Cartesian3.fromDegreesArray(ring.flatMap(c => [c[0], c[1]])),
                            width: 2,
                            material: new Cesium.ColorMaterialProperty(color.withAlpha(0.5)),
                            clampToGround: true,
                        },
                    });
                }
            }
        }
    }
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
            await renderGeoJSON(geojson);
            const cb = document.querySelector('[data-layer="geojson"]');
            if (cb) cb.checked = true;
            setLayerVisible('geojson', true);
            dropZone.textContent = `Loaded: ${file.name} (${geojson.features?.length || 0} features)`;
        }
    });
}

// ============================================================
// CESIUM CLICK / HOVER INTERACTION
// ============================================================
const tip = document.createElement('div');
tip.style.cssText = 'position:fixed;background:rgba(5,5,16,0.95);color:#ccc;font:11px monospace;padding:6px 10px;border:1px solid #00ccff;border-radius:4px;pointer-events:none;display:none;z-index:1000;max-width:280px;';
document.body.appendChild(tip);

const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction(movement => {
    const pick = viewer.scene.pick(movement.endPosition);
    if (Cesium.defined(pick) && pick.id?.properties) {
        const props = {};
        pick.id.properties.propertyNames.forEach(n => { props[n] = pick.id.properties[n]?.getValue(); });
        if (props._type === 'earthquake') {
            const eq = props;
            const ageH = (Date.now() - eq.time) / 3600000;
            const zc = { eye: '#44f', inner: '#66c', transition: '#4a4', wavefront: '#f44', 'wavefront-tail': '#f84', neutral: '#884', 'far-suppress': '#468', 'far-neutral': '#666', 'pre-antipodal': '#868', antipodal: '#c8c' };
            tip.innerHTML = `<b style="color:#ff6644">M${eq.mag?.toFixed(1)}</b> ${eq.place}<br>Depth: ${eq.depth?.toFixed(0) || '?'}km | ${ageH.toFixed(1)}h ago<br>${eq.ang_dist}deg | <span style="color:${zc[eq.zone] || '#888'}">${eq.zone}</span>`;
            tip.style.display = 'block';
            tip.style.left = (movement.endPosition.x + 14) + 'px';
            tip.style.top = (movement.endPosition.y - 10) + 'px';
            return;
        }
        if (props._type === 'plate') {
            tip.innerHTML = `<b style="color:#4488ff">${props.name}</b><br><span style="color:#889">${props.boundary_type}</span>`;
            tip.style.display = 'block';
            tip.style.left = (movement.endPosition.x + 14) + 'px';
            tip.style.top = (movement.endPosition.y - 10) + 'px';
            return;
        }
        if (props._type === 'telluric') {
            tip.innerHTML = `<b style="color:#ff8844">${props.name}</b><br>Telluric J: <span style="color:#ff4">${props.j_mA_km} mA/km</span>`;
            tip.style.display = 'block';
            tip.style.left = (movement.endPosition.x + 14) + 'px';
            tip.style.top = (movement.endPosition.y - 10) + 'px';
            return;
        }
    }
    tip.style.display = 'none';
}, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

handler.setInputAction(click => {
    const pick = viewer.scene.pick(click.position);
    if (Cesium.defined(pick) && pick.id?.properties) {
        const props = {};
        pick.id.properties.propertyNames.forEach(n => { props[n] = pick.id.properties[n]?.getValue(); });
        if (props._type === 'earthquake') { showDetail(props); return; }
        if (props._type === 'magnetometer') { showMagDetail(props); return; }
        if (props._type === 'anomaly') { showAnomalyDetail(props); return; }
        if (props._type === 'ocean_light') { showOceanLightDetail(props); return; }
    }
    document.getElementById('detail').style.display = 'none';
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

function showDetail(eq) {
    const panel = document.getElementById('detail'), content = document.getElementById('detail-content');
    const ageH = (Date.now() - eq.time) / 3600000, dt = new Date(eq.time);
    const zc = { eye: '#44f', inner: '#66c', transition: '#4a4', wavefront: '#f44', 'wavefront-tail': '#f84', neutral: '#884', 'far-suppress': '#468', 'far-neutral': '#666', 'pre-antipodal': '#868', antipodal: '#c8c' };
    const zr = { eye: '0.85x', inner: '0.92x', transition: '0.98x', wavefront: '1.36x', 'wavefront-tail': '1.09x', neutral: '0.95x', 'far-suppress': '0.82x', 'far-neutral': '0.90x', 'pre-antipodal': '1.00x', antipodal: '1.16x' };
    content.innerHTML = `<h3>M${eq.mag?.toFixed(1)} ${eq.place || 'Unknown'}</h3>
        <div class="row"><span class="k">Time</span><span class="val">${dt.toISOString().replace('T', ' ').substring(0, 19)} UTC</span></div>
        <div class="row"><span class="k">Age</span><span class="val">${ageH < 1 ? (ageH * 60).toFixed(0) + ' min' : ageH.toFixed(1) + ' hours'} ago</span></div>
        <div class="row"><span class="k">Location</span><span class="val">${eq.lat?.toFixed(3)}N, ${eq.lon?.toFixed(3)}E</span></div>
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
        <div class="row"><span class="k">Location</span><span class="val">${st.lat?.toFixed(2)}N, ${st.lon?.toFixed(2)}E</span></div>`;
    panel.style.display = 'block';
}

function showOceanLightDetail(r) {
    const panel = document.getElementById('detail'), content = document.getElementById('detail-content');
    const typeColors = { te_lapa: '#33ffcc', st_elmo: '#8866ff', eq_light: '#ff8833' };
    const typeNames = { te_lapa: 'Te Lapa', st_elmo: "St. Elmo's Fire", eq_light: 'Earthquake Light' };
    const c = typeColors[r.type] || '#fff';
    content.innerHTML = `<h3 style="color:${c}">${typeNames[r.type] || r.type}</h3>
        <div class="row"><span class="k">Report</span><span class="val">${r.name}</span></div>
        <div class="row"><span class="k">Observer</span><span class="val">${r.observer || '?'}</span></div>
        <div class="row"><span class="k">Date</span><span class="val">${r.year || '?'}</span></div>
        <div class="row"><span class="k">Location</span><span class="val">${r.lat?.toFixed(1)}°, ${r.lon?.toFixed(1)}°</span></div>
        <div class="row"><span class="k">Ocean current</span><span class="val" style="color:${c}">${r.current || '?'}</span></div>
        ${r.desc ? `<div style="margin-top:6px;font-size:11px;color:#aaa;line-height:1.4">${r.desc}</div>` : ''}`;
    panel.style.display = 'block';
}

function showAnomalyDetail(a) {
    const panel = document.getElementById('detail'), content = document.getElementById('detail-content');
    const regimeColor = a.schumann_regime === 'scatterer' ? '#ff44ff' : a.schumann_regime === 'absorber' ? '#ffaa44' : '#8888ff';
    content.innerHTML = `<h3 style="color:${regimeColor}">${a.name}</h3>
        <div class="row"><span class="k">Type</span><span class="val">${a.type}</span></div>
        <div class="row"><span class="k">Country</span><span class="val">${a.country}</span></div>
        <div class="row"><span class="k">Anomaly</span><span class="val" style="color:${regimeColor}">${a.strength_nT > 0 ? '+' : ''}${a.strength_nT} nT</span></div>
        <div class="row"><span class="k">Conductivity</span><span class="val">${a.conductivity_Sm} S/m</span></div>
        <div class="row"><span class="k">Area</span><span class="val">${a.area_km2?.toLocaleString()} km&sup2;</span></div>
        <div class="row"><span class="k">Magnetite</span><span class="val">${a.magnetite_pct}%</span></div>
        ${a.ore_Mt ? `<div class="row"><span class="k">Ore</span><span class="val">${a.ore_Mt?.toLocaleString()} Mt</span></div>` : ''}
        ${a.ree_Mt > 0 ? `<div class="row"><span class="k">REE oxide</span><span class="val">${a.ree_Mt} Mt</span></div>` : ''}
        <div style="margin-top:8px;border-top:1px solid #333;padding-top:6px;">
            <div class="row"><span class="k">Schumann (7.83 Hz)</span><span class="val" style="color:${regimeColor}">${a.schumann_regime?.toUpperCase()}</span></div>
            <div class="row"><span class="k">Skin depth</span><span class="val">${a.skin_depth_7Hz_km} km</span></div>
            <div class="row"><span class="k">Body/skin</span><span class="val">${a.body_over_skin}x</span></div>
        </div>`;
    panel.style.display = 'block';
}

// ============================================================
// THREE.JS SPACE PHYSICS (magnetosphere, solar wind, comet, CR)
// ============================================================

// Sun is rendered by Cesium (real position + lighting)
const SUN_X = 10; // used by comet/solar-wind for direction reference

// --- MAGNETOSPHERE ---
let reconnectionPositions = null, reconnectionPts = null;
const BOW_STANDOFF = () => 1.6 * magnetoCompression + 0.2;

function buildMagnetosphere() {
    clearThreeLayer('magnetosphere');
    const layer = getThreeLayer('magnetosphere');
    const comp = magnetoCompression, storm = stormLevel, S = 0.5;
    const cyan = new THREE.Color(0x00ccff), cyanDim = new THREE.Color(0x2288aa);

    for (let p = 0; p < 4; p++) {
        const phi = (p / 4) * Math.PI;
        for (let s = 0; s < 5; s++) {
            const L = 2.0 + s * 0.7;
            const pts = [];
            for (let j = 0; j <= 80; j++) {
                const theta = (j / 80) * Math.PI;
                const r = L * Math.sin(theta) * Math.sin(theta);
                let x = r * Math.sin(theta) * Math.cos(phi);
                let y = r * Math.cos(theta);
                let z = r * Math.sin(theta) * Math.sin(phi);
                if (x > 0) x *= comp * 0.7;
                if (x < 0) { x *= 1 + (1 - comp) * 0.8 + s * 0.15; y *= 1 - s * 0.04 * Math.min(1, Math.abs(x * S)); }
                pts.push(new THREE.Vector3(x * S, y * S, z * S));
            }
            layer.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(pts),
                new THREE.LineBasicMaterial({ color: s < 2 ? cyan : cyanDim, transparent: true, opacity: s < 2 ? 0.5 : 0.25 })
            ));
        }
    }

    // Bow shock
    const bowR = BOW_STANDOFF();
    const bowColor = storm > 0.5 ? 0xff6644 : 0x44ddff;
    for (let m = 0; m < 4; m++) {
        const angle = (m / 4) * Math.PI;
        const pts = [];
        for (let i = 0; i <= 40; i++) {
            const t = (i / 40) * Math.PI * 0.5;
            pts.push(new THREE.Vector3(bowR * Math.cos(t), bowR * Math.sin(t) * Math.sin(angle), bowR * Math.sin(t) * Math.cos(angle)));
        }
        layer.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: bowColor, transparent: true, opacity: 0.4 })));
    }

    // Aurora ovals (drawn in Three.js for consistency)
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

    // Ring current
    const rcIntensity = Math.min(1, Math.abs(currentDst) / 100);
    const rcMesh = new THREE.Mesh(
        new THREE.TorusGeometry(0.25, 0.02 + storm * 0.02, 12, 48),
        new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(0.08, 0.9, 0.4 + rcIntensity * 0.3), transparent: true, opacity: 0.06 + rcIntensity * 0.15, depthWrite: false })
    );
    rcMesh.rotation.x = Math.PI / 2;
    layer.add(rcMesh);

    // Reconnection
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
        const ix = i * 3, dir = i % 2 === 0 ? 1 : -1;
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
    currentKp = kp; currentDst = dst;
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

// --- SOLAR WIND PARTICLES ---
const SW_MAX = 800;
let swParticles = null, swPositions = null, swVelocities = null, swColors = null;

function initParticle(i, type) {
    const ix = i * 3;
    const spread = type === 2 ? 0.3 : 0.7;
    swPositions[ix] = 2 + Math.random() * 7;
    swPositions[ix + 1] = (Math.random() - 0.5) * spread;
    swPositions[ix + 2] = (Math.random() - 0.5) * spread;
    const baseSpeed = 0.01 + Math.random() * 0.005;
    const speedMult = type === 2 ? 2.5 : type === 1 ? 1.5 : 1.0;
    swVelocities[ix] = -baseSpeed * speedMult;
    swVelocities[ix + 1] = (Math.random() - 0.5) * 0.001;
    swVelocities[ix + 2] = (Math.random() - 0.5) * 0.001;
    if (type === 2) { swColors[ix] = 1.0; swColors[ix + 1] = 0.2; swColors[ix + 2] = 0.15; }
    else if (type === 1) { swColors[ix] = 0.3; swColors[ix + 1] = 0.85; swColors[ix + 2] = 1.0; }
    else { swColors[ix] = 1.0; swColors[ix + 1] = 0.85 + Math.random() * 0.15; swColors[ix + 2] = 0.4 + Math.random() * 0.3; }
}

function buildSolarWind() {
    clearThreeLayer('solar-wind');
    const layer = getThreeLayer('solar-wind');
    const geo = new THREE.BufferGeometry();
    swPositions = new Float32Array(SW_MAX * 3);
    swVelocities = new Float32Array(SW_MAX * 3);
    swColors = new Float32Array(SW_MAX * 3);
    const electronFrac = Math.min(0.4, swElectronFlux / 5000);
    const sepFrac = Math.min(0.2, swProtonScore * 0.2);
    const protonFrac = 1 - electronFrac - sepFrac;
    const activeCount = Math.min(SW_MAX, Math.floor(400 + swDensity * 50));
    for (let i = 0; i < SW_MAX; i++) {
        const t = i / SW_MAX;
        const type = t < protonFrac ? 0 : t < protonFrac + electronFrac ? 1 : 2;
        initParticle(i, type);
        if (i >= activeCount) { swPositions[i * 3] = 99; swPositions[i * 3 + 1] = 99; swPositions[i * 3 + 2] = 99; }
    }
    geo.setAttribute('position', new THREE.BufferAttribute(swPositions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(swColors, 3));
    swParticles = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.018, vertexColors: true, transparent: true, opacity: 0.85,
        blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
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
        if (i >= activeCount) continue;
        const ix = i * 3;
        const t = i / SW_MAX;
        const type = t < protonFrac ? 0 : t < protonFrac + electronFrac ? 1 : 2;
        swPositions[ix] += swVelocities[ix] * sf;
        swPositions[ix + 1] += swVelocities[ix + 1];
        swPositions[ix + 2] += swVelocities[ix + 2];
        const dist = Math.sqrt(swPositions[ix] ** 2 + swPositions[ix + 1] ** 2 + swPositions[ix + 2] ** 2);
        const bowDist = BOW_STANDOFF();
        if (dist < bowDist) {
            const nx = swPositions[ix] / dist, ny = swPositions[ix + 1] / dist, nz = swPositions[ix + 2] / dist;
            swVelocities[ix] += nx * 0.003; swVelocities[ix + 1] += ny * 0.003; swVelocities[ix + 2] += nz * 0.003;
        }
        if (swPositions[ix] < -2 || dist > 12) {
            const spread = type === 2 ? 0.3 : 0.7;
            swPositions[ix] = 6 + Math.random() * 3;
            swPositions[ix + 1] = (Math.random() - 0.5) * spread;
            swPositions[ix + 2] = (Math.random() - 0.5) * spread;
            const sm = type === 2 ? 2.5 : type === 1 ? 1.5 : 1.0;
            swVelocities[ix] = -(0.01 + Math.random() * 0.005) * sm;
            swVelocities[ix + 1] = (Math.random() - 0.5) * 0.001;
            swVelocities[ix + 2] = (Math.random() - 0.5) * 0.001;
        }
    }
    if (swParticles) {
        swParticles.geometry.attributes.position.needsUpdate = true;
        swParticles.geometry.attributes.color.needsUpdate = true;
    }
}

function updateSolarWindData(feeds) {
    if (!feeds) return;
    const sw = feeds.solar_wind_latest || feeds;
    if (sw.speed != null) swSpeed = sw.speed;
    if (sw.density != null) swDensity = sw.density;
    const el = feeds.electron_latest || {};
    if (el.flux != null) swElectronFlux = el.flux;
}

buildSolarWind();

// (comet removed)

// --- COSMIC RAY ---
let crActive = false, crProgress = 0, crCooldown = 0, crRate = 120;
const crTrailPts = 20;
const crPositions = new Float32Array(crTrailPts * 3);
const crGeo = new THREE.BufferGeometry();
crGeo.setAttribute('position', new THREE.BufferAttribute(crPositions, 3));
const crLine = new THREE.Line(crGeo, new THREE.LineBasicMaterial({ color: 0xaaddff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
threeScene.add(crLine);
let crEntry = new THREE.Vector3(), crDir = new THREE.Vector3(), crCharge = 1;

function spawnGCR() {
    const theta = Math.random() * Math.PI * 2, phi = Math.acos(2 * Math.random() - 1);
    crEntry.set(5 * Math.sin(phi) * Math.cos(theta), 5 * Math.cos(phi), 5 * Math.sin(phi) * Math.sin(theta));
    crDir.copy(crEntry).negate().normalize();
    crDir.x += (Math.random() - 0.5) * 0.4; crDir.y += (Math.random() - 0.5) * 0.4; crDir.z += (Math.random() - 0.5) * 0.4;
    crDir.normalize(); crCharge = Math.random() > 0.5 ? 1 : -1; crProgress = 0; crActive = true; crLine.material.opacity = 0.6;
}

function animateGCR() {
    if (!crActive) {
        crCooldown--;
        if (crCooldown <= 0) { spawnGCR(); crCooldown = crRate; }
        crLine.material.opacity *= 0.93; crGeo.attributes.position.needsUpdate = true; return;
    }
    crProgress++;
    const pos = crEntry.clone().addScaledVector(crDir, crProgress * 0.08);
    const radial = pos.clone().normalize();
    const lorentz = new THREE.Vector3().crossVectors(crDir, radial).multiplyScalar(0.003 * crCharge / (pos.length() + 0.5));
    crDir.add(lorentz).normalize();
    for (let i = crTrailPts - 1; i > 0; i--) {
        crPositions[i * 3] = crPositions[(i - 1) * 3];
        crPositions[i * 3 + 1] = crPositions[(i - 1) * 3 + 1];
        crPositions[i * 3 + 2] = crPositions[(i - 1) * 3 + 2];
    }
    crPositions[0] = pos.x; crPositions[1] = pos.y; crPositions[2] = pos.z;
    crGeo.attributes.position.needsUpdate = true;
    if (pos.length() < R * 1.05) { crActive = false; crLine.material.color.set(0xffffff); setTimeout(() => crLine.material.color.set(0xaaddff), 200); }
    if (pos.length() > 8 || crProgress > 200) crActive = false;
}

function updateCRRate(crDeviation) {
    crRate = Math.max(30, Math.floor(120 * (1 - crDeviation / 100 * 2)));
}

// ============================================================
// SIDEBAR DATA UPDATERS (unchanged from original)
// ============================================================
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
    ctx.strokeStyle = '#181833'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    ctx.strokeStyle = '#44ff88'; ctx.lineWidth = 1.2; ctx.beginPath();
    centered.forEach((v, i) => {
        const x = (i / (centered.length - 1)) * w, y = h / 2 - (v / maxAbs) * (h * 0.45);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = 'rgba(68,255,136,0.06)'; ctx.beginPath();
    centered.forEach((v, i) => { const x = (i / (centered.length - 1)) * w, y = h / 2 - (v / maxAbs) * (h * 0.45); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.lineTo(w, h / 2); ctx.lineTo(0, h / 2); ctx.closePath(); ctx.fill();
    const stEl = document.getElementById('seismo-station');
    if (stEl && data.station) stEl.textContent = data.station;
    const timeEl = document.getElementById('seismo-time');
    if (timeEl) timeEl.textContent = data.start_time || '--';
    const ampEl = document.getElementById('seismo-amp');
    if (ampEl) ampEl.textContent = `pk: ${maxAbs.toFixed(0)} counts`;
}

// --- All sidebar updaters ---
function updJellyBall(data) {
    if (!data || data.error) return;
    const jEl = document.getElementById('jb-j');
    if (jEl) { jEl.textContent = data.j_current?.toFixed(3) || '--'; jEl.style.color = data.above_critical ? '#ff4444' : data.gap_pct < 10 ? '#ffaa44' : '#44ff44'; }
    const bar = document.getElementById('jb-bar');
    if (bar) { bar.style.width = Math.min(100, (data.j_current || 0) * 100) + '%'; bar.style.background = data.above_critical ? '#ff4444' : data.gap_pct < 10 ? '#ffaa44' : '#4488ff'; }
    const marker = document.getElementById('jb-jc-marker');
    if (marker) marker.style.width = ((data.j_critical || 0.637) * 100) + '%';
    const gapEl = document.getElementById('jb-gap');
    if (gapEl) { const sign = data.gap > 0 ? '-' : '+'; gapEl.textContent = `${sign}${Math.abs(data.gap_pct || 0).toFixed(1)}%`; }
    const phaseEl = document.getElementById('jb-phase');
    if (phaseEl) {
        phaseEl.textContent = data.phase || '--';
        const p = (data.phase || '').toLowerCase();
        phaseEl.className = 'esc-badge ' + (p.includes('storm') ? 'esc-flare' : p.includes('critical') ? 'esc-active' : p.includes('recovery') ? 'esc-elevated' : 'esc-quiet');
        phaseEl.style.fontSize = '8px'; phaseEl.style.padding = '1px 5px';
    }
    const xiEl = document.getElementById('jb-xi');
    if (xiEl) { const xi = data.correlation_length_km; xiEl.textContent = xi > 1e6 ? `${(xi / 1e6).toFixed(1)}M km` : xi > 1e3 ? `${(xi / 1e3).toFixed(0)}k` : `${xi?.toFixed(0) || '--'}`; xiEl.style.color = xi > 1e6 ? '#ff4444' : xi > 1e5 ? '#ffaa44' : '#44aaff'; }
    const shieldEl = document.getElementById('jb-shield');
    if (shieldEl) { shieldEl.textContent = data.shield || '--'; shieldEl.style.color = data.shield === 'ON' ? '#4f4' : data.shield === 'OFF' ? '#f44' : '#ff4'; }
    const solarEl = document.getElementById('h-solar');
    if (solarEl && data.inputs) { const kp = data.inputs.kp || 0; const solarL2 = Math.min(1, kp / 9); solarEl.textContent = solarL2.toFixed(2); solarEl.style.color = solarL2 > 0.5 ? '#ff8844' : '#556'; }
    const lunarEl = document.getElementById('h-lunar');
    if (lunarEl) { const ref = new Date('2000-01-06T00:00:00Z').getTime(); const phase = ((Date.now() - ref) / 86400000 % 29.53059) / 29.53059; const m2 = Math.abs(Math.cos(2 * Math.PI * phase)); lunarEl.textContent = m2.toFixed(2); lunarEl.style.color = m2 > 0.7 ? '#88aaff' : '#556'; }
    const stormEl = document.getElementById('h-storm');
    if (stormEl && data.gap_pct != null) { const stormL2 = Math.max(0, 1 - Math.abs(data.gap_pct) / 30); stormEl.textContent = stormL2.toFixed(2); stormEl.style.color = stormL2 > 0.5 ? '#44ff88' : '#556'; }
    const detailEl = document.getElementById('jb-detail');
    if (detailEl) detailEl.textContent = data.phase_detail || '--';
    const hydro = data.fault_hydromechanics;
    const dynamic = hydro?.gofar_reference?.dynamic_cycle;
    const brakeEl = document.getElementById('jb-gofar-brake');
    if (brakeEl) brakeEl.textContent = dynamic?.strengthening_pct != null ? `+${dynamic.strengthening_pct.toFixed(0)}% effective stress` : '--';
    const hydroDetailEl = document.getElementById('jb-hydro-detail');
    if (hydroDetailEl && hydro) hydroDetailEl.textContent = hydro.global_zone_ratios_modified ? 'Global ratios recalibrated' : 'Gofar modeled reference; global ratios unchanged';
}

let nnData = null, nnPhase = 'compression';
function updNeural(data) {
    if (!data || data.error) return;
    nnData = data;
    renderNeuralZones();
    const modes = data.diagnostics?.mode_amplitudes;
    if (modes) {
        for (let l = 1; l <= 6; l++) {
            const val = modes[`l${l}`] ?? 0;
            const bar = document.getElementById(`mode-l${l}`);
            const score = document.getElementById(`ms-l${l}`);
            if (bar) { bar.style.width = Math.min(100, Math.abs(val) / 5 * 100) + '%'; bar.style.background = val > 0 ? (l === 2 ? '#ff8844' : l === 3 ? '#44aaff' : '#44ff88') : (l === 2 ? '#ff4466' : '#6644aa'); }
            if (score) score.textContent = (val > 0 ? '+' : '') + val.toFixed(2);
        }
    }
    const bivEl = document.getElementById('nn-biv');
    if (bivEl && data.diagnostics?.bivector_norm != null) bivEl.textContent = data.diagnostics.bivector_norm.toFixed(1);
}

function renderNeuralZones() {
    if (!nnData?.predictions?.[nnPhase]) return;
    const zones = nnData.predictions[nnPhase];
    const container = document.getElementById('nn-zones');
    if (!container) return;
    container.innerHTML = Object.entries(zones).map(([name, ratio]) => {
        const pct = Math.min(100, Math.max(0, (ratio - 0.2) / 4.8 * 100));
        const color = ratio > 1.5 ? '#ff4444' : ratio > 1.1 ? '#ffaa44' : ratio > 0.9 ? '#44ff44' : '#4488ff';
        return `<div class="det-row"><span class="det-label">${name}</span><div class="det-bar-bg"><div class="det-bar" style="width:${pct}%;background:${color}"></div></div><span class="det-score" style="color:${color}">${ratio.toFixed(2)}</span></div>`;
    }).join('');
}

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

function updFieldStrengths(data) {
    if (!data || data.error) return;
    const fwf = data.fair_weather_ez;
    if (fwf) { document.getElementById('fv-fwf').textContent = `${fwf.value} V/m`; document.getElementById('fb-fwf').style.width = Math.min(100, (fwf.value / 250) * 100) + '%'; const stEz = document.getElementById('st-ez'); if (stEz) stEz.textContent = `${fwf.value} V/m`; }
    const tel = data.telluric_j;
    if (tel) { document.getElementById('fv-telluric').textContent = `${tel.value} mA/km`; document.getElementById('fb-telluric').style.width = Math.min(100, (tel.value / 100) * 100) + '%'; document.getElementById('fb-telluric').style.background = tel.value > 20 ? '#ff4444' : '#ff8844'; }
    const man = data.mansurov_dbdt;
    if (man) { document.getElementById('fv-mansurov').textContent = `${man.value} nT/hr`; document.getElementById('fb-mansurov').style.width = Math.min(100, (man.value / 50) * 100) + '%'; }
    const sch = data.schumann_f1;
    if (sch) { document.getElementById('fv-schumann').textContent = `${sch.value} Hz`; document.getElementById('fb-schumann').style.width = Math.min(100, (sch.value / 8.5) * 100) + '%'; }
    const gic = data.gic_risk;
    if (gic) { document.getElementById('fv-gic').textContent = gic.label; document.getElementById('fb-gic').style.width = (gic.score * 100) + '%'; document.getElementById('fb-gic').style.background = gic.score > 0.5 ? '#ff4444' : gic.score > 0.2 ? '#ffaa44' : '#44ff44'; document.getElementById('fv-gic').style.color = gic.score > 0.5 ? '#ff4444' : gic.score > 0.2 ? '#ffaa44' : '#44ff44'; }
}

function updKp(d) { if (!d?.current) return; const k = d.current, el = document.getElementById('kp-metric'); el.textContent = `Kp ${k.toFixed(0)}`; el.className = 'm ' + (k < 4 ? 'q' : k < 6 ? 'a' : 's'); const s = document.getElementById('st-kp'); s.textContent = k.toFixed(0); s.className = 'v ' + (k < 4 ? 'g' : k < 6 ? '' : 'w'); }
function updSW(d) { if (!d) return; if (d.current_bz != null) { const e = document.getElementById('st-bz'); e.textContent = `${d.current_bz.toFixed(1)}`; e.className = 'v ' + (d.current_bz < -10 ? 'w' : 'g'); } if (d.current_speed != null) { const e = document.getElementById('st-vsw'); e.textContent = `${d.current_speed.toFixed(0)}`; e.className = 'v ' + (d.current_speed > 600 ? 'w' : 'g'); } drawChart('sw-chart', d.bz, { color: '#ff6666', fillNeg: true, dec: 1 }); }
function updXRS(d) { if (!d) return; if (d.current_flux) { const f = d.current_flux; const cl = f >= 1e-4 ? `X${(f / 1e-4).toFixed(1)}` : f >= 1e-5 ? `M${(f / 1e-5).toFixed(1)}` : f >= 1e-6 ? `C${(f / 1e-6).toFixed(1)}` : 'B'; document.getElementById('st-xrs').textContent = cl; } drawXRS(d); const el = document.getElementById('op-state'); el.textContent = d.state || '?'; el.className = 'state ' + (d.state === 'FALLING' ? 'falling' : d.state === 'RISING' ? 'rising' : 'stable'); }
function updSun(d) { if (!d?.images) return; window._si = d.images; const img = document.getElementById('sun-image'); if (!img.dataset.loaded) { img.src = d.images.eit_195 || Object.values(d.images)[0]; img.dataset.loaded = '1'; } }
function updLunar(d) { if (!d) return; document.getElementById('lunar-metric').textContent = `${d.name}`; document.getElementById('lunar-detail').textContent = `${d.illumination}% | F:${d.tidal_force.toFixed(2)} | dF:${d.tidal_rate.toFixed(2)}`; document.getElementById('st-moon').textContent = `${d.illumination}%`; }
function updCR(d) { if (!d?.stations) return; const ks = Object.keys(d.stations); if (!ks.length) return; const avg = ks.reduce((s, k) => s + d.stations[k].deviation_pct, 0) / ks.length; const el = document.getElementById('cr-metric'); el.textContent = `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%`; el.className = 'm ' + (d.forbush_detected ? 's' : 'q'); document.getElementById('cr-detail').textContent = d.forbush_detected ? 'FORBUSH DECREASE' : `${ks.length} stations nominal`; document.getElementById('st-cr').textContent = `${avg > 0 ? '+' : ''}${avg.toFixed(1)}%`; document.getElementById('st-cr').className = 'v ' + (d.forbush_detected ? 'w' : 'g'); }
function updGlobalCR(d) { if (!d || d.error) return; const stEl = document.getElementById('cr-stations'); if (stEl) stEl.textContent = `${d.n_stations || 0} stations`; if (d.global_mean != null) { const el = document.getElementById('cr-metric'); if (el) { el.textContent = `${d.global_mean > 0 ? '+' : ''}${d.global_mean.toFixed(1)}%`; el.className = 'm ' + (d.forbush ? 's' : 'q'); } const fb = document.getElementById('cr-forbush'); if (fb) { fb.textContent = d.forbush ? 'FORBUSH DECREASE' : 'nominal'; fb.style.color = d.forbush ? '#f44' : '#4f4'; } } }
function updTEC(d) { if (!d) return; const el = document.getElementById('tec-metric'); const det = document.getElementById('tec-detail'); if (d.available) { if (el) { el.textContent = 'LIVE'; el.className = 'm q'; } if (det) det.textContent = d.dataset || 'USTEC'; } else { if (el) { el.textContent = 'N/A'; el.className = 'm'; } if (det) det.textContent = d.note?.substring(0, 30) || 'unavailable'; } }
function updPrecip(d) { if (!d) return; const el = document.getElementById('precip-metric'); const det = document.getElementById('precip-detail'); if (el) { el.textContent = `${d.global_precip_72h || 0} mm`; el.className = 'm ' + (d.global_precip_72h > 100 ? 'a' : 'q'); } if (det) { const thunder = d.global_thunder_hours || 0; det.textContent = `${d.n_stations || 0} sites | ${thunder} storm-hrs`; } renderWeatherMarkers(d); }
function updLightning(d) { if (!d) return; const el = document.getElementById('lightning-metric'); const det = document.getElementById('lightning-detail'); if (el) { const clim = d.climatology; if (clim) { el.textContent = clim.month; el.style.color = '#ffaa44'; } const rt = d.realtime_thunder_hours || 0; if (rt > 0 && el) el.textContent += ` (${rt}h)`; } if (det && d.climatology?.hotspots) { const top = d.climatology.hotspots.sort((a, b) => b.mean_density - a.mean_density)[0]; det.textContent = top ? `peak: ${top.name}` : 'WWLLN climatology'; } }
function updPorePressure(d) { if (!d?.stations) return; const container = document.getElementById('pore-bars'); if (!container) return; container.innerHTML = d.stations.map(st => { const pp = st.depth_profile?.['100m']; if (!pp) return ''; const pct = pp.pct_tectonic; const barW = Math.min(100, pct * 3000); const color = pct > 0.01 ? '#ff4444' : pct > 0.005 ? '#ffaa44' : '#44aaff'; const name = st.name.split('(')[0].trim().substring(0, 12); return `<div class="det-row"><span class="det-label">${name}</span><div class="det-bar-bg"><div class="det-bar" style="width:${barW}%;background:${color}"></div></div><span class="det-score" style="color:${color}">${pp.total_pa.toFixed(0)}</span></div>`; }).join(''); const tidalEl = document.getElementById('pp-tidal'); if (tidalEl && d.inputs) { tidalEl.textContent = d.inputs.tidal_force > 0 ? 'spring' : 'neap'; tidalEl.style.color = Math.abs(d.inputs.tidal_force) > 0.7 ? '#88aaff' : '#556'; } const jzEl = document.getElementById('pp-jz'); if (jzEl && d.inputs) jzEl.textContent = d.inputs.telluric_j_mA_km?.toFixed(1) || '--'; const nucleationEl = document.getElementById('pp-nucleation'); const first100m = d.stations.find(st => st.depth_profile?.['100m'])?.depth_profile?.['100m']; if (nucleationEl && first100m?.hydromechanical_response) { const response = first100m.hydromechanical_response; nucleationEl.textContent = response.nucleation_tendency === 'promoting' ? 'effective stress down' : response.nucleation_tendency === 'inhibiting' ? 'effective stress up' : 'neutral'; nucleationEl.style.color = response.nucleation_tendency === 'promoting' ? '#ffaa44' : response.nucleation_tendency === 'inhibiting' ? '#44aaff' : '#778'; } }
function updCloudCharge(d) { if (!d?.stations) return; const ezEl = document.getElementById('cc-ez'); const stormEl = document.getElementById('cc-storms'); const chargeEl = document.getElementById('cc-charge'); if (ezEl) { const avgEz = d.stations.reduce((s, st) => s + (st.ez_v_m || 0), 0) / Math.max(d.stations.length, 1); ezEl.textContent = avgEz.toFixed(0); } if (stormEl) stormEl.textContent = d.active_thunderstorms || 0; if (chargeEl) chargeEl.textContent = d.global_charge_c?.toFixed(0) || '0'; const container = document.getElementById('cc-stations'); if (container) { container.innerHTML = d.stations.filter(st => st.charge_c > 0 || st.cloud_cover?.total > 50).map(st => { const cc = st.cloud_cover?.total || 0; const color = st.charge_type === 'Cb dipole' ? '#ffcc44' : st.charge_type === 'convective' ? '#ff8844' : '#4488ff'; return `<div style="display:flex;justify-content:space-between;font-size:8px;margin-bottom:1px;"><span style="color:#778;">${st.name}</span><span style="color:${color};">${st.charge_type} ${st.charge_c > 0 ? st.charge_c + 'C' : cc + '%'}</span></div>`; }).join(''); } renderCloudLayer(d); }
function updDst(data) { if (!data) return; const el = document.getElementById('dst-metric'), st = document.getElementById('st-dst'); if (data.current != null) { el.textContent = `${data.current} nT`; el.className = 'm ' + (data.current > -30 ? 'q' : data.current > -50 ? 'a' : 's'); st.textContent = `${data.current}`; st.className = 'v ' + (data.current > -30 ? 'g' : data.current > -50 ? '' : 'w'); } }

// Sun image selector
document.querySelectorAll('#sun-selector button').forEach(b => {
    b.addEventListener('click', () => {
        document.querySelectorAll('#sun-selector button').forEach(x => x.classList.remove('on'));
        b.classList.add('on');
        if (window._si?.[b.dataset.img]) document.getElementById('sun-image').src = window._si[b.dataset.img] + '?t=' + Date.now();
    });
});

// Paleomag
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
}
document.getElementById('paleomag-toggle')?.addEventListener('change', e => {
    const panel = document.getElementById('paleomag-panel');
    if (panel) panel.style.display = e.target.checked ? 'block' : 'none';
    if (e.target.checked && !palemagData) loadPaleomag();
});

// Clock + time slider
const timeSlider = document.getElementById('time-slider');
const timeVal = document.getElementById('time-val');
const timeLive = document.getElementById('time-live');
let isLive = true, historyHoursBack = 0;
if (timeSlider) {
    document.getElementById('time-control').classList.add('visible');
    timeSlider.addEventListener('input', () => { historyHoursBack = parseInt(timeSlider.value); isLive = historyHoursBack === 0; timeVal.textContent = isLive ? 'LIVE' : `-${historyHoursBack}h`; timeLive?.classList.toggle('on', isLive); });
    timeLive?.addEventListener('click', () => { timeSlider.value = 0; historyHoursBack = 0; isLive = true; timeVal.textContent = 'LIVE'; timeLive.classList.add('on'); });
}
setInterval(() => {
    const now = isLive ? new Date() : new Date(Date.now() - historyHoursBack * 3600000);
    document.getElementById('clock').textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC' + (isLive ? '' : ` (-${historyHoursBack}h)`);
}, 1000);

// Layer toggles
document.querySelectorAll('[data-layer]').forEach(inp => {
    inp.addEventListener('change', () => setLayerVisible(inp.dataset.layer, inp.checked));
});
setTimeout(() => {
    document.querySelectorAll('[data-layer]').forEach(inp => {
        if (!inp.checked) setLayerVisible(inp.dataset.layer, false);
    });
}, 500);

// ============================================================
// MAIN POLL
// ============================================================
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
        fetchJSON('/pore_pressure'),         // 18
        fetchJSON('/cloud_charge'),          // 19
        fetchJSON('/magnetic_anomalies'),    // 20
        fetchJSON('/ocean_light_phenomena'), // 21
    ]);
    const v = i => results[i]?.value;
    if (v(0)) updateEarthquakes(v(0));
    if (v(1)) { updateSubsolar(v(1)); updateJellyBall(v(1)); updateTerminator(v(1)); buildMagneticField(v(1)); buildSolarWindFlow(v(1)); }
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
    if (v(10)) { updFieldStrengths(v(10)); buildTelluricCurrents(v(10)); }
    if (v(11)) drawSeismogram(v(11));
    if (v(12)) updJellyBall(v(12));
    if (v(13)) updNeural(v(13));
    if (v(14)) { updGlobalCR(v(14)); if (v(14).global_mean != null) updateCRRate(v(14).global_mean); }
    if (v(15)) updTEC(v(15));
    if (v(16)) updPrecip(v(16));
    if (v(17)) updLightning(v(17));
    if (v(18)) updPorePressure(v(18));
    if (v(19)) updCloudCharge(v(19));
    if (v(20)) updateMagneticAnomalies(v(20));
    if (v(21)) updateOceanLightPhenomena(v(21));
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

let solarConnected = false;
function updateSolarAvailability(quality) {
    if (!quality) return;
    const ready = quality.alerting_ready === true;
    const status = String(quality.status || (ready ? 'ok' : 'unavailable')).toUpperCase();
    const age = quality.xray?.age_seconds;
    const ageText = typeof age === 'number' ? (age < 120 ? `${age}s old` : `${Math.round(age / 60)}m old`) : 'no XRS timestamp';
    const source = quality.xray?.source || 'NOAA SWPC GOES XRS';
    const freshness = document.getElementById('solar-freshness');
    if (freshness) {
        freshness.textContent = `${status} · ${source} · ${ageText}`;
        freshness.style.color = ready ? (status === 'DEGRADED' ? '#fb4' : '#6b8') : '#f66';
    }
    const dot = document.getElementById('solar-conn');
    if (dot) dot.className = `conn-dot ${solarConnected && ready ? 'live' : 'dead'}`;
    const label = document.getElementById('solar-status');
    if (label) label.textContent = ready ? (solarConnected ? '(LIVE)' : '(polling)') : `(${status})`;
    const panel = document.getElementById('detector-panel');
    if (panel) panel.style.opacity = ready ? '1' : '0.55';
    if (!ready) {
        DET_NAMES.forEach(name => {
            const bar = document.getElementById(`det-${name}`);
            const score = document.getElementById(`ds-${name}`);
            if (bar) bar.style.width = '0%';
            if (score) score.textContent = '--';
        });
        const fusedFill = document.getElementById('fused-fill');
        const fusedLabel = document.getElementById('fused-label');
        const agreement = document.getElementById('det-agreement');
        if (fusedFill) fusedFill.style.width = '0%';
        if (fusedLabel) fusedLabel.textContent = 'FUSED: unavailable';
        if (agreement) agreement.textContent = '--';
        const esc = document.getElementById('esc-state');
        if (esc) { esc.textContent = 'NO DATA'; esc.className = 'esc-badge esc-quiet'; }
        const detail = document.getElementById('esc-detail');
        if (detail) detail.textContent = 'Alerts inhibited until fresh X-ray observations arrive';
    }
}

function updDetectors(data) {
    if (!data || data.error) return;
    const quality = data.data_quality;
    if (quality) updateSolarAvailability(quality);
    if (quality && !quality.alerting_ready) return;
    const diag = data.fusion_diagnostics || data;
    const detectors = diag.raw_scores || diag.detectors || [];
    if (Array.isArray(detectors)) {
        detectors.forEach(d => {
            const name = (d.name || '').toLowerCase().replace(/[_\s-]/g, '');
            const matchName = DET_NAMES.find(n => name.includes(n)) || name;
            const val = d.percentile_rank ?? d.score ?? d.raw_score ?? null;
            const bar = document.getElementById(`det-${matchName}`);
            const scoreEl = document.getElementById(`ds-${matchName}`);
            if (bar && val != null) { bar.style.width = Math.min(100, val * 100) + '%'; bar.style.background = scoreColor(val); }
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
    const agree = data.detector_agreement ?? diag.detector_agreement ?? data.agreement ?? null;
    if (agree != null) { const el = document.getElementById('det-agreement'); if (el) el.textContent = agree; }
}

function updEscalation(data) {
    if (!data || data.error) return;
    if (data.data_quality) updateSolarAvailability(data.data_quality);
    if (data.data_quality && !data.data_quality.alerting_ready) return;
    data = data.escalation || data;
    const level = (data.level_label || data.level || data.state || 'quiet').toLowerCase();
    const el = document.getElementById('esc-state');
    if (el) { el.textContent = level.toUpperCase(); el.className = 'esc-badge esc-' + level; }
    const stE = document.getElementById('st-esc');
    if (stE) { stE.textContent = level.toUpperCase(); stE.className = 'v ' + (level === 'quiet' ? 'g' : level === 'flare' ? 'w' : ''); }
    const detail = document.getElementById('esc-detail');
    if (detail) { const spikes = data.hardness_spikes_in_window ?? data.hardness_spike_count ?? '--'; const peak = data.peak_fused ?? data.peak ?? '--'; detail.textContent = `Spikes: ${spikes} | Peak: ${typeof peak === 'number' ? peak.toFixed(2) : peak}`; }
}

const PW_NAMES = ['forbush', 'heep', 'ssc', 'mansurov', 'lunar'];
const PW_COLORS = { forbush: '#6644ff', heep: '#44ffaa', ssc: '#ff44aa', mansurov: '#ffff44', lunar: '#888' };

function updPathways(data) {
    if (!data || data.error) return;
    if (data.data_quality) updateSolarAvailability(data.data_quality);
    if (data.data_quality && !data.data_quality.alerting_ready) {
        const stressEl = document.getElementById('stress-val');
        if (stressEl) { stressEl.textContent = '--'; stressEl.style.color = '#778'; }
        const panel = document.getElementById('pathway-panel');
        if (panel) panel.style.opacity = '0.55';
        return;
    }
    const panel = document.getElementById('pathway-panel');
    if (panel) panel.style.opacity = '1';
    const stressor = data.stressor || data;
    const pathways = stressor.pathways || (Array.isArray(data) ? data : []);
    let totalStress = 0;
    const pwList = Array.isArray(pathways) ? pathways : Object.values(pathways);
    pwList.forEach(pw => {
        const name = (pw.name || pw.pathway || '').toLowerCase().replace(/[_\s-]/g, '');
        const matchName = PW_NAMES.find(n => name.includes(n));
        if (!matchName) return;
        const score = pw.score ?? pw.value ?? 0;
        const effect = (pw.effect || pw.direction || '').toLowerCase();
        const isSuppression = effect.includes('suppress');
        const bar = document.getElementById(`pw-${matchName}`);
        const dirEl = document.getElementById(`pwd-${matchName}`);
        const scoreEl = document.getElementById(`pws-${matchName}`);
        if (bar) { bar.style.width = Math.min(100, Math.abs(score) * 100) + '%'; bar.style.background = pw.active ? PW_COLORS[matchName] || '#888' : '#333'; }
        if (dirEl) { dirEl.textContent = isSuppression ? '-' : '+'; dirEl.style.color = isSuppression ? '#f44' : '#4f4'; }
        if (scoreEl) scoreEl.textContent = Math.abs(score).toFixed(2);
        totalStress += isSuppression ? -Math.abs(score) : Math.abs(score);
    });
    const stressIdx = stressor.total ?? data.total_stress ?? data.stressor_index ?? totalStress;
    const stressEl = document.getElementById('stress-val');
    if (stressEl) { const val = typeof stressIdx === 'number' ? stressIdx : totalStress; stressEl.textContent = (val >= 0 ? '+' : '') + val.toFixed(2); stressEl.style.color = val > 0.3 ? '#f44' : val > 0 ? '#ff4' : '#4f4'; }
}

// SSE Streams
function connectSSE() {
    try {
        const sse = new EventSource(`${SOLAR_API}/metrics`);
        sse.onopen = () => { solarConnected = true; const st = document.getElementById('solar-status'); if (st) st.textContent = '(connected; awaiting data)'; };
        const handleMetrics = e => { try {
            const d = JSON.parse(e.data);
            if (d.data_quality) updateSolarAvailability(d.data_quality);
            if (d.fusion_diagnostics || d.fused_score != null || d.raw_scores) updDetectors(d);
            if (d.escalation) updEscalation({ escalation: d.escalation, data_quality: d.data_quality });
            else if (d.level || d.level_label) updEscalation(d);
            if (d.stressor?.pathways) updPathways({ ...d.stressor, data_quality: d.data_quality });
            else if (d.pathways || Array.isArray(d)) updPathways(d);
            if (d.bz != null) updateMagnetosphereCompression(d.bz);
        } catch (_) { } };
        // Axum emits named "metrics" events; onmessage only receives unnamed events.
        sse.addEventListener('metrics', handleMetrics);
        sse.onmessage = handleMetrics;
        sse.addEventListener('availability', e => { try { const d = JSON.parse(e.data); updateSolarAvailability({ status: 'unavailable', alerting_ready: false, ...d }); } catch (_) { } });
        sse.onerror = () => { solarConnected = false; const dot = document.getElementById('solar-conn'); if (dot) dot.className = 'conn-dot dead'; const st = document.getElementById('solar-status'); if (st) st.textContent = '(polling)'; };
    } catch (_) { }
    try {
        const alerts = new EventSource(`${SOLAR_API}/alerts`);
        const handleAlert = e => { try {
            const a = JSON.parse(e.data);
            if (a.data_status === 'stale' || a.data_status === 'starting' || a.alerting_ready === false) return;
            const banner = document.getElementById('alert-banner'); if (!banner) return;
            const type = a.alert_type || a.type || a.kind || 'solar_alert', msg = a.message || a.msg || JSON.stringify(a);
            banner.textContent = `${String(type).replaceAll('_', ' ').toUpperCase()}: ${msg}`;
            banner.style.display = 'block';
            banner.style.background = String(type).includes('flare') ? 'rgba(255,50,50,0.95)' : 'rgba(40,160,255,0.95)';
            setTimeout(() => { banner.style.display = 'none'; }, 15000);
        } catch (_) { } };
        alerts.addEventListener('alert', handleAlert);
        alerts.onmessage = handleAlert;
    } catch (_) { }
}
connectSSE();

async function pollSolar() {
    try {
        const [statusResp, feedsResp, healthResp] = await Promise.all([
            fetch(`${SOLAR_API}/status`),
            fetch(`${SOLAR_API}/feeds`),
            fetch(`${SOLAR_API}/health`),
        ]);
        const status = await statusResp.json();
        const feeds = await feedsResp.json();
        const health = await healthResp.json();
        if (health?.data_quality) updateSolarAvailability(health.data_quality);
        if (statusResp.ok && status && !status.error) {
            updDetectors(status);
            if (status.escalation) updEscalation({ escalation: status.escalation, data_quality: status.data_quality });
            if (status.stressor) updPathways({ ...status.stressor, data_quality: status.data_quality });
            if (feedsResp.ok && feeds && !feeds.error) updateSolarWindData(feeds);
            const protonDet = status.fusion_diagnostics?.raw_scores?.find(d => d.name === 'proton');
            if (protonDet) swProtonScore = protonDet.percentile_rank;
            return;
        }
    } catch (_) { }
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

// Cesium has its own render loop — no animation needed.
console.log('Global Resonance: CesiumJS globe initialized');
