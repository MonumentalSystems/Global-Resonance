// SPIKE ONLY -- deliberately NOT imported by main.js. Answers one question
// before any integration work: does cesium-wind-layer render OUR global grid
// (lat -64.9..80.1, lon -179.9..175.1) correctly, or does it hit the
// pole-distortion and empty-render failure modes its issue tracker documents
// for global extent?
//
// HOW TO RUN. Needs a REAL GPU -- headless SwiftShader renders nothing at all
// (verified: WebGL context alive, tilesLoaded false, 0/19321 canvas pixels lit),
// so a blank frame there is uninformative about the layer. In a normal browser
// on the dev server, paste into the console:
//
//   const { windSpike } = await import('/src/wind-spike.js');
//   console.log(await windSpike(window.viewer));
//
// Then LOOK at the globe. The numbers only prove construction succeeded; the
// documented failure mode is rendering nothing while reporting success, so the
// visual check is the actual test. Watch specifically for: no particles at all,
// particles offset further from the equator, and behaviour near +-80 lat.
import { WindLayer } from 'cesium-wind-layer';

export async function windSpike(viewer) {
    const res = await fetch('/api/ocean_currents');
    const payload = await res.json();
    const g = payload.grid;

    const lats = g.lats;
    const lons = g.lons;
    const width = lons.length;
    const height = lats.length;

    // The library wants dense arrays; our grid carries nulls over land. Zero
    // them so a null cannot be mistaken for a velocity, and record how many so
    // a blank render can be told apart from genuinely absent data.
    let nulls = 0;
    const u = new Array(width * height);
    const v = new Array(width * height);
    for (let i = 0; i < width * height; i += 1) {
        const uu = g.u[i];
        const vv = g.v[i];
        if (uu == null || vv == null) { nulls += 1; u[i] = 0; v[i] = 0; }
        else { u[i] = uu; v[i] = vv; }
    }

    const finite = u.filter(Number.isFinite).length;
    const uMin = Math.min(...u), uMax = Math.max(...u);
    const vMin = Math.min(...v), vMax = Math.max(...v);

    const data = {
        u: { array: Float32Array.from(u), min: uMin, max: uMax },
        v: { array: Float32Array.from(v), min: vMin, max: vMax },
        width,
        height,
        bounds: {
            west: lons[0],
            south: lats[0],
            east: lons[lons.length - 1],
            north: lats[lats.length - 1],
        },
    };

    const layer = new WindLayer(viewer, data, {
        particlesTextureSize: 128,
        lineWidth: { min: 1, max: 2 },
        speedFactor: 8,
        dropRate: 0.003,
        dropRateBump: 0.01,
        colors: ['#4488ff', '#44ffaa', '#ffdd44'],
        flipY: false,
    });

    return {
        gridWidth: width,
        gridHeight: height,
        cells: width * height,
        nullCells: nulls,
        finiteU: finite,
        uRange: [uMin, uMax],
        vRange: [vMin, vMax],
        bounds: data.bounds,
        layerCreated: !!layer,
        layer,
    };
}

if (typeof window !== 'undefined') window.__windSpike = windSpike;
