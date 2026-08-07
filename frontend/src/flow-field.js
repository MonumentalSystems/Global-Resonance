export function sampleVector(grid, lat, lon) {
    if (!grid || lat < grid.minLat || lat > grid.maxLat) return null;

    // Global grids store each longitude meridian once. Treat longitude as a
    // periodic axis so the final column interpolates across the antimeridian
    // to column zero instead of leaving a half-cell gap at either edge.
    const period = grid.dLon * grid.nLon;
    if (!Number.isFinite(period) || period <= 0) return null;
    const wrappedLon = grid.minLon + (((lon - grid.minLon) % period) + period) % period;

    const row = (lat - grid.minLat) / grid.dLat;
    const col = (wrappedLon - grid.minLon) / grid.dLon;
    const row0 = Math.floor(row);
    const row1 = Math.min(grid.nLat - 1, row0 + 1);
    const col0 = Math.floor(col) % grid.nLon;
    const col1 = (col0 + 1) % grid.nLon;
    if (row0 < 0 || row1 >= grid.nLat || col0 < 0 || col0 >= grid.nLon) return null;

    const at = (values, r, c) => values[r * grid.nLon + c];
    const values = [
        at(grid.u, row0, col0), at(grid.v, row0, col0),
        at(grid.u, row1, col0), at(grid.v, row1, col0),
        at(grid.u, row0, col1), at(grid.v, row0, col1),
        at(grid.u, row1, col1), at(grid.v, row1, col1),
    ];
    if (values.some(value => value == null || !Number.isFinite(value))) return null;

    const [u00, v00, u10, v10, u01, v01, u11, v11] = values;
    const rowMix = row - row0;
    const colMix = col - col0;
    const u0 = u00 * (1 - rowMix) + u10 * rowMix;
    const u1 = u01 * (1 - rowMix) + u11 * rowMix;
    const v0 = v00 * (1 - rowMix) + v10 * rowMix;
    const v1 = v01 * (1 - rowMix) + v11 * rowMix;
    return {
        u: u0 * (1 - colMix) + u1 * colMix,
        v: v0 * (1 - colMix) + v1 * colMix,
    };
}

export function seedMovingParticle(grid, options = {}) {
    const random = options.random || Math.random;
    const minimumSpeed = options.minimumSpeed ?? 0;
    const maxAttempts = options.maxAttempts ?? 16;
    let candidate = null;

    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        candidate = {
            lat: grid.minLat + random() * (grid.maxLat - grid.minLat),
            lon: grid.minLon + random() * (grid.maxLon - grid.minLon),
        };
        const vector = sampleVector(grid, candidate.lat, candidate.lon);
        if (vector && Math.hypot(vector.u, vector.v) >= minimumSpeed) return candidate;
    }
    return candidate;
}

export function colorBucket(speed, stops) {
    const value = Number.isFinite(speed) ? speed : 0;
    for (let index = 0; index < stops.length - 1; index += 1) {
        if (value <= stops[index + 1].s) return index;
    }
    return Math.max(0, stops.length - 2);
}
