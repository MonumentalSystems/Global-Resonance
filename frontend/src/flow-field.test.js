import assert from 'node:assert/strict';
import test from 'node:test';

import { colorBucket, sampleVector, seedMovingParticle } from './flow-field.js';

const grid = {
    minLat: 0, maxLat: 1, minLon: 0, maxLon: 1,
    dLat: 1, dLon: 1, nLat: 2, nLon: 2,
    u: [0, 2, 2, 4],
    v: [0, 0, 2, 2],
};

test('sampleVector bilinearly interpolates a vector cell', () => {
    assert.deepEqual(sampleVector(grid, 0.5, 0.5), { u: 2, v: 1 });
});

test('sampleVector rejects cells containing a land/missing value', () => {
    assert.equal(sampleVector({ ...grid, u: [0, null, 2, 4] }, 0.5, 0.5), null);
});

test('sampleVector interpolates periodically across the antimeridian', () => {
    const globalGrid = {
        minLat: 0, maxLat: 1, minLon: -179.875, maxLon: 175.125,
        dLat: 1, dLon: 5, nLat: 2, nLon: 72,
        u: Array(144).fill(0),
        v: Array(144).fill(0),
    };
    for (const row of [0, 1]) {
        globalGrid.u[row * 72 + 71] = 10;
        globalGrid.u[row * 72] = 20;
    }

    const vector = sampleVector(globalGrid, 0.5, 179);
    assert.ok(vector);
    assert.ok(vector.u > 10 && vector.u < 20);
    assert.equal(vector.v, 0);
});

test('seedMovingParticle rejects stagnant cells until moving water is found', () => {
    const values = [0, 0, 0.9, 0.9];
    let index = 0;
    const particle = seedMovingParticle(grid, {
        minimumSpeed: 1,
        random: () => values[index++],
        maxAttempts: 2,
    });
    assert.deepEqual(particle, { lat: 0.9, lon: 0.9 });
});

test('colorBucket clamps values to the nearest display band', () => {
    const stops = [{ s: 0 }, { s: 0.5 }, { s: 1 }, { s: 2 }];
    assert.equal(colorBucket(-1, stops), 0);
    assert.equal(colorBucket(0.75, stops), 1);
    assert.equal(colorBucket(9, stops), 2);
});
