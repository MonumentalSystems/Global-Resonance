import assert from 'node:assert/strict';
import test from 'node:test';

import { escapeHtml } from './dom-safe.js';

test('escapeHtml neutralizes feed text and attribute injection', () => {
    assert.equal(
        escapeHtml('<img src=x onerror="alert(1)"> O\'Brien & Co.'),
        '&lt;img src=x onerror=&quot;alert(1)&quot;&gt; O&#39;Brien &amp; Co.',
    );
});
