const HTML_ENTITIES = Object.freeze({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
});

/** Escape untrusted feed text before inserting it into an HTML template. */
export function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => HTML_ENTITIES[character]);
}
