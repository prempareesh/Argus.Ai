const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('fs');
const path = require('path');

// Mock req and res helpers
function createReqRes(method = 'GET', body = {}, headers = {}) {
    const req = {
        method,
        body,
        headers
    };
    const res = {
        statusCode: 200,
        headers: {},
        body: null,
        setHeader(k, v) { this.headers[k.toLowerCase()] = v; },
        status(code) { this.statusCode = code; return this; },
        json(data) { this.body = data; return this; },
        end() { return this; }
    };
    return { req, res };
}

test('Vision API Gateway (/api/vision)', async (t) => {
    // Load handler
    const handler = require('../api/vision.js');

    await t.test('1. OPTIONS returns 200 with CORS headers', async () => {
        const { req, res } = createReqRes('OPTIONS');
        await handler(req, res);
        assert.equal(res.statusCode, 200);
        assert.equal(res.headers['access-control-allow-origin'], '*');
        assert.ok(res.headers['access-control-allow-methods'].includes('POST'));
    });

    await t.test('2. GET returns 200 health check status', async () => {
        const { req, res } = createReqRes('GET');
        await handler(req, res);
        assert.equal(res.statusCode, 200);
        assert.equal(res.body.status, 'online');
        assert.equal(res.body.service, 'Argus Vision AI Gateway');
        assert.ok(res.body.primaryModel.includes('nemotron'));
    });

    await t.test('3. Unsupported method PUT returns 405 Method Not Allowed', async () => {
        const { req, res } = createReqRes('PUT');
        await handler(req, res);
        assert.equal(res.statusCode, 405);
        assert.equal(res.body.success, false);
    });

    await t.test('4. POST without image returns 400 Bad Request', async () => {
        const { req, res } = createReqRes('POST', {});
        await handler(req, res);
        assert.equal(res.statusCode, 400);
        assert.equal(res.body.success, false);
        assert.ok(res.body.error.includes('Missing required "image" field'));
    });

    await t.test('5. POST with invalid image format returns 400 Bad Request', async () => {
        const { req, res } = createReqRes('POST', { image: 'not-a-data-url' });
        await handler(req, res);
        assert.equal(res.statusCode, 400);
        assert.equal(res.body.success, false);
        assert.ok(res.body.error.includes('Invalid image format'));
    });

    await t.test('6. POST with valid image queries NVIDIA Vision NIM and returns 200 description', async () => {
        const imgPath = path.join(__dirname, '..', 'assets', 'img', 'live-view-pathway.jpg');
        const imgBase64 = fs.readFileSync(imgPath).toString('base64');
        const dataUrl = `data:image/jpeg;base64,${imgBase64}`;

        const { req, res } = createReqRes('POST', {
            image: dataUrl,
            mode: 'scene'
        });

        await handler(req, res);
        assert.equal(res.statusCode, 200);
        assert.equal(res.body.success, true);
        assert.ok(typeof res.body.description === 'string' && res.body.description.length > 10);
        assert.ok(res.body.model.includes('nemotron'));
    });
});
