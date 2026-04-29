const { test, expect } = require('@playwright/test');

async function installBrowserStubs(page) {
  await page.addInitScript(() => {
    class MockWebSocket {
      constructor(url) {
        this.url = url;
        this.readyState = MockWebSocket.CONNECTING;
        this.binaryType = 'blob';
        window.__mockWebSockets.push(this);
        setTimeout(() => {
          this.readyState = MockWebSocket.OPEN;
          this.onopen?.({ target: this });
        }, 0);
      }

      send(data) {
        this.sent = this.sent || [];
        this.sent.push(data);
      }

      close(code = 1000, reason = '') {
        this.readyState = MockWebSocket.CLOSED;
        this.onclose?.({ code, reason, target: this });
      }

      receiveJson(data) {
        this.onmessage?.({ data: JSON.stringify(data), target: this });
      }

      receiveBytes(buffer) {
        this.onmessage?.({ data: buffer, target: this });
      }
    }

    MockWebSocket.CONNECTING = 0;
    MockWebSocket.OPEN = 1;
    MockWebSocket.CLOSING = 2;
    MockWebSocket.CLOSED = 3;

    window.__mockWebSockets = [];
    window.WebSocket = MockWebSocket;

    window.Notification = {
      permission: 'denied',
      requestPermission: async () => 'denied',
    };

    Object.defineProperty(Navigator.prototype, 'mediaSession', {
      configurable: true,
      value: {
      metadata: null,
      playbackState: 'none',
      setActionHandler() {},
      },
    });
    window.MediaMetadata = class {
      constructor(metadata) {
        Object.assign(this, metadata);
      }
    };
  });
}

async function mockDashboardApi(page, sessionResponse) {
  let currentSessionCalls = 0;

  await page.route('**/css2?**', route => route.fulfill({ status: 200, body: '' }));
  await page.route('**/silence.wav', route => route.fulfill({
    status: 204,
    contentType: 'audio/wav',
    body: '',
  }));
  await page.route('**/api/session/current', route => {
    currentSessionCalls += 1;
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(sessionResponse),
    });
  });
  await page.route('**/api/engine/status', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      device: 'cpu',
      intended_device: 'cpu',
      is_fallback: false,
      model_size: 'tiny.en',
      compute_type: 'int8',
      fallback_count: 0,
    }),
  }));
  await page.route('**/api/audio/buffer-status', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      window_size_sec: 7200,
      buffer_size_bytes: 0,
      buffer_size_mb: 0,
      oldest_timestamp: null,
      newest_timestamp: null,
      num_chunks: 0,
      duration_available_sec: 0,
    }),
  }));

  return {
    getCurrentSessionCalls: () => currentSessionCalls,
  };
}

async function openDashboard(page, sessionResponse) {
  await installBrowserStubs(page);
  const api = await mockDashboardApi(page, sessionResponse);
  const consoleErrors = [];
  const pageErrors = [];

  page.on('console', message => {
    if (message.type() === 'error') {
      consoleErrors.push(message.text());
    }
  });
  page.on('pageerror', error => pageErrors.push(error.message));

  await page.goto('/');
  await expect(page.locator('#status-badge')).toHaveText('Connected');

  return { api, consoleErrors, pageErrors };
}

test.describe('dashboard initial load', () => {
  test('loads an empty session state without console errors', async ({ page }) => {
    const { api, consoleErrors, pageErrors } = await openDashboard(page, {
      active: false,
    });

    await expect(page.locator('#transcript-feed .placeholder')).toHaveText('Waiting for incoming audio...');
    await expect(page.locator('#speaker-count')).toHaveText('0 Speakers Detected');
    await expect(page.locator('#engine-stat')).toHaveText('CPU');
    await expect(page.locator('#buffer-size-stat')).toHaveText('0.0MB');
    await expect.poll(api.getCurrentSessionCalls).toBe(1);
    expect(consoleErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('renders existing session transcripts in order', async ({ page }) => {
    await openDashboard(page, {
      active: true,
      name: 'Morning Shift',
      created_at: '2026-04-29T08:00:00',
      updated_at: '2026-04-29T08:02:00',
      transcripts: [
        {
          timestamp: '2026-04-29 08:00:00',
          origin_time: 1777459200.1,
          duration: 1.6,
          speaker: 'Dispatcher',
          text: 'Engine 4 respond to Main Street',
          confidence: -0.22,
        },
        {
          timestamp: '2026-04-29 08:00:03',
          origin_time: 1777459203.2,
          duration: 2.1,
          speaker: 'Engine 4',
          text: 'Engine 4 en route',
          confidence: -0.31,
        },
      ],
    });

    const items = page.locator('#transcript-feed .transcript-item');
    await expect(items).toHaveCount(2);
    await expect(items.nth(0).locator('.speaker')).toHaveText('Dispatcher');
    await expect(items.nth(0).locator('.transcript-text')).toHaveText('Engine 4 respond to Main Street');
    await expect(items.nth(1).locator('.speaker')).toHaveText('Engine 4');
    await expect(items.nth(1).locator('.transcript-text')).toHaveText('Engine 4 en route');
    await expect(page.locator('#speaker-count')).toHaveText('2 Speakers Detected');
    await expect(page.locator('#speaker-filters .dropdown-item')).toHaveCount(2);
  });
});

test.describe('dashboard WebSocket transcript flow', () => {
  test('appends a transcript received over the WebSocket', async ({ page }) => {
    await openDashboard(page, { active: false });

    await page.evaluate(() => {
      window.__mockWebSockets[0].receiveJson({
        type: 'transcript',
        timestamp: '2026-04-29 09:15:00',
        origin_time: 1777463700.5,
        duration: 2.4,
        speaker: 'Medic 2',
        text: 'Medic 2 arriving on scene',
        confidence: -0.18,
        processing_time: 0.4,
      });
    });

    const items = page.locator('#transcript-feed .transcript-item');
    await expect(items).toHaveCount(1);
    await expect(page.locator('#transcript-feed .placeholder')).toHaveCount(0);
    await expect(items.first().locator('.speaker')).toHaveText('Medic 2');
    await expect(items.first().locator('.transcript-text')).toHaveText('Medic 2 arriving on scene');
    await expect(items.first().locator('.confidence')).toHaveText('-0.18');
    await expect(items.first().locator('.timestamp')).toContainText('2026-04-29 09:15:00');
    await expect(items.first().locator('.timestamp')).toContainText('(2.4s)');
    await expect(page.locator('#latency-stat')).toContainText('ms');
    await expect(page.locator('#process-time-stat')).toHaveText('400ms');
  });

  test('does not double-render loaded session transcripts when live messages arrive', async ({ page }) => {
    await openDashboard(page, {
      active: true,
      name: 'Existing Session',
      created_at: '2026-04-29T08:00:00',
      updated_at: '2026-04-29T08:00:00',
      transcripts: [
        {
          timestamp: '2026-04-29 08:00:00',
          origin_time: 1777459200.1,
          duration: 1.2,
          speaker: 'Dispatcher',
          text: 'Initial dispatch message',
          confidence: -0.2,
        },
      ],
    });

    await page.evaluate(() => {
      window.__mockWebSockets[0].receiveJson({
        type: 'transcript',
        timestamp: '2026-04-29 08:00:10',
        origin_time: 1777459210.1,
        duration: 1.8,
        speaker: 'Unit 1',
        text: 'Unit 1 copies',
        confidence: -0.24,
      });
    });

    const items = page.locator('#transcript-feed .transcript-item');
    await expect(items).toHaveCount(2);
    await expect(page.locator('#transcript-feed .transcript-text', { hasText: 'Initial dispatch message' })).toHaveCount(1);
    await expect(page.locator('#transcript-feed .transcript-text', { hasText: 'Unit 1 copies' })).toHaveCount(1);
  });
});
