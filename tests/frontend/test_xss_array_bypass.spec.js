const { test, expect } = require('@playwright/test');

test('XSS Array Bypass Prevention', async ({ page }) => {
  await page.goto('/');

  // Insert payload via an array directly to addTranscriptItem
  await page.evaluate(() => {
    addTranscriptItem({
      speaker: ["Attacker", "<script>window.XSS_TRIGGERED = true;</script>"],
      text: ["Normal text", "<img src=x onerror=window.XSS_TRIGGERED=true>"],
      confidence: 0.99,
      timestamp: "12:00:00"
    });
  });

  // Verify the script didn't execute
  const xssTriggered = await page.evaluate(() => window.XSS_TRIGGERED || false);
  expect(xssTriggered).toBe(false);

  // Verify the HTML is properly escaped
  const feedHtml = await page.evaluate(() => document.getElementById('transcript-feed').innerHTML);
  expect(feedHtml).not.toContain('<script>');
  expect(feedHtml).toContain('&lt;script&gt;');
  expect(feedHtml).not.toContain('<img src=x');
  expect(feedHtml).toContain('&lt;img src=x');
});
