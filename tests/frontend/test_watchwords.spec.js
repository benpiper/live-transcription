const { test, expect } = require('@playwright/test');

test('Watchword highlights work efficiently', async ({ page }) => {
  // Navigate to the live preview (will be started in a separate command)
  await page.goto('/');

  // Verify the page loads
  await expect(page).toHaveTitle(/Live Transcription/);

  // Add a watchword
  await page.fill('#watchword-input', 'testword');
  await page.click('#add-watchword');

  // Add a mock transcript containing the watchword using page.evaluate
  // Since we don't have a real WebSocket connection in the static server, we can directly call addTranscriptItem
  await page.evaluate(() => {
    // Need to set sessionLoaded = true so the function processes the item
    sessionLoaded = true;
    addTranscriptItem({
      speaker: 'Speaker A',
      text: 'This is a message with the testword inside.',
      timestamp: '12:00:00',
      confidence: -0.1
    });
  });

  // Verify the item is highlighted
  const highlight = page.locator('.watchword-highlight');
  await expect(highlight).toBeVisible();
  await expect(highlight).toHaveText('testword');

  // Ensure highlight class is added to the transcript item
  const item = page.locator('.transcript-item').last();
  await expect(item).toHaveClass(/highlight/);
});
