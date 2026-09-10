const { test, expect } = require('@playwright/test');

test('Watchwords, Audio Processing and Speaker Filters reset buttons have disabled states', async ({ page }) => {
  await page.goto('http://127.0.0.1:4173'); // Using web server port from config

  // 1. Watchwords
  const clearWatchwords = page.locator('#clear-watchwords');
  await expect(clearWatchwords).toBeDisabled();
  await expect(clearWatchwords).toHaveAttribute('title', 'No watchwords to clear');

  // Add a watchword
  await page.fill('#watchword-input', 'hello');
  await page.click('#add-watchword');

  await expect(clearWatchwords).toBeEnabled();
  await expect(clearWatchwords).toHaveAttribute('title', 'Clear all watchwords');

  // 2. Audio Processing
  const resetAudio = page.locator('#audio-processing-reset');
  await expect(resetAudio).toBeDisabled();
  await expect(resetAudio).toHaveAttribute('title', 'Processing already at default settings');

  // Change a setting
  await page.click('#audio-processing-collapse'); await page.fill('#compressor-ratio', '10'); await page.dispatchEvent('#compressor-ratio', 'input');
  await expect(resetAudio).toBeEnabled();
  await expect(resetAudio).toHaveAttribute('title', 'Reset audio processing to defaults');

  // 3. Speaker Filters
  const resetFilters = page.locator('#reset-filters');
  await expect(resetFilters).toBeDisabled();
  await expect(resetFilters).toHaveAttribute('title', 'No filters active');

  // Add some speakers via evaluate to mock it
  await page.evaluate(() => {
    speakers = ['Speaker 1', 'Speaker 2'];
    renderSpeakerFilters();
  });

  // Deselect all
  await page.click('#speaker-dropdown-toggle');
  await page.click('#deselect-all-speakers');

  await expect(resetFilters).toBeEnabled();
  await expect(resetFilters).toHaveAttribute('title', 'Clear speaker filters');
});
