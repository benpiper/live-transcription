import { test, expect } from '@playwright/test';
import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  // Take initial screenshot (should have disabled clear button and "No watchwords added" message)
  await page.screenshot({ path: 'watchword_empty_state.png' });

  // Type into the input box to enable the Add button
  await page.fill('#watchword-input', 'test');
  await page.waitForTimeout(500);

  // Click add watchword
  await page.click('#add-watchword');
  await page.waitForTimeout(1000);

  // Screenshot after adding watchword (should show clear button enabled and "test" tag)
  await page.screenshot({ path: 'watchword_added.png' });

  // Click clear watchwords
  await page.click('#clear-watchwords', { force: true });
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'watchword_cleared.png' });

  await browser.close();
})();
