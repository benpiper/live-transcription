const { test, expect } = require('@playwright/test');

test('confirm clear watchwords', async ({ page }) => {
    await page.goto('http://localhost:3000/');

    // Add a watchword
    await page.fill('#watchword-input', 'test');
    await page.click('#add-watchword');

    // Setup dialog handler
    let dialogAppeared = false;
    page.on('dialog', async dialog => {
        dialogAppeared = true;
        expect(dialog.message()).toContain('Are you sure');
        await dialog.accept();
    });

    // Click clear
    await page.click('#clear-watchwords');

    // Verify dialog appeared
    expect(dialogAppeared).toBe(true);
});
