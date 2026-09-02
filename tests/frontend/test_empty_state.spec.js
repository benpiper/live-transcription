const { test, expect } = require('@playwright/test');

test('Watchwords list shows empty state and disables Clear All', async ({ page }) => {
    await page.goto('/');

    // Evaluate script to clear watchwords and render
    await page.evaluate(() => {
        watchwords = [];
        renderWatchwords();
    });

    // Check for the placeholder
    const list = page.locator('#watchwords-list');
    await expect(list.locator('.placeholder')).toHaveText('No watchwords added');

    // Check if Clear All is disabled and has correct title
    const clearBtn = page.locator('#clear-watchwords');
    await expect(clearBtn).toBeDisabled();
    await expect(clearBtn).toHaveAttribute('title', 'No watchwords to clear');

    // Add a watchword and check if it updates properly
    await page.evaluate(() => {
        watchwords = ['test'];
        renderWatchwords();
    });

    // The placeholder should be gone
    await expect(list.locator('.placeholder')).toHaveCount(0);

    // Clear All should be enabled
    await expect(clearBtn).toBeEnabled();
    await expect(clearBtn).toHaveAttribute('title', 'Clear all watchwords');
});
