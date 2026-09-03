const { test, expect } = require('@playwright/test');

test('watchwords list empty state disables clear button', async ({ page }) => {
    // Start local server and wait for the page to load
    await page.goto('http://localhost:4173/');

    // Initial state check
    const clearBtn = page.locator('#clear-watchwords');
    const watchwordsList = page.locator('#watchwords-list');

    // First clear any existing watchwords from local storage just in case
    await page.evaluate(() => {
        localStorage.removeItem('watchwords');
    });
    await page.reload();

    // Verify button is disabled and has empty state
    await expect(clearBtn).toBeDisabled();
    await expect(clearBtn).toHaveAttribute('title', 'No watchwords to clear');
    await expect(watchwordsList.locator('.placeholder')).toBeVisible();

    // Add a watchword
    await page.fill('#watchword-input', 'testword');
    await page.click('#add-watchword');

    // Verify button is enabled now
    await expect(clearBtn).not.toBeDisabled();
    await expect(clearBtn).toHaveAttribute('title', 'Clear all watchwords');
    await expect(watchwordsList.locator('.placeholder')).not.toBeVisible();

    // Clear via button
    page.once('dialog', dialog => dialog.accept());
    await clearBtn.click();

    // Verify empty state is back
    await expect(clearBtn).toBeDisabled();
    await expect(clearBtn).toHaveAttribute('title', 'No watchwords to clear');
    await expect(watchwordsList.locator('.placeholder')).toBeVisible();
});
