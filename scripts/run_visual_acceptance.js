/**
 * Local Integration and Visual Acceptance Testing Suite
 *
 * Uses Playwright with system Google Chrome to visit every page of the React application
 * at Desktop (1440x900) and Tablet (768x1024) resolutions, capturing full-page screenshots,
 * asserting analytical integrity (including the Tallur regression test), testing map
 * interactions, testing community filtering, and verifying assistant tool execution.
 */

const { chromium } = require('../frontend/node_modules/playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5173';
const API_URL = 'http://localhost:8000';
const CHROME_PATH = '/usr/bin/google-chrome';
const SCREENSHOT_DIR = path.resolve(__dirname, '../reports/acceptance_screenshots');

// Ensure screenshot directories exist
fs.mkdirSync(path.join(SCREENSHOT_DIR, 'desktop'), { recursive: true });
fs.mkdirSync(path.join(SCREENSHOT_DIR, 'tablet'), { recursive: true });

async function runAcceptanceTests() {
  console.log('===============================================================');
  console.log('Cyclone Preparedness Platform: Visual Acceptance & Integration');
  console.log('===============================================================');

  const report = {
    timestamp: new Date().toISOString(),
    pagesTested: [],
    analyticalChecks: {},
    assistantChecks: {},
    consoleErrors: [],
    failedRequests: [],
    screenshots: [],
    passed: true,
  };

  const browser = await chromium.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const viewports = [
    { name: 'desktop', width: 1440, height: 900 },
    { name: 'tablet', width: 768, height: 1024 },
  ];

  for (const vp of viewports) {
    console.log(`\n--- Running tests for Viewport: ${vp.name.toUpperCase()} (${vp.width}x${vp.height}) ---`);

    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: 1,
    });

    const page = await context.newPage();

    // Listen for console errors and failed requests
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Ignore expected benign warnings
        if (!text.includes('React Router Future Flag') && !text.includes('ResizeObserver')) {
          console.error(`[Browser Console Error] [${vp.name}]:`, text);
          report.consoleErrors.push({ viewport: vp.name, error: text });
        }
      }
    });

    page.on('requestfailed', (req) => {
      console.warn(`[Network Request Failed] [${vp.name}]:`, req.url(), req.failure()?.errorText);
      report.failedRequests.push({ viewport: vp.name, url: req.url(), error: req.failure()?.errorText });
    });

    const pagesToTest = [
      { route: '/', name: '01_command_center_dashboard' },
      { route: '/map', name: '02_geospatial_intelligence_map' },
      { route: '/communities', name: '03_community_explorer' },
      { route: '/communities/Tallur', name: '04_tallur_community_dossier' },
      { route: '/districts', name: '05_district_intelligence' },
      { route: '/analytics', name: '06_analytics_and_models' },
      { route: '/assistant', name: '07_decision_assistant' },
    ];

    for (const p of pagesToTest) {
      const url = `${BASE_URL}${p.route}`;
      console.log(`Testing ${url}...`);

      const response = await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
      if (!response || response.status() >= 400) {
        console.error(`Page ${url} failed with status ${response?.status()}`);
        report.passed = false;
      }

      // Wait a moment for chart animations / map tiles to settle
      await page.waitForTimeout(1000);

      const shotPath = path.join(SCREENSHOT_DIR, vp.name, `${p.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      report.screenshots.push({ viewport: vp.name, page: p.name, path: shotPath });
      console.log(`  Screenshot saved: ${shotPath}`);

      if (!report.pagesTested.includes(p.route)) {
        report.pagesTested.push(p.route);
      }
    }

    // Interactive tests only on desktop viewport
    if (vp.name === 'desktop') {
      console.log('\n--- Executing Interactive Functional Tests ---');

      // 1. Community Table Search and Filter Test
      console.log('Testing Community Search & Autocomplete...');
      await page.goto(`${BASE_URL}/communities`, { waitUntil: 'networkidle' });
      await page.fill('input[placeholder*="Search by community name"]', 'Tallur');
      await page.click('button[type="submit"]');
      await page.waitForTimeout(1000);

      const tallurInTable = await page.locator('text=Tallur').first().isVisible();
      console.log(`  Search for 'Tallur': ${tallurInTable ? 'FOUND' : 'NOT FOUND'}`);

      // 2. Tallur Regression Verification on Dossier Page
      console.log('Testing Tallur Analytical Regression Baseline...');
      await page.goto(`${BASE_URL}/communities/Tallur`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(1000);

      const pageContent = await page.content();
      const hasScore7434 = pageContent.includes('74.34');
      const hasTierHigh = pageContent.includes('High');
      const hasBandAdequate = pageContent.includes('Band 4') || pageContent.includes('Adequate');
      const hasRank1833 = pageContent.includes('1833') || pageContent.includes('1,833');
      const hasCoast168 = pageContent.includes('1.68');

      console.log(`  Tallur AHP Score (74.34): ${hasScore7434 ? 'PASS' : 'FAIL'}`);
      console.log(`  Tallur AHP Tier (High): ${hasTierHigh ? 'PASS' : 'FAIL'}`);
      console.log(`  Tallur Priority Band (Band 4: Adequate): ${hasBandAdequate ? 'PASS' : 'FAIL'}`);
      console.log(`  Tallur Regional Rank (#1833): ${hasRank1833 ? 'PASS' : 'FAIL'}`);
      console.log(`  Tallur Coastal Proximity (1.68 km): ${hasCoast168 ? 'PASS' : 'FAIL'}`);

      report.analyticalChecks.tallur = {
        score7434: hasScore7434,
        tierHigh: hasTierHigh,
        bandAdequate: hasBandAdequate,
        rank1833: hasRank1833,
        coast168: hasCoast168,
      };

      if (!hasScore7434 || !hasTierHigh || !hasBandAdequate || !hasRank1833 || !hasCoast168) {
        report.passed = false;
      }

      // 3. Map Marker Selection Test
      console.log('Testing Map Marker Interactive Inspection...');
      await page.goto(`${BASE_URL}/map`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);

      // Click on circle marker on the map
      const markers = page.locator('.leaflet-interactive');
      const markerCount = await markers.count();
      console.log(`  Leaflet map rendered with ${markerCount} interactive markers`);

      if (markerCount > 0) {
        await markers.first().click({ force: true });
        await page.waitForTimeout(1000);
        const drawerOpen = await page.locator('text=AHP Score').count();
        console.log(`  Map side inspector opened on marker click: ${drawerOpen > 0 ? 'PASS' : 'FAIL'}`);
        report.analyticalChecks.mapMarkerInteraction = drawerOpen > 0;
        if (drawerOpen === 0) report.passed = false;
      }

      // 4. Assistant Conversational Turn with Tool Grounding
      console.log('Testing Conversational Assistant with Deterministic Tool Grounding...');
      await page.goto(`${BASE_URL}/assistant`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(1000);
      const promptButton = page.locator('button:has-text("Explain why Tallur has its current priority classification.")');

      if (await promptButton.isVisible()) {
        await promptButton.click();
        console.log('  Clicked suggested prompt: "Explain why Tallur has its current priority classification."');
        // Wait for assistant response to complete
        await page.waitForSelector('text=Authoritative Metric Verification', { timeout: 45000 });
        await page.waitForTimeout(1000);

        const assistantAnswer = await page.content();
        const hasAHPReconciliation = assistantAnswer.includes('Authoritative Metric Verification') && assistantAnswer.includes('74.34');
        const hasToolExecuted = assistantAnswer.includes('explain_preparedness_assessment') || assistantAnswer.includes('lookup_community') || assistantAnswer.includes('Deterministic Tools Executed') || assistantAnswer.includes('Grounded Deterministic Tools');

        console.log(`  Assistant Tool Grounding executed: ${hasToolExecuted ? 'PASS' : 'FAIL'}`);
        console.log(`  Assistant AHP Ground Truth reconciliation verified: ${hasAHPReconciliation ? 'PASS' : 'FAIL'}`);

        report.assistantChecks = {
          toolExecuted: hasToolExecuted,
          ahpReconciliation: hasAHPReconciliation,
        };

        if (!hasToolExecuted || !hasAHPReconciliation) {
          report.passed = false;
        }

        // Screenshot of interactive chat turn
        const chatShot = path.join(SCREENSHOT_DIR, 'desktop', '07_assistant_live_interaction.png');
        await page.screenshot({ path: chatShot, fullPage: true });
        report.screenshots.push({ viewport: 'desktop', page: '07_assistant_live_interaction', path: chatShot });
      }
    }

    await context.close();
  }

  await browser.close();

  // Save report to disk
  const reportPath = path.resolve(__dirname, '../reports/acceptance_test_report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\nAcceptance testing complete. Full report written to ${reportPath}`);
}

runAcceptanceTests().catch((err) => {
  console.error('Fatal acceptance test error:', err);
  process.exit(1);
});
