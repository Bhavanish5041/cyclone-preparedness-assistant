import '@testing-library/jest-dom';

// Polyfill ResizeObserver for Recharts in jsdom
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

global.ResizeObserver = ResizeObserverMock as any;
window.ResizeObserver = ResizeObserverMock as any;

// Polyfill scrollIntoView for jsdom
window.HTMLElement.prototype.scrollIntoView = function () {};
