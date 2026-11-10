/** Formatting helpers for numbers, percentages, and labels */

export function formatNumber(num: number | undefined | null): string {
  if (num === undefined || num === null) return 'N/A';
  return new Intl.NumberFormat('en-IN').format(num);
}

export function formatDecimal(num: number | undefined | null, digits: number = 2): string {
  if (num === undefined || num === null) return 'N/A';
  return num.toFixed(digits);
}

export function formatPercent(num: number | undefined | null, digits: number = 1): string {
  if (num === undefined || num === null) return 'N/A';
  return `${num.toFixed(digits)}%`;
}

export function formatBandName(band: string | undefined | null): string {
  if (!band) return 'Unassigned';
  return band.replace('BAND_', 'Band ').replace('_', ': ');
}
