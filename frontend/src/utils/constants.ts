/** Application constants, color mappings, and baseline metadata */

export const BAND_COLORS: Record<string, { bg: string; text: string; border: string; hex: string; label: string }> = {
  BAND_1_CRITICAL: {
    bg: 'bg-red-950/40',
    text: 'text-[#E05D52]',
    border: 'border-[#E05D52]/40',
    hex: '#E05D52',
    label: 'Band 1: Critical',
  },
  BAND_2_SPECIALIZED: {
    bg: 'bg-amber-950/40',
    text: 'text-[#E5983B]',
    border: 'border-[#E5983B]/40',
    hex: '#E5983B',
    label: 'Band 2: Specialized',
  },
  BAND_3_MODERATE: {
    bg: 'bg-emerald-950/40',
    text: 'text-[#48A976]',
    border: 'border-[#48A976]/40',
    hex: '#48A976',
    label: 'Band 3: Moderate',
  },
  BAND_4_ADEQUATE: {
    bg: 'bg-sky-950/40',
    text: 'text-[#4EA5D9]',
    border: 'border-[#4EA5D9]/40',
    hex: '#4EA5D9',
    label: 'Band 4: Adequate',
  },
};

export const TIER_COLORS: Record<string, { bg: string; text: string; border: string; hex: string }> = {
  High: {
    bg: 'bg-sky-950/40',
    text: 'text-[#4EA5D9]',
    border: 'border-[#4EA5D9]/40',
    hex: '#4EA5D9',
  },
  Medium: {
    bg: 'bg-amber-950/40',
    text: 'text-[#E5983B]',
    border: 'border-[#E5983B]/40',
    hex: '#E5983B',
  },
  Low: {
    bg: 'bg-red-950/40',
    text: 'text-[#E05D52]',
    border: 'border-[#E05D52]/40',
    hex: '#E05D52',
  },
};

export const CLUSTER_COLORS: Record<number, { hex: string; label: string }> = {
  0: { hex: '#4EA5D9', label: 'Urban & Peri-Urban Well-Resourced Corridor' },
  1: { hex: '#E5983B', label: 'Remote Foothill & Healthcare-Isolated' },
  2: { hex: '#9D6DE6', label: 'Socioeconomically Vulnerable Rural Interior' },
  3: { hex: '#E05D52', label: 'High-Exposure Coastal Rural Belt' },
};
