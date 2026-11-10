/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Manrope', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        // Dark Chocolate / Espresso Palette
        roast: '#120D0A',       // Deep roast background
        espresso: '#211713',    // Dark espresso surfaces
        chocolate: '#30221B',   // Chocolate panels
        mocha: '#493326',       // Mocha elevated surfaces
        walnut: '#76503A',      // Walnut accents / borders
        bronze: {
          DEFAULT: '#C18B58',   // Muted bronze
          light: '#D4A373',
          dark: '#A46F3E',
        },
        ivory: '#E9DCCB',       // Warm ivory primary text
        taupe: {
          DEFAULT: '#A98F7C',   // Muted taupe secondary text
          muted: '#826D5E',     // Tertiary metadata text
        },
        // Semantic Priority Colors (strictly preserved and high contrast)
        band: {
          critical: '#E05D52',   // Critical Red
          specialized: '#E5983B',// Specialized Amber
          moderate: '#48A976',   // Moderate Green
          adequate: '#4EA5D9',   // Adequate Blue
        },
        tier: {
          high: '#4EA5D9',
          medium: '#E5983B',
          low: '#E05D52',
        },
      },
    },
  },
  plugins: [],
}
