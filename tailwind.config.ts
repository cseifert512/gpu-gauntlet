import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'challenge-primary': '#00ff88',
        'challenge-bg': '#0a0a0f',
        'challenge-card': '#12121a',
      },
    },
  },
  plugins: [],
};

export default config;

