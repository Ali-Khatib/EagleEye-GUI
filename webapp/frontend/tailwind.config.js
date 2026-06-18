/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0a0e1a",
          800: "#0f1424",
          700: "#141a2e",
          600: "#1c2440",
          500: "#2a3358",
        },
        accent: {
          400: "#3aa9ff",
          500: "#2491f0",
          600: "#0c75d3",
        },
        good: "#22c55e",
        warn: "#f59e0b",
        bad: "#ef4444",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "Segoe UI",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(58,169,255,0.25), 0 8px 32px -8px rgba(58,169,255,0.35)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.72" },
        },
      },
      animation: {
        shimmer: "shimmer 1.8s ease-in-out infinite",
        "pulse-soft": "pulse-soft 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
