/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        "horizon-navy": "#001733",
        "signal-blue": "#006aed",
        "cyan-dawn": "#18dcdc",
        "slate-whisper": "#68748d",
        "graphite-dim": "#464e5d",
        fog: "#d1d6e0",
        mist: "#e6e9f0",
        hailstone: "#f3f4f8",
        paper: "#ffffff",
        coal: "#000000",
        good: "#1a7a4c",
        warn: "#b45309",
        bad: "#b42318",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
      },
      fontSize: {
        caption: ["12px", { lineHeight: "1.3" }],
        "body-sm": ["14px", { lineHeight: "1.4" }],
        body: ["16px", { lineHeight: "1.5" }],
        subheading: ["20px", { lineHeight: "1.2", letterSpacing: "-0.6px" }],
        "subheading-lg": ["24px", { lineHeight: "1.2", letterSpacing: "-0.72px" }],
        "heading-sm": ["36px", { lineHeight: "1.1", letterSpacing: "-1.368px" }],
        heading: ["44px", { lineHeight: "1", letterSpacing: "-1.672px" }],
        "heading-lg": ["52px", { lineHeight: "0.97", letterSpacing: "-2.08px" }],
        display: ["64px", { lineHeight: "0.96", letterSpacing: "-2.56px" }],
        hero: ["90px", { lineHeight: "0.96", letterSpacing: "-3.6px" }],
      },
      maxWidth: {
        page: "1280px",
      },
      borderRadius: {
        badge: "4px",
        card: "8px",
      },
      spacing: {
        18: "72px",
        22: "88px",
        24: "96px",
        29: "116px",
      },
      boxShadow: {
        none: "none",
      },
      keyframes: {
        "ink-fill": {
          from: { color: "#d1d6e0" },
          to: { color: "#001733" },
        },
        "content-enter": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "content-enter": "content-enter 0.4s ease-out",
        shimmer: "shimmer 1.8s ease-in-out infinite",
      },
      transitionDuration: {
        400: "400ms",
      },
    },
  },
  plugins: [],
};
