import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { cn } from "../../lib/cn";

interface Props {
  value: string | number | undefined;
  digits?: number;
  className?: string;
}

function parseNumeric(value: string | number | undefined): {
  n: number | null;
  raw: string;
} {
  if (value === undefined || value === null || value === "") {
    return { n: null, raw: "—" };
  }
  if (value === "N/A") return { n: null, raw: "N/A" };
  const n = typeof value === "number" ? value : Number(value);
  if (Number.isNaN(n)) return { n: null, raw: String(value) };
  return { n, raw: String(value) };
}

export default function CountUp({ value, digits = 2, className }: Props) {
  const reduced = useReducedMotion();
  const { n, raw } = parseNumeric(value);
  const [shown, setShown] = useState(reduced || n === null ? n : 0);
  const ref = useRef<HTMLSpanElement | null>(null);
  const started = useRef(false);

  useEffect(() => {
    started.current = false;
    setShown(reduced || n === null ? n : 0);
  }, [n, reduced]);

  useEffect(() => {
    if (n === null || reduced) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting || started.current) return;
        started.current = true;
        const start = performance.now();
        const duration = 700;
        const tick = (now: number) => {
          const t = Math.min(1, (now - start) / duration);
          const eased = 1 - Math.pow(1 - t, 3);
          setShown(n * eased);
          if (t < 1) requestAnimationFrame(tick);
          else setShown(n);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.4 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [n, reduced]);

  const display =
    n === null
      ? raw
      : Number.isInteger(n) && digits === 0
      ? Math.round(shown ?? 0).toLocaleString()
      : Number.isInteger(n)
      ? Math.round(shown ?? 0).toLocaleString()
      : (shown ?? 0).toFixed(digits);

  return (
    <span ref={ref} className={cn("tabular-nums", className)}>
      {display}
    </span>
  );
}
