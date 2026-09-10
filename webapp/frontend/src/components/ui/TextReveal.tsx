import { useEffect, useRef, useState, type ReactNode } from "react";
import { useReducedMotion } from "../../hooks/useReducedMotion";
import { cn } from "../../lib/cn";

interface Props {
  children: string;
  className?: string;
  as?: "p" | "h2" | "h3" | "div";
}

export default function TextReveal({ children, className, as = "p" }: Props) {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLElement | null>(null);
  const [active, setActive] = useState(reduced);
  const words = children.trim().split(/\s+/);
  const Tag = as;

  useEffect(() => {
    if (reduced) {
      setActive(true);
      return;
    }
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setActive(true);
      },
      { threshold: 0.35, rootMargin: "0px 0px -10% 0px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [reduced]);

  return (
    <Tag
      ref={ref as never}
      className={cn("text-horizon-navy", className)}
    >
      {words.map((word, i) => (
        <span
          key={`${word}-${i}`}
          className="inline-block"
          style={{
            color: active || reduced ? "#001733" : "#d1d6e0",
            transition: reduced
              ? "none"
              : `color 700ms ease ${i * 40}ms`,
          }}
        >
          {word}
          {i < words.length - 1 ? "\u00A0" : null}
        </span>
      ))}
    </Tag>
  );
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "text-caption font-medium uppercase tracking-[0.12em] text-slate-whisper",
        className
      )}
    >
      {children}
    </div>
  );
}
