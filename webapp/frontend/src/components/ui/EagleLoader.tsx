import { useEffect, useState } from "react";
import { cn } from "../../lib/cn";

/** Stylized eagle head — spins, then morphs into an indeterminate bar. */
export function EagleHead({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("eagle-head-svg", className)}
      aria-hidden
    >
      <path
        d="M8 28c4-10 14-16 24-14 2 6-1 14-8 18-6 4-13 3-16-4Z"
        fill="currentColor"
        opacity="0.25"
      />
      <path
        d="M10 26c5-8 15-13 22-11 1 5-2 11-7 14-5 3-11 2-15-3Z"
        fill="currentColor"
      />
      <path
        d="M30 12c6 2 10 7 10 13 0 1-6 0-8-2-3-3-4-7-2-11Z"
        fill="currentColor"
        opacity="0.85"
      />
      <circle cx="22" cy="20" r="2.2" fill="#0a0e1a" />
      <circle cx="22.6" cy="19.4" r="0.7" fill="#fff" opacity="0.9" />
      <path
        d="M34 18c4 1 6 4 5 7-3-1-6-3-7-6 0-1 1-1 2-1Z"
        fill="currentColor"
        opacity="0.9"
      />
    </svg>
  );
}

type Phase = "eagle" | "morph" | "bar";

interface EagleSpinnerProps {
  size?: "sm" | "md" | "lg";
  label?: string;
  className?: string;
}

const sizeMap = {
  sm: { box: "w-5 h-5", bar: "w-8 h-1", eagle: "w-5 h-5" },
  md: { box: "w-10 h-10", bar: "w-14 h-1.5", eagle: "w-8 h-8" },
  lg: { box: "w-14 h-14", bar: "w-20 h-2", eagle: "w-11 h-11" },
};

/** Inline loader: eagle spins → morphs → bar pulses (loops bar phase). */
export function EagleSpinner({ size = "sm", label, className }: EagleSpinnerProps) {
  const [phase, setPhase] = useState<Phase>("eagle");
  const s = sizeMap[size];

  useEffect(() => {
    const t1 = window.setTimeout(() => setPhase("morph"), 900);
    const t2 = window.setTimeout(() => setPhase("bar"), 1300);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, []);

  return (
    <span
      className={cn("inline-flex items-center gap-2 text-accent-300", className)}
      role="status"
      aria-label={label ?? "Loading"}
    >
      <span className={cn("relative grid place-items-center", s.box)}>
        <EagleHead
          className={cn(
            s.eagle,
            "text-accent-400 absolute transition-all duration-500",
            phase === "eagle" && "eagle-spin-active opacity-100 scale-100",
            phase === "morph" && "opacity-0 scale-50 rotate-180",
            phase === "bar" && "opacity-0 scale-0"
          )}
        />
        <span
          className={cn(
            "absolute transition-all duration-500 overflow-hidden rounded-full bg-ink-600/80",
            s.bar,
            phase === "bar" ? "opacity-100 scale-100" : "opacity-0 scale-x-0",
            phase === "bar" && "eagle-bar-indeterminate"
          )}
        />
      </span>
      {label && <span className="text-[12px] text-slate-300">{label}</span>}
    </span>
  );
}

interface EagleTransitionBarProps {
  active: boolean;
  label?: string;
}

/** Fixed top-of-page transition loader (tab / route changes). */
export function EagleTransitionBar({ active, label }: EagleTransitionBarProps) {
  const [visible, setVisible] = useState(false);
  const [phase, setPhase] = useState<Phase>("eagle");

  useEffect(() => {
    if (!active) {
      setVisible(false);
      setPhase("eagle");
      return;
    }
    setVisible(true);
    setPhase("eagle");
    const t1 = window.setTimeout(() => setPhase("morph"), 700);
    const t2 = window.setTimeout(() => setPhase("bar"), 1100);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [active]);

  if (!visible && !active) return null;

  return (
    <div
      className={cn(
        "fixed top-0 inset-x-0 z-[100] pointer-events-none transition-opacity duration-300",
        active ? "opacity-100" : "opacity-0"
      )}
      aria-live="polite"
      aria-busy={active}
    >
      <div className="h-[3px] bg-ink-700/90 overflow-hidden">
        <div
          className={cn(
            "h-full transition-all duration-500",
            phase === "bar" ? "eagle-top-bar w-full" : "w-0"
          )}
        />
      </div>
      <div className="flex justify-center -mt-4">
        <div
          className={cn(
            "rounded-full border border-accent-500/30 bg-ink-900/95 px-3 py-1.5 shadow-glow flex items-center gap-2 transition-all duration-500",
            phase === "bar" ? "scale-95 opacity-90" : "scale-100 opacity-100"
          )}
        >
          <span className="relative w-7 h-7 grid place-items-center">
            <EagleHead
              className={cn(
                "w-6 h-6 text-accent-400 absolute transition-all duration-500",
                phase === "eagle" && "eagle-spin-active opacity-100",
                phase === "morph" && "opacity-0 scale-50 rotate-90",
                phase === "bar" && "opacity-0 scale-0"
              )}
            />
            <span
              className={cn(
                "absolute w-5 h-1 rounded-full overflow-hidden bg-ink-600 transition-all duration-400",
                phase === "bar" ? "opacity-100 scale-100 eagle-bar-indeterminate" : "opacity-0 scale-x-0"
              )}
            />
          </span>
          {label && (
            <span className="text-[11px] text-slate-300 pr-0.5">{label}</span>
          )}
        </div>
      </div>
    </div>
  );
}
