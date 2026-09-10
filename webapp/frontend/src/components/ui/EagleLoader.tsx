import { useEffect, useState } from "react";
import { cn } from "../../lib/cn";

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
      <circle cx="22" cy="20" r="2.2" fill="#ffffff" />
      <circle cx="22.6" cy="19.4" r="0.7" fill="#001733" opacity="0.9" />
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
      className={cn("inline-flex items-center gap-2 text-signal-blue", className)}
      role="status"
      aria-label={label ?? "Loading"}
    >
      <span className={cn("relative grid place-items-center", s.box)}>
        <EagleHead
          className={cn(
            s.eagle,
            "text-signal-blue absolute transition-all duration-500",
            phase === "eagle" && "eagle-spin-active opacity-100 scale-100",
            phase === "morph" && "opacity-0 scale-50 rotate-180",
            phase === "bar" && "opacity-0 scale-0"
          )}
        />
        <span
          className={cn(
            "absolute transition-all duration-500 overflow-hidden rounded-full bg-mist",
            s.bar,
            phase === "bar" ? "opacity-100 scale-100" : "opacity-0 scale-x-0",
            phase === "bar" && "eagle-bar-indeterminate"
          )}
        />
      </span>
      {label && <span className="text-caption text-slate-whisper">{label}</span>}
    </span>
  );
}

interface EagleTransitionBarProps {
  active: boolean;
  label?: string;
}

export function EagleTransitionBar({ active }: EagleTransitionBarProps) {
  if (!active) return null;

  return (
    <div
      className="fixed top-0 inset-x-0 z-[100] pointer-events-none"
      aria-live="polite"
      aria-busy={active}
    >
      <div className="h-[2px] bg-mist overflow-hidden">
        <div className="h-full eagle-top-bar w-full" />
      </div>
    </div>
  );
}
