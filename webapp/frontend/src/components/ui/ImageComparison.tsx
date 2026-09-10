import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "../../lib/cn";

interface Props {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export default function ImageComparison({
  beforeSrc,
  afterSrc,
  beforeLabel = "Before",
  afterLabel = "After",
  className,
}: Props) {
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);
  const box = useRef<HTMLDivElement | null>(null);

  const setFromClientX = useCallback((clientX: number) => {
    const el = box.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(100, Math.max(0, next)));
  }, []);

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!dragging.current) return;
      setFromClientX(e.clientX);
    };
    const onUp = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [setFromClientX]);

  return (
    <div
      ref={box}
      className={cn(
        "relative overflow-hidden rounded-card bg-hailstone aspect-[16/10] select-none",
        className
      )}
      onPointerDown={(e) => {
        dragging.current = true;
        (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
        setFromClientX(e.clientX);
      }}
    >
      <img
        src={afterSrc}
        alt={afterLabel}
        className="absolute inset-0 h-full w-full object-contain"
        draggable={false}
      />
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        <img
          src={beforeSrc}
          alt={beforeLabel}
          className="h-full w-full object-contain"
          draggable={false}
        />
      </div>
      <div
        className="absolute inset-y-0 z-10 w-px bg-paper"
        style={{ left: `${pos}%` }}
      >
        <button
          type="button"
          aria-label="Drag comparison slider"
          className="absolute top-1/2 left-1/2 h-10 w-10 -translate-x-1/2 -translate-y-1/2 rounded-full border border-mist bg-paper text-horizon-navy"
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 4));
            if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 4));
          }}
        >
          ⇆
        </button>
      </div>
      <span className="absolute left-3 top-3 text-caption font-medium uppercase tracking-[0.1em] text-paper bg-horizon-navy/70 px-2 py-1 rounded-badge">
        {beforeLabel}
      </span>
      <span className="absolute right-3 top-3 text-caption font-medium uppercase tracking-[0.1em] text-paper bg-horizon-navy/70 px-2 py-1 rounded-badge">
        {afterLabel}
      </span>
    </div>
  );
}
