import { useId, useState, type ReactNode } from "react";
import { cn } from "../../lib/cn";

type Side = "top" | "bottom" | "left" | "right";

const sideClass: Record<Side, string> = {
  top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
  bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
  left: "right-full top-1/2 -translate-y-1/2 mr-2",
  right: "left-full top-1/2 -translate-y-1/2 ml-2",
};

interface Props {
  content: ReactNode;
  children: ReactNode;
  side?: Side;
  className?: string;
  /** Show only when condition is true (e.g. disabled buttons) */
  when?: boolean;
}

export default function Tooltip({
  content,
  children,
  side = "top",
  className,
  when,
}: Props) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const show = when === undefined ? open : when && open;

  return (
    <span
      className={cn("relative inline-flex max-w-full", className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span aria-describedby={show ? id : undefined} className="inline-flex max-w-full">
        {children}
      </span>
      <span
        id={id}
        role="tooltip"
        className={cn(
          "pointer-events-none absolute z-50 w-max max-w-[240px] rounded-md border border-ink-500/80 bg-ink-800 px-2.5 py-1.5 text-[11px] leading-snug text-slate-200 shadow-lg transition-all duration-150",
          sideClass[side],
          show ? "opacity-100 translate-y-0 scale-100" : "opacity-0 scale-95 pointer-events-none"
        )}
      >
        {content}
      </span>
    </span>
  );
}
