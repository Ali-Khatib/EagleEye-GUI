import { ArrowRight } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "../../lib/cn";

type Variant = "primary" | "ghost" | "secondary";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  arrow?: boolean;
  children: ReactNode;
}

export default function Button({
  variant = "primary",
  arrow = false,
  className,
  children,
  ...rest
}: Props) {
  const styles: Record<Variant, string> = {
    primary:
      "bg-signal-blue text-paper hover:translate-y-[-1px] disabled:opacity-50 disabled:translate-y-0",
    ghost:
      "bg-transparent text-horizon-navy hover:underline underline-offset-4",
    secondary:
      "bg-hailstone text-horizon-navy border border-mist hover:border-horizon-navy/20",
  };

  return (
    <button
      className={cn(
        "group inline-flex items-center justify-center gap-2 rounded-[8px] px-6 py-4 text-base md:text-lg font-medium transition-all duration-300 disabled:cursor-not-allowed",
        styles[variant],
        className
      )}
      {...rest}
    >
      {children}
      {arrow && (
        <ArrowRight
          size={18}
          className="transition-transform duration-300 group-hover:translate-x-1"
          aria-hidden
        />
      )}
    </button>
  );
}

export function CircularArrowButton({
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cn(
        "grid h-12 w-12 place-items-center rounded-full bg-signal-blue text-paper transition-transform duration-300 hover:scale-[1.04]",
        className
      )}
      {...rest}
    >
      <ArrowRight size={18} aria-hidden />
    </button>
  );
}
