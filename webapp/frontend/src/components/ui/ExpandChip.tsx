import type { LucideIcon } from "lucide-react";
import { cn } from "../../lib/cn";

interface Props {
  icon: LucideIcon;
  label: string;
  active?: boolean;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
}

export default function ExpandChip({
  icon: Icon,
  label,
  active,
  onClick,
  disabled,
  title,
}: Props) {
  return (
    <button
      type="button"
      title={title ?? label}
      aria-label={label}
      aria-current={active ? "page" : undefined}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "group/chip inline-flex items-center rounded-full px-3 py-2.5 text-horizon-navy transition-colors duration-300",
        "hover:bg-hailstone focus-visible:bg-hailstone",
        active && "bg-hailstone text-signal-blue",
        disabled && "opacity-40 cursor-not-allowed"
      )}
    >
      <Icon
        size={22}
        className="shrink-0 transition-transform duration-300 group-hover/chip:scale-125 group-focus-visible/chip:scale-125"
        aria-hidden
      />
      <span
        className={cn(
          "max-w-0 overflow-hidden whitespace-nowrap text-sm font-medium opacity-0",
          "transition-all duration-300 ease-out",
          "group-hover/chip:ml-1.5 group-hover/chip:max-w-[11rem] group-hover/chip:opacity-100",
          "group-focus-visible/chip:ml-1.5 group-focus-visible/chip:max-w-[11rem] group-focus-visible/chip:opacity-100"
        )}
      >
        {label}
      </span>
    </button>
  );
}
