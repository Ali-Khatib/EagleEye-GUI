import { cn } from "../../lib/cn";

interface Props {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  hint?: string;
}

export default function StatusIndicator({
  label,
  value,
  tone = "neutral",
  hint,
}: Props) {
  const dot =
    tone === "good"
      ? "bg-good"
      : tone === "warn"
      ? "bg-warn"
      : tone === "bad"
      ? "bg-bad"
      : "bg-fog";

  return (
    <div className="hidden sm:flex flex-col leading-tight min-w-0" title={hint}>
      <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-slate-whisper">
        {label}
      </span>
      <span className="flex items-center gap-1.5 text-caption font-medium text-horizon-navy truncate">
        <span className={cn("h-1.5 w-1.5 rounded-full shrink-0", dot)} />
        {value}
      </span>
    </div>
  );
}
