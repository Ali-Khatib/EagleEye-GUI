import { cn } from "../../lib/cn";

type SkeletonProps = {
  className?: string;
  /** Pulse shimmer (default) or static block */
  variant?: "shimmer" | "pulse";
};

export function Skeleton({ className, variant = "shimmer" }: SkeletonProps) {
  return (
    <div
      className={cn(
        "rounded-md bg-ink-600/40",
        variant === "shimmer" && "skeleton-shimmer",
        variant === "pulse" && "animate-pulse bg-ink-600/60",
        className
      )}
      aria-hidden
    />
  );
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn("h-3", i === lines - 1 ? "w-2/3" : "w-full")}
        />
      ))}
    </div>
  );
}

export function PipelineCardSkeleton() {
  return (
    <div className="rounded-2xl border border-ink-600/60 bg-ink-800/40 overflow-hidden flex flex-col animate-pulse-soft">
      <div className="p-5 flex flex-col gap-3">
        <div className="flex justify-between gap-3">
          <div className="flex-1 space-y-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-5 w-3/4" />
          </div>
          <Skeleton className="h-5 w-14 rounded-full" />
        </div>
        <SkeletonText lines={2} />
        <div className="flex gap-1.5 mt-1">
          <Skeleton className="h-5 w-12 rounded-md" />
          <Skeleton className="h-5 w-12 rounded-md" />
          <Skeleton className="h-5 w-20 rounded-md" />
        </div>
        <Skeleton className="h-3 w-full mt-1" />
        <div className="grid grid-cols-3 gap-1.5 mt-1">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-7 rounded-md" />
          ))}
        </div>
      </div>
      <Skeleton className="h-9 mx-5 mb-4 rounded-md" />
      <div className="border-t border-ink-600/60 p-4 flex justify-between gap-3 mt-auto bg-ink-900/30">
        <Skeleton className="h-3 w-28" />
        <Skeleton className="h-8 w-20 rounded-md" />
      </div>
    </div>
  );
}

export function ResultTileSkeleton() {
  return (
    <div className="rounded-2xl border border-ink-600/60 bg-ink-800/40 overflow-hidden">
      <div className="px-4 py-3 border-b border-ink-600/40 flex justify-between gap-2">
        <div className="space-y-2 flex-1">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-4 w-2/3" />
        </div>
        <Skeleton className="h-5 w-14 rounded-full shrink-0" />
      </div>
      <Skeleton className="aspect-video w-full rounded-none" />
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-px bg-ink-600/30">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-10 rounded-none" />
        ))}
      </div>
    </div>
  );
}
