import { cn } from "../../lib/cn";

type SkeletonProps = {
  className?: string;
  variant?: "shimmer" | "pulse";
};

export function Skeleton({ className, variant = "shimmer" }: SkeletonProps) {
  return (
    <div
      className={cn(
        "rounded-[8px] bg-hailstone",
        variant === "shimmer" && "skeleton-shimmer",
        variant === "pulse" && "animate-pulse",
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
    <div className="overflow-hidden rounded-card border border-mist bg-paper">
      <Skeleton className="aspect-[16/10] w-full rounded-none" />
      <div className="p-6 space-y-3">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-8 w-2/3" />
        <SkeletonText lines={3} />
      </div>
    </div>
  );
}

export function ResultTileSkeleton() {
  return (
    <div className="overflow-hidden rounded-card border border-mist">
      <div className="px-4 py-3 border-b border-mist flex justify-between gap-2">
        <div className="space-y-2 flex-1">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      </div>
      <Skeleton className="aspect-video w-full rounded-none" />
    </div>
  );
}
