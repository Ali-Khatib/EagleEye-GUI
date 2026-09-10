import { DATASETS, DATASET_LABEL, type Dataset, type DatasetInfo } from "../../types";
import { cn } from "../../lib/cn";
import Tooltip from "./Tooltip";

interface Props {
  value: Dataset;
  onChange: (d: Dataset) => void;
  infos: DatasetInfo[];
}

export default function DatasetSelector({ value, onChange, infos }: Props) {
  const activeIndex = DATASETS.indexOf(value);

  return (
    <div
      className="relative hidden sm:grid grid-cols-3 items-center p-1 rounded-[8px] border border-mist bg-paper"
      role="tablist"
      aria-label="Dataset"
    >
      <span
        className="dataset-indicator absolute top-1 bottom-1 left-1 rounded-[6px] bg-hailstone"
        style={{
          width: `calc((100% - 8px) / ${DATASETS.length})`,
          transform: `translateX(${activeIndex * 100}%)`,
        }}
        aria-hidden
      />
      {DATASETS.map((d) => {
        const info = infos.find((i) => i.id === d);
        const ready = info ? info.yolo_ready : true;
        const isActive = value === d;
        const title = ready
          ? `${DATASET_LABEL[d]} weights: ${info?.yolo_path ?? ""}`
          : `${DATASET_LABEL[d]} YOLO checkpoint not found yet (training in progress).`;
        return (
          <Tooltip key={d} content={title} side="bottom" when={!ready || !!info?.yolo_path}>
            <button
              role="tab"
              aria-selected={isActive}
              onClick={() => ready && onChange(d)}
              disabled={!ready}
              className={cn(
                "relative z-10 w-full px-3 py-2 text-body-sm md:text-base font-medium transition-colors duration-300",
                isActive
                  ? "text-signal-blue"
                  : ready
                  ? "text-horizon-navy hover:text-signal-blue"
                  : "text-fog cursor-not-allowed"
              )}
            >
              {DATASET_LABEL[d]}
              {!ready && (
                <span className="ml-1 text-[9px] uppercase tracking-widest text-warn">
                  training
                </span>
              )}
            </button>
          </Tooltip>
        );
      })}
    </div>
  );
}
