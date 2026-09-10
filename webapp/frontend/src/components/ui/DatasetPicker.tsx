import type { LucideIcon } from "lucide-react";
import { DATASETS, DATASET_LABEL, type Dataset, type DatasetInfo } from "../../types";
import ExpandChip from "./ExpandChip";

interface Props {
  value: Dataset;
  onChange: (d: Dataset) => void;
  infos: DatasetInfo[];
  icons: Record<Dataset, LucideIcon>;
}

export default function DatasetPicker({ value, onChange, infos, icons }: Props) {
  return (
    <div className="group/ds relative flex items-center">
      {DATASETS.map((d) => {
        const info = infos.find((i) => i.id === d);
        const ready = info ? info.yolo_ready : true;
        return (
          <ExpandChip
            key={d}
            icon={icons[d]}
            label={DATASET_LABEL[d]}
            active={value === d}
            disabled={!ready}
            expandGroup="ds"
            onClick={() => ready && onChange(d)}
            title={
              ready
                ? `Use ${DATASET_LABEL[d]}`
                : `${DATASET_LABEL[d]} still training`
            }
          />
        );
      })}
      <div
        className="pointer-events-none absolute left-1/2 top-[calc(100%+10px)] z-50 -translate-x-1/2 opacity-0 translate-y-1 transition-all duration-300 ease-out group-hover/ds:opacity-100 group-hover/ds:translate-y-0 group-focus-within/ds:opacity-100 group-focus-within/ds:translate-y-0"
      >
        <div className="whitespace-nowrap rounded-full bg-horizon-navy px-3.5 py-1.5 text-[12px] font-medium text-paper">
          Choose a dataset
        </div>
      </div>
    </div>
  );
}
