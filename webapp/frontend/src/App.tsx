import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Car,
  Cpu,
  Film,
  Image,
  LayoutGrid,
  Plane,
} from "lucide-react";
import HomeTab from "./components/HomeTab";
import PipelinesTab from "./components/PipelinesTab";
import ResultsTab from "./components/ResultsTab";
import VideoTab from "./components/VideoTab";
import { EagleTransitionBar } from "./components/ui/EagleLoader";
import ExpandChip from "./components/ui/ExpandChip";
import { api } from "./api";
import {
  DATASETS,
  DATASET_LABEL,
  type Dataset,
  type DatasetInfo,
} from "./types";

type Tab = "home" | "pipelines" | "video" | "results";

const STORAGE_KEY = "sahi-ui-dataset";

const TABS: { id: Tab; label: string; icon: typeof LayoutGrid }[] = [
  { id: "pipelines", label: "Pipelines", icon: LayoutGrid },
  { id: "results", label: "Results", icon: BarChart3 },
  { id: "video", label: "Video", icon: Film },
];

const DATASET_ICON: Record<Dataset, typeof Plane> = {
  visdrone: Plane,
  kitti: Car,
  stock: Image,
};

export default function App() {
  const [tab, setTab] = useState<Tab>("home");
  const [dataset, setDataset] = useState<Dataset>(() => {
    const saved =
      typeof localStorage !== "undefined"
        ? (localStorage.getItem(STORAGE_KEY) as Dataset | null)
        : null;
    return saved && DATASETS.includes(saved) ? saved : "visdrone";
  });
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [gpuInfo, setGpuInfo] = useState<{
    device: string;
    name: string | null;
  } | null>(null);
  const [datasetInfos, setDatasetInfos] = useState<DatasetInfo[]>([]);
  const [tabTransition, setTabTransition] = useState(false);
  const [contentKey, setContentKey] = useState(0);

  const changeTab = useCallback(
    (next: Tab) => {
      if (next === tab) return;
      setTabTransition(true);
      window.setTimeout(() => {
        setTab(next);
        setContentKey((k) => k + 1);
        setTabTransition(false);
        window.scrollTo({ top: 0, behavior: "smooth" });
      }, 220);
    },
    [tab]
  );

  const changeDataset = useCallback((d: Dataset) => {
    setDataset(d);
  }, []);

  useEffect(() => {
    api
      .health(dataset)
      .then((h) => {
        setHealthOk(true);
        setGpuInfo({
          device: h.inference_device ?? "cpu",
          name: h.gpu_name ?? null,
        });
      })
      .catch(() => {
        setHealthOk(false);
        setGpuInfo(null);
      });
    if (typeof localStorage !== "undefined")
      localStorage.setItem(STORAGE_KEY, dataset);
  }, [dataset]);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      api
        .datasets()
        .then((data) => {
          if (!cancelled) setDatasetInfos(data);
        })
        .catch(() => {
          if (!cancelled) setDatasetInfos([]);
        });
    };
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    if (datasetInfos.length === 0) return;
    const current = datasetInfos.find((d) => d.id === dataset);
    if (current && !current.yolo_ready) {
      const ready = datasetInfos.find((d) => d.yolo_ready);
      if (ready) setDataset(ready.id);
    }
  }, [datasetInfos, dataset]);

  const gpuLabel =
    gpuInfo?.device === "cuda"
      ? (gpuInfo.name ?? "CUDA").replace("NVIDIA ", "").replace("GeForce ", "")
      : "CPU only";

  return (
    <div className="min-h-full flex flex-col bg-paper text-horizon-navy">
      <EagleTransitionBar active={tabTransition} />
      <header className="pointer-events-none fixed top-3 inset-x-0 z-[80] flex justify-center px-3">
        <div className="pointer-events-auto flex max-w-full items-center gap-1 rounded-full border border-mist bg-paper/95 px-2.5 py-2 backdrop-blur-md">
          <button
            type="button"
            onClick={() => changeTab("home")}
            className="grid h-11 w-11 shrink-0 place-items-center rounded-full aurora-gradient text-[11px] font-semibold text-paper"
            aria-label="Home"
            title="EagleEye AI home"
          >
            EE
          </button>
          <span className="mx-0.5 h-4 w-px shrink-0 bg-horizon-navy/15" aria-hidden />
          <nav className="flex items-center" aria-label="Primary">
            {TABS.map((t) => (
              <ExpandChip
                key={t.id}
                icon={t.icon}
                label={t.label}
                active={tab === t.id}
                onClick={() => changeTab(t.id)}
              />
            ))}
          </nav>
          <span className="mx-0.5 h-4 w-px shrink-0 bg-horizon-navy/15" aria-hidden />
          {DATASETS.map((d) => {
            const info = datasetInfos.find((i) => i.id === d);
            const ready = info ? info.yolo_ready : true;
            return (
              <ExpandChip
                key={d}
                icon={DATASET_ICON[d]}
                label={DATASET_LABEL[d]}
                active={dataset === d}
                disabled={!ready}
                onClick={() => ready && changeDataset(d)}
                title={
                  ready
                    ? DATASET_LABEL[d]
                    : `${DATASET_LABEL[d]} still training`
                }
              />
            );
          })}
        </div>
      </header>

      <div className="pointer-events-none fixed bottom-4 right-4 z-[80]">
        <div className="pointer-events-auto flex items-center gap-1 rounded-full border border-mist bg-paper/95 px-2 py-1.5 backdrop-blur-md">
          <ExpandChip
            icon={Cpu}
            label={gpuLabel}
            title={
              gpuInfo?.device === "cuda"
                ? `GPU ${gpuInfo.name ?? "CUDA"}`
                : "CPU only"
            }
          />
          <ExpandChip
            icon={Activity}
            label={
              healthOk === true
                ? "API online"
                : healthOk === false
                ? "API offline"
                : "API"
            }
            title={
              healthOk === true
                ? "Backend API is reachable"
                : healthOk === false
                ? "Start the backend with webapp/start.ps1"
                : "Checking API"
            }
          />
        </div>
      </div>

      <main className="flex-1">
        {healthOk === false && (
          <div className="mx-auto max-w-page px-5 md:px-8 pt-24">
            <div className="border border-mist bg-hailstone text-horizon-navy text-body-sm px-4 py-3 rounded-[8px]">
              <div className="font-medium">Backend is offline.</div>
              <div className="text-slate-whisper text-caption mt-0.5">
                Open a new terminal and run{" "}
                <span className="kbd">python webapp/backend/main.py</span>, or
                use <span className="kbd">.\webapp\start.ps1</span>.
              </div>
            </div>
          </div>
        )}

        <div key={`${contentKey}-${dataset}`} className="content-enter">
          {tab === "home" && (
            <HomeTab
              dataset={dataset}
              onStart={() => changeTab("pipelines")}
              onResults={() => changeTab("results")}
              onVideo={() => changeTab("video")}
            />
          )}
          {tab === "pipelines" && (
            <div className="mx-auto max-w-page px-5 md:px-8 pt-28 pb-16 md:pb-24">
              <PipelinesTab
                dataset={dataset}
                onSeeResults={() => changeTab("results")}
              />
            </div>
          )}
          {tab === "video" && (
            <VideoTab dataset={dataset} />
          )}
          {tab === "results" && (
            <div className="mx-auto max-w-page px-5 md:px-8 pt-28 pb-16 md:pb-24">
              <ResultsTab dataset={dataset} />
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-mist">
        <div className="mx-auto max-w-page px-5 md:px-8 py-12 flex flex-col md:flex-row gap-8 md:items-start justify-between">
          <div>
            <div className="text-body-sm font-medium">EagleEye AI</div>
            <p className="mt-2 text-caption text-slate-whisper max-w-sm">
              TÜBİTAK ARDEB 3501 · Project 124E099 · Real-time scalable AI
              camera design.
            </p>
          </div>
          <div className="flex gap-12 text-body-sm font-medium">
            <div className="flex flex-col gap-2">
              <button
                onClick={() => changeTab("home")}
                className="text-left hover:text-signal-blue transition-colors"
              >
                Home
              </button>
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => changeTab(t.id)}
                  className="text-left hover:text-signal-blue transition-colors"
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="flex flex-col gap-2 text-slate-whisper font-normal">
              <span>VisDrone</span>
              <span>KITTI</span>
              <span>Stock (COCO)</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
