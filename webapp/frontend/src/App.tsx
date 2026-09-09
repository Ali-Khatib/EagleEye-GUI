import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Cpu,
  Film,
  Home,
  LayoutGrid,
} from "lucide-react";
import HomeTab from "./components/HomeTab";
import PipelinesTab from "./components/PipelinesTab";
import ResultsTab from "./components/ResultsTab";
import VideoTab from "./components/VideoTab";
import { EagleTransitionBar } from "./components/ui/EagleLoader";
import Tooltip from "./components/ui/Tooltip";
import { api } from "./api";
import {
  DATASETS,
  DATASET_LABEL,
  type Dataset,
  type DatasetInfo,
} from "./types";

type Tab = "home" | "pipelines" | "video" | "results";

const STORAGE_KEY = "sahi-ui-dataset";

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
  const [pendingTab, setPendingTab] = useState<Tab | null>(null);

  const changeTab = useCallback((next: Tab) => {
    if (next === tab) return;
    setPendingTab(next);
    setTabTransition(true);
    window.setTimeout(() => {
      setTab(next);
      setContentKey((k) => k + 1);
      setTabTransition(false);
      setPendingTab(null);
    }, 480);
  }, [tab]);

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

  // Auto-redirect away from a non-ready dataset (only after we have data)
  useEffect(() => {
    if (datasetInfos.length === 0) return;
    const current = datasetInfos.find((d) => d.id === dataset);
    if (current && !current.yolo_ready) {
      const ready = datasetInfos.find((d) => d.yolo_ready);
      if (ready) setDataset(ready.id);
    }
  }, [datasetInfos, dataset]);

  const tabs: { id: Tab; label: string; icon: JSX.Element }[] = [
    { id: "home", label: "Home", icon: <Home size={16} /> },
    { id: "pipelines", label: "Pipelines", icon: <LayoutGrid size={16} /> },
    { id: "video", label: "Video", icon: <Film size={16} /> },
    { id: "results", label: "Results", icon: <BarChart3 size={16} /> },
  ];

  return (
    <div className="min-h-full flex flex-col">
      <EagleTransitionBar
        active={tabTransition}
        label={
          pendingTab
            ? `Opening ${tabs.find((t) => t.id === pendingTab)?.label ?? "view"}…`
            : "Loading…"
        }
      />
      <header className="sticky top-0 z-20 backdrop-blur bg-ink-900/70 border-b border-ink-600/50">
        <div className="max-w-7xl mx-auto flex items-center gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-400 to-accent-600 grid place-items-center shadow-glow">
              <Cpu size={18} />
            </div>
            <div className="leading-tight">
              <div className="text-[15px] font-semibold tracking-tight">
                EagleEye AI
              </div>
              <div className="text-[11px] text-slate-400">
                TÜBİTAK ARDEB 3501 · Project 124E099
              </div>
            </div>
          </div>

          <nav className="ml-6 hidden md:flex items-center gap-1">
            {tabs.map((t) => (
              <Tooltip key={t.id} content={`Open ${t.label}`} side="bottom">
                <button
                  onClick={() => changeTab(t.id)}
                  className={`px-3 py-1.5 rounded-md text-sm flex items-center gap-2 transition ${
                    tab === t.id
                      ? "bg-accent-500/15 text-accent-400 border border-accent-500/30"
                      : "text-slate-300 hover:text-white hover:bg-ink-700/50 border border-transparent"
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              </Tooltip>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <DatasetToggle
              value={dataset}
              onChange={setDataset}
              infos={datasetInfos}
            />
            <Tooltip
              content={
                gpuInfo?.device === "cuda"
                  ? `Pipelines use GPU: ${gpuInfo.name ?? "CUDA"}`
                  : "No CUDA GPU — pipelines run on CPU (much slower)"
              }
              side="bottom"
            >
              <div
                className={`hidden sm:flex items-center gap-2 text-xs px-2.5 py-1 rounded-full border ${
                  gpuInfo?.device === "cuda"
                    ? "bg-violet-500/10 text-violet-300 border-violet-500/30"
                    : "bg-amber-500/10 text-amber-300 border-amber-500/30"
                }`}
              >
                <Cpu size={12} />
                {gpuInfo?.device === "cuda"
                  ? gpuInfo.name?.replace("NVIDIA ", "") ?? "GPU"
                  : "CPU only"}
              </div>
            </Tooltip>
            <Tooltip
              content={
                healthOk === true
                  ? "Backend API is reachable for the selected dataset."
                  : healthOk === false
                  ? "Start the backend with webapp/start.ps1"
                  : "Checking API connection…"
              }
              side="bottom"
            >
              <div
                className={`flex items-center gap-2 text-xs px-2.5 py-1 rounded-full border ${
                  healthOk === true
                    ? "bg-good/10 text-good border-good/30"
                    : healthOk === false
                    ? "bg-bad/10 text-bad border-bad/30"
                    : "bg-ink-700 text-slate-400 border-ink-500"
                }`}
              >
                <Activity size={12} className={healthOk === null ? "animate-pulse" : ""} />
                {healthOk === true
                  ? "API online"
                  : healthOk === false
                  ? "API offline"
                  : "API…"}
              </div>
            </Tooltip>
          </div>
        </div>

        <nav className="md:hidden max-w-7xl mx-auto px-4 pb-3 flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => changeTab(t.id)}
              className={`flex-1 px-3 py-1.5 rounded-md text-sm flex items-center justify-center gap-2 ${
                tab === t.id
                  ? "bg-accent-500/15 text-accent-400 border border-accent-500/30"
                  : "text-slate-300 bg-ink-700/40 border border-transparent"
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Backend offline banner – visible in any tab so the user notices */}
          {healthOk === false && (
            <div className="mb-6 rounded-lg border border-bad/40 bg-bad/10 text-bad text-sm px-4 py-3">
              <div className="font-semibold">Backend is offline.</div>
              <div className="text-bad/80 text-[12px] mt-0.5">
                Open a new terminal and run{" "}
                <span className="kbd">python webapp/backend/main.py</span>, or
                use <span className="kbd">.\webapp\start.ps1</span> to launch
                both.
              </div>
            </div>
          )}

          <div key={contentKey} className="content-enter">
            {tab === "home" && (
              <HomeTab onStart={() => changeTab("pipelines")} />
            )}
            {tab === "pipelines" && (
              <PipelinesTab
                dataset={dataset}
                onSeeResults={() => changeTab("results")}
              />
            )}
            {tab === "video" && <VideoTab dataset={dataset} />}
            {tab === "results" && <ResultsTab dataset={dataset} />}
          </div>
        </div>
      </main>

      <footer className="border-t border-ink-600/50 py-6 text-center text-xs text-slate-500">
        Built for the TÜBİTAK ARDEB 3501 research project · 124E099 · Real-time
        scalable AI camera design.
      </footer>
    </div>
  );
}

function DatasetToggle({
  value,
  onChange,
  infos,
}: {
  value: Dataset;
  onChange: (d: Dataset) => void;
  infos: DatasetInfo[];
}) {
  return (
    <div className="hidden sm:flex items-center gap-1 p-1 rounded-full border border-ink-600/60 bg-ink-800/60">
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
              key={d}
              onClick={() => ready && onChange(d)}
              disabled={!ready}
              className={`px-3 py-1 rounded-full text-[12px] transition flex items-center gap-1.5 ${
                isActive
                  ? "bg-accent-500 text-ink-900 font-semibold shadow-glow"
                  : ready
                  ? "text-slate-300 hover:text-white"
                  : "text-slate-500 cursor-not-allowed opacity-60"
              }`}
            >
              {DATASET_LABEL[d]}
              {!ready && (
                <span
                  className={`text-[9px] uppercase tracking-widest px-1.5 py-px rounded ${
                    isActive
                      ? "bg-ink-900/30 text-ink-900"
                      : "bg-warn/15 text-warn border border-warn/30"
                  }`}
                >
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
