import { useCallback, useEffect, useState } from "react";
import HomeTab from "./components/HomeTab";
import PipelinesTab from "./components/PipelinesTab";
import ResultsTab from "./components/ResultsTab";
import VideoTab from "./components/VideoTab";
import { EagleTransitionBar } from "./components/ui/EagleLoader";
import DatasetSelector from "./components/ui/DatasetSelector";
import StatusIndicator from "./components/ui/StatusIndicator";
import Tooltip from "./components/ui/Tooltip";
import { api } from "./api";
import { DATASETS, type Dataset, type DatasetInfo } from "./types";
import { cn } from "./lib/cn";

type Tab = "home" | "pipelines" | "video" | "results";

const STORAGE_KEY = "sahi-ui-dataset";

const TABS: { id: Tab; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "pipelines", label: "Pipelines" },
  { id: "results", label: "Results" },
  { id: "video", label: "Video" },
];

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
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const changeTab = useCallback(
    (next: Tab) => {
      if (next === tab) return;
      setMenuOpen(false);
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

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const gpuLabel =
    gpuInfo?.device === "cuda"
      ? (gpuInfo.name ?? "CUDA").replace("NVIDIA ", "").replace("GeForce ", "")
      : "CPU only";

  return (
    <div className="min-h-full flex flex-col bg-paper text-horizon-navy">
      <EagleTransitionBar active={tabTransition} />
      <header
        className={cn(
          "sticky top-0 z-[80] bg-paper/95 border-b transition-[height,border-color] duration-300",
          scrolled ? "border-mist" : "border-transparent"
        )}
      >
        <div
          className={cn(
            "mx-auto max-w-page flex items-center gap-6 px-5 md:px-8 transition-[padding] duration-300",
            scrolled ? "py-3" : "py-5"
          )}
        >
          <button
            type="button"
            onClick={() => changeTab("home")}
            className="flex items-center gap-3 text-left shrink-0"
          >
            <span
              className="h-8 w-8 rounded-[8px] aurora-gradient grid place-items-center text-paper text-[11px] font-semibold"
              aria-hidden
            >
              EE
            </span>
            <span className="leading-tight">
              <span className="block text-base md:text-lg font-medium tracking-tight">
                EagleEye AI
              </span>
              <span className="hidden sm:block text-[11px] text-slate-whisper">
                TÜBİTAK ARDEB 3501 · 124E099
              </span>
            </span>
          </button>

          <nav className="hidden md:flex items-center gap-8 ml-6">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => changeTab(t.id)}
                className={cn(
                  "relative text-lg md:text-xl font-medium tracking-tight transition-colors duration-300",
                  tab === t.id
                    ? "text-signal-blue"
                    : "text-horizon-navy hover:text-signal-blue"
                )}
              >
                {t.label}
                <span
                  className={cn(
                    "absolute -bottom-1 left-0 h-px bg-signal-blue transition-all duration-300",
                    tab === t.id ? "w-full" : "w-0"
                  )}
                />
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-5">
            <DatasetSelector
              value={dataset}
              onChange={changeDataset}
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
              <StatusIndicator
                label="GPU"
                value={gpuLabel}
                tone={gpuInfo?.device === "cuda" ? "good" : "warn"}
              />
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
              <StatusIndicator
                label="API"
                value={
                  healthOk === true
                    ? "Online"
                    : healthOk === false
                    ? "Offline"
                    : "…"
                }
                tone={
                  healthOk === true
                    ? "good"
                    : healthOk === false
                    ? "bad"
                    : "neutral"
                }
              />
            </Tooltip>
            <button
              className="md:hidden text-lg font-medium"
              onClick={() => setMenuOpen((v) => !v)}
              aria-expanded={menuOpen}
            >
              Menu
            </button>
          </div>
        </div>

        {menuOpen && (
          <nav className="md:hidden border-t border-mist px-5 py-3 flex flex-col gap-2">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => changeTab(t.id)}
                className={cn(
                  "text-left py-2 text-lg font-medium",
                  tab === t.id ? "text-signal-blue" : "text-horizon-navy"
                )}
              >
                {t.label}
              </button>
            ))}
          </nav>
        )}
      </header>

      <main className="flex-1">
        {healthOk === false && (
          <div className="mx-auto max-w-page px-5 md:px-8 pt-6">
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
            <div className="mx-auto max-w-page px-5 md:px-8 py-16 md:py-24">
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
            <div className="mx-auto max-w-page px-5 md:px-8 py-16 md:py-24">
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
