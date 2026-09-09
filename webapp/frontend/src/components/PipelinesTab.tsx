import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Gauge,
  Hash,
  Play,
  PlayCircle,
  Target,
  Trash2,
  XCircle,
} from "lucide-react";
import { AlertTriangle } from "lucide-react";
import { api, fmt, fmtInt, fmtScore } from "../api";
import {
  DATASET_LABEL,
  type Dataset,
  type DatasetInfo,
  type Experiment,
} from "../types";
import UploadCard from "./UploadCard";
import { EagleSpinner } from "./ui/EagleLoader";
import { PipelineCardSkeleton } from "./ui/Skeleton";
import Tooltip from "./ui/Tooltip";

interface Props {
  dataset: Dataset;
  onSeeResults: () => void;
}

const COMPONENT_COLORS: Record<string, string> = {
  YOLO: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  SAM3: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  SAHI: "bg-amber-500/15 text-amber-300 border-amber-500/30",
};

export default function PipelinesTab({ dataset, onSeeResults }: Props) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [elapsed, setElapsed] = useState<Record<string, number>>({});
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);
  const [showScores, setShowScores] = useState(false);

  const load = async () => {
    try {
      const data = await api.experiments(dataset);
      setExperiments(data);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
    try {
      const all = await api.datasets();
      setDatasetInfo(all.find((d) => d.id === dataset) ?? null);
    } catch {
      setDatasetInfo(null);
    }
  };

  useEffect(() => {
    setBusy({});
    setErrors({});
    setElapsed({});
    setLoading(true);
    load();
  }, [dataset]);

  const waitForRun = async (id: string) => {
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500));
      const data = await api.experiments(dataset);
      setExperiments(data);
      const exp = data.find((e) => e.id === id);
      if (!exp) break;
      if (exp.elapsed != null) {
        setElapsed((p) => ({ ...p, [id]: exp.elapsed! }));
      }
      if (exp.run_status === "running") continue;
      if (exp.run_status === "error") {
        try {
          const detail = await api.result(dataset, id);
          setErrors((p) => ({
            ...p,
            [id]: detail.stderr || "Run failed",
          }));
        } catch {
          setErrors((p) => ({ ...p, [id]: "Run failed" }));
        }
        return false;
      }
      if (exp.run_status === "done") {
        onSeeResults();
        return true;
      }
      break;
    }
    return false;
  };

  const yoloReady = datasetInfo ? datasetInfo.yolo_ready : true;
  const sam3Ready = datasetInfo?.sam_ready ?? true;

  const runOne = async (id: string) => {
    setBusy((p) => ({ ...p, [id]: true }));
    setErrors((p) => ({ ...p, [id]: "" }));
    setElapsed((p) => ({ ...p, [id]: 0 }));
    setExperiments((prev) =>
      prev.map((e) =>
        e.id === id ? { ...e, run_status: "running" as const, has_result: false } : e
      )
    );
    try {
      await api.run(dataset, id);
      await waitForRun(id);
      await load();
    } catch (e: any) {
      setErrors((p) => ({ ...p, [id]: e?.message ?? String(e) }));
      setExperiments((prev) =>
        prev.map((e) =>
          e.id === id ? { ...e, run_status: "error" as const } : e
        )
      );
    } finally {
      setBusy((p) => ({ ...p, [id]: false }));
    }
  };

  const needsYolo = (e: Experiment) => e.components.includes("YOLO");
  const needsYolov8 = (e: Experiment) =>
    !!e.needs_yolov8 || e.components.includes("YOLOv8");
  const needsSam3 = (e: Experiment) =>
    e.components.includes("SAM3") || e.key.includes("sam3");

  const isExperimentBlocked = (e: Experiment) => {
    if (needsYolov8(e)) return e.weights_ready === false;
    if (!yoloReady && needsYolo(e)) return true;
    if (!sam3Ready && needsSam3(e)) return true;
    return false;
  };

  const blockedReason = (e: Experiment) => {
    if (needsYolov8(e) && e.weights_ready === false) {
      if (dataset !== "visdrone") return "YOLOv8 pipeline is VisDrone-only.";
      return "YOLOv8 weights (train_v8_visdrone/best.pt) not found.";
    }
    if (!yoloReady && needsYolo(e)) {
      return `${DATASET_LABEL[dataset]} YOLO weights not available yet.`;
    }
    return "SAM 3 weights (sam3.pt) not found.";
  };

  const runAll = async () => {
    for (const exp of experiments) {
      if (isExperimentBlocked(exp)) continue;
      await runOne(exp.id);
    }
    onSeeResults();
  };

  const resetResults = async () => {
    if (
      !window.confirm(
        `Clear all pipeline results and images for ${DATASET_LABEL[dataset]}?`
      )
    ) {
      return;
    }
    setResetting(true);
    setError(null);
    try {
      await api.clearResults(dataset);
      await load();
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setResetting(false);
    }
  };

  const anyBusy = useMemo(
    () => Object.values(busy).some(Boolean),
    [busy]
  );

  return (
    <div className="flex flex-col gap-6">
      {/* Upload */}
      <UploadCard dataset={dataset} onUploaded={load} />

      {/* Header strip */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            Pipelines
            <span className="text-[11px] font-normal px-2 py-0.5 rounded-full border border-accent-500/30 bg-accent-500/10 text-accent-300">
              {DATASET_LABEL[dataset]}
            </span>
          </h2>
          <p className="text-sm text-slate-400">
            Give a picture, choose a pipeline, and the app runs everything
            automatically. Results open after each successful run.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Tooltip
            content={
              showScores
                ? "Hide P/R/F1 and accuracy scores"
                : "Show P/R/F1 and accuracy scores (requires ground-truth label)"
            }
            side="bottom"
          >
            <button
              onClick={() => setShowScores((v) => !v)}
              className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition ${
                showScores
                  ? "bg-accent-500/15 border-accent-500/40 text-accent-300"
                  : "border-ink-600/60 bg-ink-700/60 hover:bg-ink-700 text-slate-200"
              }`}
            >
              <BarChart3 size={16} />
              {showScores ? "Hide scores" : "Show scores"}
            </button>
          </Tooltip>
          <Tooltip content="Delete all result images and metrics for this dataset" side="bottom">
            <button
              disabled={resetting || anyBusy || loading}
              onClick={resetResults}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-ink-600/60 bg-ink-700/60 hover:bg-bad/20 hover:text-bad text-slate-200 disabled:opacity-50 transition"
            >
              {resetting ? <EagleSpinner size="sm" /> : <Trash2 size={16} />}
              Reset
            </button>
          </Tooltip>
          <Tooltip
            content="Run every available pipeline sequentially on the current image."
            side="bottom"
          >
            <button
              disabled={anyBusy || experiments.length === 0 || loading}
              onClick={runAll}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-accent-500/40 bg-accent-500/10 hover:bg-accent-500/20 text-accent-300 font-medium disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {anyBusy ? (
                <EagleSpinner size="sm" />
              ) : (
                <PlayCircle size={16} />
              )}
              Run all {experiments.length || 6}
            </button>
          </Tooltip>
          <Tooltip content="Jump to the Results tab" side="bottom">
            <button
              onClick={onSeeResults}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-ink-700/60 hover:bg-ink-700 text-slate-200 transition"
            >
              See Results <ArrowRight size={14} />
            </button>
          </Tooltip>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-bad/30 bg-bad/10 text-bad text-sm px-4 py-3">
          {error}
        </div>
      )}

      {!yoloReady && (
        <div className="rounded-lg border border-warn/40 bg-warn/10 text-warn text-sm px-4 py-3 flex items-start gap-2">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <div className="font-semibold">
              {DATASET_LABEL[dataset]} YOLO checkpoint not found.
            </div>
            <div className="text-warn/80 text-[12px] mt-0.5">
              Expected at <span className="kbd">{datasetInfo?.yolo_path}</span>.
              Training is probably still running. Pipelines that need YOLO are
              disabled. Switch the dataset toggle to a dataset that's ready, or
              wait for training to finish — this banner will disappear
              automatically.
            </div>
          </div>
        </div>
      )}

      {!sam3Ready && (
        <div className="rounded-lg border border-warn/40 bg-warn/10 text-warn text-sm px-4 py-3 flex items-start gap-2">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <div className="font-semibold">SAM 3 weights not found.</div>
            <div className="text-warn/80 text-[12px] mt-0.5">
              Required file:{" "}
              <span className="kbd break-all block mt-0.5">{datasetInfo?.sam_path}</span>
              Request access at huggingface.co/facebook/sam3, download sam3.pt,
              copy to that path, then restart webapp/start.ps1. Or: huggingface-cli
              login, then python scripts/download_sam3.py
            </div>
          </div>
        </div>
      )}

      {/* Cards grid */}
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
        {loading && experiments.length === 0
          ? Array.from({ length: 6 }).map((_, i) => (
              <PipelineCardSkeleton key={`sk-${i}`} />
            ))
          : experiments.map((e) => {
          const isBusy = !!busy[e.id];
          const lastErr = errors[e.id];
          const runSecs = elapsed[e.id] ?? e.elapsed;
          const isOpen = openId === e.id;
          const blocked = isExperimentBlocked(e);
          const pillStatus = isBusy
            ? "running"
            : e.has_result
            ? "done"
            : lastErr
            ? "error"
            : "idle";

          return (
            <div
              key={e.id}
              className={`rounded-2xl border border-ink-600/60 bg-ink-800/40 overflow-hidden flex flex-col transition-opacity ${
                isBusy ? "ring-1 ring-accent-500/30 animate-pulse-soft" : ""
              }`}
            >
              <div className="p-5 flex flex-col gap-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-widest text-slate-400">
                      Pipeline {e.id}
                    </div>
                    <div className="font-semibold text-lg leading-tight mt-0.5">
                      {e.name}
                    </div>
                  </div>
                  <StatusPill status={pillStatus} />
                </div>

                <p className="text-sm text-slate-400">{e.subtitle}</p>

                <div className="flex flex-wrap gap-1.5 mt-1">
                  {e.components.map((c) => (
                    <span
                      key={c}
                      className={`text-[11px] px-2 py-0.5 rounded-md border ${
                        COMPONENT_COLORS[c] ??
                        "bg-ink-700 border-ink-500 text-slate-300"
                      }`}
                    >
                      {c}
                    </span>
                  ))}
                  <span
                    className={`text-[11px] px-2 py-0.5 rounded-md border ${
                      e.metric_kind === "class_aware"
                        ? "bg-cyan-500/10 border-cyan-500/30 text-cyan-300"
                        : "bg-fuchsia-500/10 border-fuchsia-500/30 text-fuchsia-300"
                    }`}
                  >
                    {e.metric_kind === "class_aware"
                      ? "class-aware"
                      : "class-agnostic"}
                  </span>
                </div>

                <div className="text-[12px] text-slate-500 font-mono mt-1 truncate">
                  {e.script}
                </div>

                {e.metrics && !isBusy && (
                  <MetricChips m={e.metrics} showScores={showScores} />
                )}
                {isBusy && (
                  <div className="grid grid-cols-3 gap-1.5 mt-1">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <div
                        key={i}
                        className="h-7 rounded-md skeleton-shimmer bg-ink-600/30"
                      />
                    ))}
                  </div>
                )}
              </div>

              <button
                onClick={() => setOpenId(isOpen ? null : e.id)}
                className="px-5 py-2 text-xs text-slate-400 hover:text-white border-t border-ink-600/60 flex items-center justify-between"
              >
                <span>{isOpen ? "Hide details" : "Show description"}</span>
                <ChevronDown
                  size={14}
                  className={`transition-transform ${
                    isOpen ? "rotate-180" : ""
                  }`}
                />
              </button>
              {isOpen && (
                <div className="px-5 pb-4 text-sm text-slate-300 border-t border-ink-600/40 bg-ink-900/40">
                  {e.description}
                </div>
              )}

              <div className="border-t border-ink-600/60 p-4 flex items-center justify-between gap-3 mt-auto bg-ink-900/30">
                <div className="text-[11px] text-slate-400">
                  Auto-saves image, CSV, TXT
                </div>
                <Tooltip
                  content={
                    blocked
                      ? blockedReason(e)
                      : `Run ${e.name} on the current image.`
                  }
                  side="top"
                >
                  <button
                    onClick={() => runOne(e.id)}
                    disabled={isBusy || blocked}
                    className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-md bg-accent-500 text-ink-900 font-semibold text-sm hover:bg-accent-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
                  >
                    {isBusy ? (
                      <>
                        <EagleSpinner size="sm" />
                        Running…
                        {runSecs != null && runSecs > 0
                          ? ` ${Math.round(runSecs)}s`
                          : ""}
                      </>
                    ) : blocked ? (
                      <>
                        <AlertTriangle size={14} />
                        {needsYolov8(e) && e.weights_ready === false
                          ? dataset !== "visdrone"
                            ? "VisDrone only"
                            : "Needs YOLOv8"
                          : !yoloReady && needsYolo(e)
                          ? "Needs YOLO"
                          : "Needs SAM 3"}
                      </>
                    ) : (
                      <>
                        <Play size={14} />
                        Run
                      </>
                    )}
                  </button>
                </Tooltip>
              </div>

              {lastErr && (
                <div className="px-5 py-2 border-t border-bad/30 bg-bad/5 text-bad text-xs">
                  <XCircle size={12} className="inline -mt-0.5 mr-1" />
                  {lastErr.slice(0, 220)}
                  {lastErr.length > 220 ? "…" : ""}
                </div>
              )}
              {e.has_result && runSecs != null && runSecs > 0 && !isBusy && !lastErr && (
                <div className="px-5 py-2 border-t border-good/30 bg-good/5 text-good text-xs flex items-center gap-2">
                  <CheckCircle2 size={12} />
                  Done in {runSecs.toFixed(1)} s
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricChips({
  m,
  showScores,
}: {
  m: NonNullable<Experiment["metrics"]>;
  showScores: boolean;
}) {
  return (
    <div className="grid grid-cols-3 gap-1.5 mt-1">
      {showScores && (
        <>
          <Chip icon={<Target size={11} />} label="P" value={fmtScore(m, m.precision, 3)} tone="emerald" />
          <Chip icon={<Target size={11} />} label="R" value={fmtScore(m, m.recall, 3)} tone="emerald" />
          <Chip icon={<Target size={11} />} label="F1" value={fmtScore(m, m.f1_score, 3)} tone="emerald" />
          <Chip icon={<Target size={11} />} label="Acc" value={fmtScore(m, m.accuracy, 3)} tone="emerald" />
        </>
      )}
      <Chip icon={<Hash size={11} />} label="Count" value={fmtInt(m.count)} tone="cyan" />
      <Chip icon={<Gauge size={11} />} label="Time" value={fmt(m.runtime_seconds, 2)} tone="amber" />
      <Chip icon={<Gauge size={11} />} label="FPS" value={fmt(m.fps, 2)} tone="slate" />
    </div>
  );
}

function Chip({
  icon,
  label,
  value,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone: "cyan" | "amber" | "slate" | "emerald";
}) {
  const cls: Record<string, string> = {
    cyan: "bg-cyan-500/10 border-cyan-500/30 text-cyan-200",
    amber: "bg-amber-500/10 border-amber-500/30 text-amber-200",
    slate: "bg-ink-700/40 border-ink-500 text-slate-300",
    emerald: "bg-emerald-500/10 border-emerald-500/30 text-emerald-200",
  };
  return (
    <div
      className={`flex items-center gap-1.5 px-2 py-1 rounded-md border text-[11px] ${cls[tone]}`}
    >
      <span className="opacity-80">{icon}</span>
      <span className="uppercase tracking-wider opacity-70">{label}</span>
      <span className="ml-auto font-mono font-semibold">{value}</span>
    </div>
  );
}

function StatusPill({
  status,
}: {
  status: "idle" | "running" | "done" | "error";
}) {
  const map = {
    idle: { text: "no result", cls: "bg-ink-700 text-slate-300 border-ink-500" },
    running: {
      text: "running",
      cls: "bg-accent-500/10 text-accent-300 border-accent-500/30",
    },
    done: { text: "ready", cls: "bg-good/10 text-good border-good/30" },
    error: { text: "error", cls: "bg-bad/10 text-bad border-bad/30" },
  } as const;
  const m = map[status];
  return (
    <span
      className={`text-[11px] px-2 py-0.5 rounded-full border ${m.cls}`}
    >
      {m.text}
    </span>
  );
}
