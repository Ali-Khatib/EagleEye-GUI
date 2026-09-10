import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Play } from "lucide-react";
import { api, fmt, fmtInt, fmtScore } from "../api";
import {
  DATASET_LABEL,
  type Dataset,
  type DatasetInfo,
  type Experiment,
  type ResultRow,
} from "../types";
import UploadCard from "./UploadCard";
import { EagleSpinner } from "./ui/EagleLoader";
import { PipelineCardSkeleton } from "./ui/Skeleton";
import Tooltip from "./ui/Tooltip";
import Button from "./ui/Button";
import { Eyebrow } from "./ui/TextReveal";
import { cn } from "../lib/cn";

interface Props {
  dataset: Dataset;
  onSeeResults: () => void;
}

const COPY: Record<
  string,
  { strength: string; limitation: string; inference: string }
> = {
  "1": {
    inference: "Full-frame, single pass",
    strength: "Highest FPS. Clean baseline for large objects.",
    limitation: "Distant and tiny instances are often missed.",
  },
  "2": {
    inference: "Sliced tiles + GREEDYNMM merge",
    strength: "Recovers small and crowded objects.",
    limitation: "More compute; overlapping boxes need NMS.",
  },
  "3": {
    inference: "Text-prompt concept segmentation",
    strength: "Open vocabulary; produces masks, not only boxes.",
    limitation: "Class-agnostic vs YOLO labels unless prompts align.",
  },
  "4": {
    inference: "YOLO boxes, SAM 3 mask refine",
    strength: "Keeps YOLO classes; cleaner instance shapes.",
    limitation: "Cannot invent objects YOLO never proposed.",
  },
  "5": {
    inference: "SAHI detect → merge → SAM 3 masks",
    strength: "Best small-object recall with class labels preserved.",
    limitation: "Slowest of the live-capable still-image stack.",
  },
  "6": {
    inference: "Same as #5, YOLOv8n backbone",
    strength: "Paper-comparable VisDrone baseline.",
    limitation: "VisDrone only; needs train_v8_visdrone weights.",
  },
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
  const [bust, setBust] = useState(() => Date.now());

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
    setBust(Date.now());
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
          const stderr = (detail as ResultRow & { stderr?: string }).stderr;
          setErrors((p) => ({
            ...p,
            [id]: stderr || "Run failed",
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

  const anyBusy = useMemo(() => Object.values(busy).some(Boolean), [busy]);

  return (
    <div className="flex flex-col gap-16">
      <header className="max-w-3xl">
        <Eyebrow>{DATASET_LABEL[dataset]}</Eyebrow>
        <h1 className="mt-4 text-heading-sm md:text-heading-lg font-medium tracking-[-0.038em]">
          Six ways to see the same scene.
        </h1>
        <p className="mt-4 text-body text-slate-whisper max-w-xl">
          From fast full-frame detection to multi-stage detection and
          segmentation. Give a picture, run a pipeline, compare the evidence.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Button
            className="group"
            arrow
            disabled={anyBusy || experiments.length === 0 || loading}
            onClick={runAll}
          >
            {anyBusy ? "Running…" : `Run all ${experiments.length || 6}`}
          </Button>
          <Button variant="secondary" onClick={onSeeResults}>
            See results
          </Button>
          <Button
            variant="ghost"
            disabled={resetting || anyBusy || loading}
            onClick={resetResults}
          >
            {resetting ? "Resetting…" : "Reset results"}
          </Button>
          <Button variant="ghost" onClick={() => setShowScores((v) => !v)}>
            {showScores ? "Hide scores" : "Show scores"}
          </Button>
        </div>
      </header>

      <UploadCard dataset={dataset} onUploaded={load} />

      {error && (
        <div className="rounded-[8px] border border-mist bg-hailstone text-bad text-body-sm px-4 py-3">
          {error}
        </div>
      )}

      {!yoloReady && (
        <Banner
          title={`${DATASET_LABEL[dataset]} YOLO checkpoint not found.`}
          body={`Expected at ${datasetInfo?.yolo_path}. Pipelines that need YOLO are disabled.`}
        />
      )}
      {!sam3Ready && (
        <Banner
          title="SAM 3 weights not found."
          body="Request sam3.pt from Hugging Face (facebook/sam3), then restart the backend."
        />
      )}

      <div className="flex flex-col gap-24">
        {loading && experiments.length === 0
          ? Array.from({ length: 3 }).map((_, i) => (
              <PipelineCardSkeleton key={`sk-${i}`} />
            ))
          : experiments.map((e, idx) => {
              const isBusy = !!busy[e.id];
              const lastErr = errors[e.id];
              const runSecs = elapsed[e.id] ?? e.elapsed;
              const isOpen = openId === e.id;
              const blocked = isExperimentBlocked(e);
              const copy = COPY[e.id];
              const featured = e.id === "5";
              const reverse = idx % 2 === 1;
              const imgSrc = e.has_result
                ? api.resultImageUrl(dataset, e.id, bust)
                : api.originalImageUrl(dataset, bust);

              return (
                <article
                  key={e.id}
                  className={cn(
                    "grid lg:grid-cols-2 gap-10 lg:gap-16 items-center rounded-card p-0",
                    featured && "lg:col-span-2 bg-horizon-navy text-paper px-6 py-12 md:px-12 md:py-16 -mx-5 md:mx-0"
                  )}
                >
                  <div className={cn(reverse && "lg:order-2")}>
                    <div
                      className={cn(
                        "overflow-hidden rounded-card bg-hailstone group border",
                        featured ? "border-white/10" : "border-mist hover:border-signal-blue",
                        "transition-colors duration-400"
                      )}
                    >
                      <img
                        src={imgSrc}
                        alt={e.name}
                        className="w-full aspect-[4/3] object-cover transition-transform duration-500 group-hover:scale-[1.03]"
                        onError={(ev) => {
                          ev.currentTarget.src = api.originalImageUrl(dataset, bust);
                        }}
                      />
                    </div>
                  </div>

                  <div className={cn(reverse && "lg:order-1")}>
                    <Eyebrow className={featured ? "text-paper/50" : undefined}>
                      Pipeline {e.id.padStart(2, "0")}
                      {featured ? " · Featured" : ""}
                    </Eyebrow>
                    <h2
                      className={cn(
                        "mt-3 text-heading-sm font-medium tracking-[-0.038em]",
                        featured ? "text-paper" : "text-horizon-navy"
                      )}
                    >
                      {e.name}
                    </h2>
                    <p
                      className={cn(
                        "mt-3 text-body",
                        featured ? "text-paper/70" : "text-slate-whisper"
                      )}
                    >
                      {e.subtitle}
                    </p>

                    <dl className="mt-8 grid grid-cols-1 gap-4 text-body-sm">
                      <Row label="Approach" value={e.components.join(" → ")} invert={featured} />
                      <Row label="Inference" value={copy?.inference ?? e.script} invert={featured} />
                      <Row label="Strength" value={copy?.strength ?? "—"} invert={featured} />
                      <Row label="Limitation" value={copy?.limitation ?? "—"} invert={featured} />
                      {e.metrics && (
                        <>
                          <Row
                            label="Runtime"
                            value={`${fmt(e.metrics.runtime_seconds, 2)} s · ${fmt(e.metrics.fps, 2)} FPS`}
                            invert={featured}
                          />
                          {showScores && (
                            <Row
                              label="P / R / F1"
                              value={`${fmtScore(e.metrics, e.metrics.precision, 3)} · ${fmtScore(e.metrics, e.metrics.recall, 3)} · ${fmtScore(e.metrics, e.metrics.f1_score, 3)}`}
                              invert={featured}
                            />
                          )}
                          <Row
                            label="Count"
                            value={fmtInt(e.metrics.count)}
                            invert={featured}
                          />
                        </>
                      )}
                    </dl>

                    <div className="mt-8 flex flex-wrap items-center gap-3">
                      <Tooltip
                        content={blocked ? blockedReason(e) : `Run ${e.name} on the current image.`}
                      >
                        <Button
                          className={cn("group", featured && "bg-signal-blue")}
                          arrow={!isBusy && !blocked}
                          disabled={isBusy || blocked}
                          onClick={() => runOne(e.id)}
                        >
                          {isBusy ? (
                            <span className="inline-flex items-center gap-2">
                              <EagleSpinner size="sm" />
                              Running
                              {runSecs != null && runSecs > 0
                                ? ` ${Math.round(runSecs)}s`
                                : ""}
                            </span>
                          ) : blocked ? (
                            needsYolov8(e) && e.weights_ready === false
                              ? dataset !== "visdrone"
                                ? "VisDrone only"
                                : "Needs YOLOv8"
                              : !yoloReady && needsYolo(e)
                              ? "Needs YOLO"
                              : "Needs SAM 3"
                          ) : (
                            <span className="inline-flex items-center gap-2">
                              <Play size={14} />
                              Run
                            </span>
                          )}
                        </Button>
                      </Tooltip>
                      <button
                        onClick={() => setOpenId(isOpen ? null : e.id)}
                        className={cn(
                          "text-body-sm font-medium transition-transform duration-300 hover:translate-x-0.5",
                          featured ? "text-paper" : "text-horizon-navy"
                        )}
                      >
                        {isOpen ? "Hide architecture" : "Architecture & parameters"}
                      </button>
                    </div>

                    {isOpen && (
                      <p
                        className={cn(
                          "mt-6 text-body-sm leading-relaxed",
                          featured ? "text-paper/70" : "text-graphite-dim"
                        )}
                      >
                        {e.description} Script: {e.script}. Metric kind:{" "}
                        {e.metric_kind.replace("_", "-")}.
                      </p>
                    )}
                    {lastErr && (
                      <p className="mt-4 text-caption text-bad">
                        {lastErr.slice(0, 280)}
                        {lastErr.length > 280 ? "…" : ""}
                      </p>
                    )}
                    {e.has_result && runSecs != null && runSecs > 0 && !isBusy && !lastErr && (
                      <p
                        className={cn(
                          "mt-4 text-caption",
                          featured ? "text-paper/60" : "text-slate-whisper"
                        )}
                      >
                        Done in {runSecs.toFixed(1)} s
                      </p>
                    )}
                  </div>
                </article>
              );
            })}
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  invert,
}: {
  label: string;
  value: string;
  invert?: boolean;
}) {
  return (
    <div className="grid grid-cols-[140px_1fr] gap-4 border-t border-current/10 pt-4">
      <dt className={cn("text-caption uppercase tracking-[0.1em]", invert ? "text-paper/40" : "text-slate-whisper")}>
        {label}
      </dt>
      <dd className={invert ? "text-paper" : "text-horizon-navy"}>{value}</dd>
    </div>
  );
}

function Banner({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-[8px] border border-mist bg-hailstone px-4 py-3 flex items-start gap-2 text-body-sm">
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warn" />
      <div>
        <div className="font-medium">{title}</div>
        <div className="text-slate-whisper text-caption mt-0.5">{body}</div>
      </div>
    </div>
  );
}
