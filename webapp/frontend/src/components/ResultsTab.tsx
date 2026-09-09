import { useEffect, useState } from "react";
import { ImageOff, BarChart3, RefreshCw, Trash2 } from "lucide-react";
import { api, fmt, fmtInt, fmtScore, fmtScoreInt, hasGtMetrics } from "../api";
import { DATASET_LABEL, type Dataset, type ResultRow } from "../types";
import {
  ImageLightbox,
  LightboxHint,
  lightboxImageClass,
  type LightboxImage,
} from "./ImageLightbox";
import { EagleSpinner } from "./ui/EagleLoader";
import { ResultTileSkeleton, Skeleton } from "./ui/Skeleton";
import Tooltip from "./ui/Tooltip";

interface Props {
  dataset: Dataset;
}

export default function ResultsTab({ dataset }: Props) {
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  // bumped every load() so <img> URLs change and the browser doesn't serve a
  // stale cached image after upload / random-pick / pipeline run.
  const [bust, setBust] = useState<number>(() => Date.now());
  const [lightbox, setLightbox] = useState<LightboxImage | null>(null);
  const [resetting, setResetting] = useState(false);
  const [showScores, setShowScores] = useState(false);

  const openLightbox = (src: string, title: string) => {
    setLightbox({ src, title });
  };

  const load = async (opts?: { background?: boolean }) => {
    const background = opts?.background && rows.length > 0;
    if (background) setRefreshing(true);
    else if (rows.length === 0) setLoading(true);
    try {
      const data = await api.results(dataset);
      setRows(data);
      setBust(Date.now());
        setSelected((prev) => {
        const stillValid = data.some(
          (r) => r.id === prev && (r.has_result || r.metrics)
        );
        if (stillValid) return prev;
        return (
          data.find((r) => r.has_result || r.metrics)?.id ?? data[0]?.id ?? null
        );
      });
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    setRows([]);
    load();
  }, [dataset]);

  // Auto-refresh while on the Results tab so newly finished runs show up.
  useEffect(() => {
    const ms = rows.some((r) => r.run_status === "running") ? 1500 : 4000;
    const id = setInterval(() => load({ background: true }), ms);
    return () => clearInterval(id);
  }, [dataset, rows.length, rows.some((r) => r.run_status === "running")]);

  const completed = rows.filter((r) => r.metrics);
  const selectedRow = rows.find((r) => r.id === selected) ?? null;

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
      setSelected(null);
      await load();
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            Results
            <span className="text-[11px] font-normal px-2 py-0.5 rounded-full border border-accent-500/30 bg-accent-500/10 text-accent-300">
              {DATASET_LABEL[dataset]}
            </span>
          </h2>
          <p className="text-sm text-slate-400">
            {completed.length}/{rows.length} pipelines have metrics saved.
            Click a pipeline for details.{" "}
            <LightboxHint /> on any image.
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {completed.length > 0 && (
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
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm border transition ${
                  showScores
                    ? "bg-accent-500/15 border-accent-500/40 text-accent-300"
                    : "bg-ink-700/60 border-ink-600/60 hover:bg-ink-700 text-slate-200"
                }`}
              >
                <BarChart3 size={14} />
                {showScores ? "Hide scores" : "Show scores"}
              </button>
            </Tooltip>
          )}
          {refreshing && (
            <span className="text-[11px] text-slate-400 inline-flex items-center gap-1.5">
              <EagleSpinner size="sm" />
              Updating…
            </span>
          )}
          <Tooltip content="Delete all result images and metrics for this dataset" side="bottom">
            <button
              onClick={resetResults}
              disabled={resetting || (loading && rows.length === 0)}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-ink-700/60 hover:bg-bad/20 hover:text-bad text-sm disabled:opacity-50 border border-ink-600/60"
            >
              {resetting ? <EagleSpinner size="sm" /> : <Trash2 size={14} />}
              Reset
            </button>
          </Tooltip>
          <Tooltip content="Reload pipeline results from the server" side="bottom">
            <button
              onClick={() => load({ background: rows.length > 0 })}
              disabled={loading && rows.length === 0}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-ink-700/60 hover:bg-ink-700 text-sm disabled:opacity-50"
            >
              <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
              Refresh
            </button>
          </Tooltip>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-bad/30 bg-bad/10 text-bad text-sm px-4 py-3">
          {error}
        </div>
      )}

      {/* Pipeline tiles */}
      <div
        className={`grid sm:grid-cols-2 lg:grid-cols-3 gap-4 transition-opacity duration-300 ${
          refreshing ? "opacity-80" : "opacity-100"
        }`}
      >
        {loading && rows.length === 0
          ? Array.from({ length: 6 }).map((_, i) => (
              <ResultTileSkeleton key={`sk-${i}`} />
            ))
          : rows.map((r) => {
          const isActive = r.id === selected;
          const m = r.metrics;
          return (
            <button
              key={r.id}
              onClick={() => setSelected(r.id)}
              className={`text-left rounded-2xl border bg-ink-800/40 transition overflow-hidden ${
                isActive
                  ? "border-accent-500/60 ring-1 ring-accent-500/40"
                  : "border-ink-600/60 hover:border-accent-500/40"
              }`}
            >
              <div className="px-4 py-3 border-b border-ink-600/40 flex items-center justify-between gap-2">
                <div>
                  <div className="text-[11px] uppercase tracking-widest text-slate-400">
                    Pipeline {r.id}
                  </div>
                  <div className="font-semibold leading-tight mt-0.5">
                    {r.name}
                  </div>
                </div>
                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full border ${
                    r.run_status === "running"
                      ? "bg-accent-500/10 text-accent-300 border-accent-500/30"
                      : r.has_result
                      ? "bg-good/10 text-good border-good/30"
                      : "bg-ink-700 text-slate-400 border-ink-500"
                  }`}
                >
                  {r.run_status === "running"
                    ? `running ${Math.round(r.elapsed ?? 0)}s`
                    : r.has_result
                    ? "ready"
                    : "no result"}
                </span>
              </div>

              {r.run_status === "running" ? (
                <div className="bg-ink-900/40 flex flex-col items-center gap-2 text-accent-300 text-sm py-10">
                  <EagleSpinner size="sm" />
                  Running… {Math.round(r.elapsed ?? 0)}s
                </div>
              ) : r.has_result ? (
                <div className="bg-ink-900/60 grid place-items-center aspect-video overflow-hidden relative">
                  <img
                    src={api.resultImageUrl(dataset, r.id, bust)}
                    alt={r.name}
                    className={`w-full h-full object-contain ${lightboxImageClass}`}
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      openLightbox(
                        api.resultImageUrl(dataset, r.id, bust),
                        `${r.name} · ${DATASET_LABEL[dataset]}`
                      );
                    }}
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.display =
                        "none";
                    }}
                  />
                </div>
              ) : (
                <div className="bg-ink-900/40 flex flex-col items-center gap-2 text-slate-500 text-sm py-10">
                  <ImageOff size={20} />
                  not run yet
                </div>
              )}

              {m && showScores && (
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-px bg-ink-600/30 text-[12px]">
                  <Stat label="P" value={fmtScore(m, m.precision, 3)} />
                  <Stat label="R" value={fmtScore(m, m.recall, 3)} />
                  <Stat label="F1" value={fmtScore(m, m.f1_score, 3)} />
                  <Stat label="Acc" value={fmtScore(m, m.accuracy, 3)} />
                  <Stat label="Count" value={fmtInt(m.count)} />
                  <Stat label="FPS" value={fmt(m.fps, 2)} />
                </div>
              )}
              {m && !showScores && (
                <div className="grid grid-cols-2 gap-px bg-ink-600/30 text-[12px]">
                  <Stat label="Count" value={fmtInt(m.count)} />
                  <Stat label="FPS" value={fmt(m.fps, 2)} />
                </div>
              )}
            </button>
          );
        })}
      </div>

      {loading && rows.length === 0 && (
        <div className="flex justify-center py-2">
          <EagleSpinner size="md" label="Loading results…" />
        </div>
      )}

      {/* Selected pipeline: before/after + full metric detail */}
      {selectedRow && (selectedRow.metrics || selectedRow.has_result) && (
        <div className="rounded-2xl border border-ink-600/60 bg-ink-800/40 overflow-hidden content-enter">
          <div className="px-5 py-4 border-b border-ink-600/40 flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-[11px] uppercase tracking-widest text-slate-400">
                Selected
              </div>
              <div className="text-lg font-semibold">{selectedRow.name}</div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              {selectedRow.components.map((c) => (
                <span
                  key={c}
                  className="px-2 py-0.5 rounded-md border border-ink-500 bg-ink-700/60 text-slate-300"
                >
                  {c}
                </span>
              ))}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-px bg-ink-600/40">
            <ImagePane
              title="Before (input)"
              src={api.originalImageUrl(dataset, bust)}
              available
              onMaximize={openLightbox}
            />
            <ImagePane
              title={`After · ${selectedRow.basename}.jpg`}
              src={api.resultImageUrl(dataset, selectedRow.id, bust)}
              available={selectedRow.has_result}
              onMaximize={openLightbox}
            />
          </div>

          <DetailPanel row={selectedRow} showScores={showScores} />
        </div>
      )}

      <ImageLightbox image={lightbox} onClose={() => setLightbox(null)} />
    </div>
  );
}

// -----------------------------------------------------------------------------

function Stat({ label, value }: { label: string; value: string }) {
  const tips: Record<string, string> = {
    P: "Precision — correct detections / all predictions",
    R: "Recall — correct detections / all ground-truth objects",
    F1: "Harmonic mean of precision and recall",
    Acc: "Accuracy at IoU threshold",
    Count: "Total matched evaluation count",
    FPS: "Frames per second (inference speed)",
  };
  return (
    <Tooltip content={tips[label] ?? label} side="top">
      <div className="bg-ink-900/40 px-3 py-2 flex flex-col cursor-default">
        <span className="text-[10px] uppercase tracking-widest text-slate-500">
          {label}
        </span>
        <span className="font-mono text-slate-200">{value}</span>
      </div>
    </Tooltip>
  );
}

function ImagePane({
  title,
  src,
  available,
  onMaximize,
}: {
  title: string;
  src: string;
  available: boolean;
  onMaximize: (src: string, title: string) => void;
}) {
  const [imgLoading, setImgLoading] = useState(true);

  useEffect(() => {
    setImgLoading(true);
  }, [src]);

  return (
    <div className="bg-ink-900/40">
      <div className="px-4 py-2 text-[11px] uppercase tracking-widest text-slate-400 border-b border-ink-600/40 flex items-center justify-between gap-2">
        <span>{title}</span>
        {available && <LightboxHint />}
      </div>
      <div className="aspect-[16/10] grid place-items-center bg-ink-900/60 relative">
        {available && imgLoading && (
          <Skeleton className="absolute inset-4 rounded-lg" />
        )}
        {available ? (
          <img
            src={src}
            alt={title}
            className={`w-full h-full object-contain ${lightboxImageClass} transition-opacity duration-300 ${
              imgLoading ? "opacity-0" : "opacity-100"
            }`}
            onLoad={() => setImgLoading(false)}
            onDoubleClick={() => onMaximize(src, title)}
            onError={(e) => {
              setImgLoading(false);
              e.currentTarget.style.display = "none";
            }}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-slate-500 text-sm py-10">
            <ImageOff size={24} />
            no image yet
          </div>
        )}
      </div>
    </div>
  );
}

function DetailPanel({
  row,
  showScores,
}: {
  row: ResultRow;
  showScores: boolean;
}) {
  const m = row.metrics;
  const showGt = hasGtMetrics(m);
  const runtimeItems: { label: string; value: string }[] = [
    { label: "Mode", value: m?.mode_name ?? row.name },
    { label: "Device", value: m?.inference_device ?? "—" },
    { label: "GPU", value: m?.gpu_name ?? "—" },
    { label: "Image", value: m?.dataset_image ?? "—" },
    { label: "Count", value: fmtInt(m?.count) },
    { label: "Predicted", value: fmtInt(m?.predicted_count) },
    { label: "Runtime (s)", value: fmt(m?.runtime_seconds, 3) },
    { label: "FPS", value: fmt(m?.fps, 2) },
    { label: "YOLO size MB", value: fmt(m?.yolo_model_size_mb, 2) },
    { label: "YOLO params", value: fmtInt(m?.yolo_parameter_count) },
    { label: "SAM size MB", value: fmt(m?.sam_checkpoint_size_mb, 2) },
  ];
  const scoreItems: { label: string; value: string }[] = [
    { label: "Ground truth", value: fmtScoreInt(m, m?.ground_truth_count) },
    { label: "IoU thresh", value: fmt(m?.iou_threshold, 2) },
    { label: "TP", value: fmtScoreInt(m, m?.true_positives) },
    { label: "FP", value: fmtScoreInt(m, m?.false_positives) },
    { label: "FN", value: fmtScoreInt(m, m?.false_negatives) },
    { label: "Precision", value: fmtScore(m, m?.precision, 3) },
    { label: "Recall", value: fmtScore(m, m?.recall, 3) },
    { label: "F1", value: fmtScore(m, m?.f1_score, 3) },
    { label: "Accuracy", value: fmtScore(m, m?.accuracy, 3) },
  ];
  const items = showScores ? [...runtimeItems, ...scoreItems] : runtimeItems;
  return (
    <div className="p-5 border-t border-ink-600/40">
      {!showGt && m && showScores && (
        <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-[13px] text-amber-100">
          No ground-truth label for this image — P/R/F1 cannot be computed.
          Upload a YOLO <code className="text-amber-200">.txt</code> label with
          your image, then re-run the pipeline. Or use{" "}
          <b>Pick random validation image</b> on the Upload tab.
        </div>
      )}
      {!showScores && m && (
        <p className="mb-4 text-[13px] text-slate-400">
          Press <b className="text-slate-300">Show scores</b> above to view
          P/R/F1, accuracy, and TP/FP/FN.
        </p>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {items.map((it) => (
          <div
            key={it.label}
            className="rounded-lg border border-ink-600/60 bg-ink-700/30 p-3"
          >
            <div className="text-[10px] uppercase tracking-widest text-slate-500">
              {it.label}
            </div>
            <div className="font-mono text-[13px] mt-0.5">{it.value}</div>
          </div>
        ))}
      </div>

      {m?.notes && (
        <div className="mt-4 text-[12px] text-slate-400 border-t border-ink-600/40 pt-3">
          <b>Notes:</b> {m.notes}
        </div>
      )}
    </div>
  );
}
