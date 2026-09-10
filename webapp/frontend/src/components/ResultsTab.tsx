import { useEffect, useMemo, useState } from "react";
import { api, fmt, fmtInt, fmtScore, fmtScoreInt, hasGtMetrics } from "../api";
import { DATASET_LABEL, type Dataset, type ResultRow } from "../types";
import {
  ImageLightbox,
  LightboxHint,
  type LightboxImage,
} from "./ImageLightbox";
import { EagleSpinner } from "./ui/EagleLoader";
import Tooltip from "./ui/Tooltip";
import Button from "./ui/Button";
import ImageComparison from "./ui/ImageComparison";
import CountUp from "./ui/CountUp";
import { Eyebrow } from "./ui/TextReveal";
import { cn } from "../lib/cn";

interface Props {
  dataset: Dataset;
}

export default function ResultsTab({ dataset }: Props) {
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [bust, setBust] = useState<number>(() => Date.now());
  const [lightbox, setLightbox] = useState<LightboxImage | null>(null);
  const [resetting, setResetting] = useState(false);
  const [showScores, setShowScores] = useState(true);

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

  useEffect(() => {
    const ms = rows.some((r) => r.run_status === "running") ? 1500 : 4000;
    const id = setInterval(() => load({ background: true }), ms);
    return () => clearInterval(id);
  }, [dataset, rows.length, rows.some((r) => r.run_status === "running")]);

  const completed = rows.filter((r) => r.metrics);
  const selectedRow = rows.find((r) => r.id === selected) ?? null;

  const bestF1Id = useMemo(() => {
    let best = -1;
    let id: string | null = null;
    for (const r of completed) {
      if (!hasGtMetrics(r.metrics)) continue;
      const n = Number(r.metrics?.f1_score);
      if (!Number.isNaN(n) && n > best) {
        best = n;
        id = r.id;
      }
    }
    return id;
  }, [completed]);

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
    <div className="flex flex-col gap-16">
      <header className="max-w-3xl">
        <Eyebrow>Results · {DATASET_LABEL[dataset]}</Eyebrow>
        <h1 className="mt-4 text-heading-sm md:text-heading-lg font-medium tracking-[-0.038em]">
          How much does each stage actually improve detection?
        </h1>
        <p className="mt-4 text-body text-slate-whisper">
          {completed.length}/{rows.length} pipelines have metrics. Signal Blue
          marks the selected row and the strongest F1 when ground truth exists.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Button variant="secondary" onClick={() => setShowScores((v) => !v)}>
            {showScores ? "Hide scores" : "Show scores"}
          </Button>
          <Button
            variant="ghost"
            onClick={resetResults}
            disabled={resetting || (loading && rows.length === 0)}
          >
            {resetting ? "Resetting…" : "Reset"}
          </Button>
          <Button
            variant="ghost"
            onClick={() => load({ background: rows.length > 0 })}
          >
            Refresh
          </Button>
          {refreshing && <EagleSpinner size="sm" label="Updating" />}
        </div>
      </header>

      {error && (
        <div className="rounded-[8px] border border-mist bg-hailstone text-bad text-body-sm px-4 py-3">
          {error}
        </div>
      )}

      <div className="overflow-x-auto border-y border-mist">
        <table className="w-full min-w-[720px] text-left text-body-sm">
          <thead>
            <tr className="text-caption font-medium uppercase tracking-[0.08em] text-slate-whisper">
              <th className="py-4 pr-4">Pipeline</th>
              {showScores && (
                <>
                  <th className="py-4 pr-4">Precision</th>
                  <th className="py-4 pr-4">Recall</th>
                  <th className="py-4 pr-4">F1</th>
                </>
              )}
              <th className="py-4 pr-4">Count</th>
              <th className="py-4 pr-4">FPS</th>
              <th className="py-4">Status</th>
            </tr>
          </thead>
          <tbody>
            {loading && rows.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-10 text-slate-whisper">
                  Loading results…
                </td>
              </tr>
            ) : (
              rows.map((r) => {
                const m = r.metrics;
                const isSel = r.id === selected;
                const isBest = r.id === bestF1Id;
                return (
                  <tr
                    key={r.id}
                    onClick={() => setSelected(r.id)}
                    className={cn(
                      "border-t border-mist cursor-pointer transition-colors duration-300",
                      isSel ? "bg-hailstone" : "hover:bg-hailstone/60"
                    )}
                  >
                    <td className="py-4 pr-4">
                      <div
                        className={cn(
                          "font-medium",
                          isSel || isBest ? "text-signal-blue" : "text-horizon-navy"
                        )}
                      >
                        {r.name}
                      </div>
                      <div className="text-caption text-slate-whisper">
                        Pipeline {r.id}
                      </div>
                    </td>
                    {showScores && (
                      <>
                        <td className="py-4 pr-4 tabular-nums">
                          {m ? (
                            <CountUp value={hasGtMetrics(m) ? m.precision : "N/A"} digits={3} />
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="py-4 pr-4 tabular-nums">
                          {m ? (
                            <CountUp value={hasGtMetrics(m) ? m.recall : "N/A"} digits={3} />
                          ) : (
                            "—"
                          )}
                        </td>
                        <td
                          className={cn(
                            "py-4 pr-4 tabular-nums",
                            isBest && "text-signal-blue font-medium"
                          )}
                        >
                          {m ? (
                            <CountUp value={hasGtMetrics(m) ? m.f1_score : "N/A"} digits={3} />
                          ) : (
                            "—"
                          )}
                        </td>
                      </>
                    )}
                    <td className="py-4 pr-4 tabular-nums">
                      {m ? <CountUp value={m.count} digits={0} /> : "—"}
                    </td>
                    <td className="py-4 pr-4 tabular-nums">
                      {m ? <CountUp value={m.fps} digits={2} /> : "—"}
                    </td>
                    <td className="py-4 text-caption uppercase tracking-[0.08em] text-slate-whisper">
                      {r.run_status === "running"
                        ? `Running ${Math.round(r.elapsed ?? 0)}s`
                        : r.has_result
                        ? "Ready"
                        : "Idle"}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {selectedRow && (selectedRow.metrics || selectedRow.has_result) && (
        <section className="flex flex-col gap-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <Eyebrow>Visual comparison</Eyebrow>
              <h2 className="mt-2 text-heading-sm font-medium tracking-[-0.038em]">
                {selectedRow.name}
              </h2>
              <p className="mt-2 text-caption text-slate-whisper">
                Drag the slider. <LightboxHint />
              </p>
            </div>
            <Button
              variant="ghost"
              onClick={() =>
                selectedRow.has_result &&
                setLightbox({
                  src: api.resultImageUrl(dataset, selectedRow.id, bust),
                  title: `${selectedRow.name} · ${DATASET_LABEL[dataset]}`,
                })
              }
            >
              Fullscreen
            </Button>
          </div>

          {selectedRow.has_result ? (
            <ImageComparison
              beforeSrc={api.originalImageUrl(dataset, bust)}
              afterSrc={api.resultImageUrl(dataset, selectedRow.id, bust)}
              beforeLabel="Original"
              afterLabel="Pipeline"
            />
          ) : (
            <div className="rounded-card bg-hailstone aspect-[16/10] grid place-items-center text-slate-whisper text-body-sm">
              Run this pipeline to compare frames.
            </div>
          )}

          <DetailPanel row={selectedRow} showScores={showScores} />
        </section>
      )}

      <hr className="border-mist" />

      <section className="grid lg:grid-cols-2 gap-12 py-8">
        <h2 className="text-heading-sm font-medium tracking-[-0.038em] leading-[1.2]">
          Accuracy is only half of the problem.
        </h2>
        <p className="text-body leading-relaxed text-graphite-dim">
          {`Higher recall can mean more false positives. SAHI can recover small objects but increase computation. SAM 3 can introduce additional detections through text prompting. The hybrid system therefore has to balance accuracy, recall, precision, and runtime on the same ${DATASET_LABEL[dataset]} image.`}
        </p>
      </section>

      <ImageLightbox image={lightbox} onClose={() => setLightbox(null)} />
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
  const primary = m
    ? [
        { label: "FPS", value: fmt(m.fps, 2) },
        { label: "Runtime", value: `${fmt(m.runtime_seconds, 3)} s` },
        { label: "Predicted", value: fmtInt(m.predicted_count) },
      ]
    : [];
  const rest: { label: string; value: string }[] = [
    { label: "Device", value: m?.inference_device ?? "—" },
    { label: "GPU", value: m?.gpu_name ?? "—" },
    { label: "YOLO size", value: `${fmt(m?.yolo_model_size_mb, 2)} MB` },
    { label: "YOLO params", value: fmtInt(m?.yolo_parameter_count) },
    { label: "SAM size", value: `${fmt(m?.sam_checkpoint_size_mb, 2)} MB` },
    { label: "IoU", value: fmt(m?.iou_threshold, 2) },
  ];
  if (showScores) {
    rest.push(
      { label: "GT", value: fmtScoreInt(m, m?.ground_truth_count) },
      { label: "TP", value: fmtScoreInt(m, m?.true_positives) },
      { label: "FP", value: fmtScoreInt(m, m?.false_positives) },
      { label: "FN", value: fmtScoreInt(m, m?.false_negatives) },
      { label: "Precision", value: fmtScore(m, m?.precision, 3) },
      { label: "Recall", value: fmtScore(m, m?.recall, 3) },
      { label: "F1", value: fmtScore(m, m?.f1_score, 3) },
      { label: "Accuracy", value: fmtScore(m, m?.accuracy, 3) }
    );
  }

  return (
    <div>
      {!showGt && m && showScores && (
        <p className="mb-6 text-body-sm text-slate-whisper">
          No ground-truth label for this image — P/R/F1 cannot be computed.
          Upload a YOLO .txt label, or pick a random validation image.
        </p>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 mb-10">
        {primary.map((it) => (
          <div key={it.label} className="border-t border-mist pt-4">
            <div className="text-caption uppercase tracking-[0.1em] text-slate-whisper">
              {it.label}
            </div>
            <div className="mt-2 text-heading-sm font-medium tracking-tight tabular-nums">
              {it.value}
            </div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-6">
        {rest.map((it) => (
          <Tooltip key={it.label} content={it.label}>
            <div>
              <div className="text-[10px] uppercase tracking-[0.1em] text-slate-whisper">
                {it.label}
              </div>
              <div className="mt-1 text-body-sm text-horizon-navy tabular-nums">
                {it.value}
              </div>
            </div>
          </Tooltip>
        ))}
      </div>
      {m?.notes && (
        <p className="mt-8 text-caption text-slate-whisper">{m.notes}</p>
      )}
    </div>
  );
}
