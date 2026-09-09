import { useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  ImagePlus,
  Shuffle,
  Upload,
  XCircle,
} from "lucide-react";
import { api } from "../api";
import { DATASET_LABEL, type Dataset } from "../types";
import { EagleSpinner } from "./ui/EagleLoader";
import { Skeleton } from "./ui/Skeleton";
import Tooltip from "./ui/Tooltip";

interface Props {
  dataset: Dataset;
  onUploaded?: () => void;
}

export default function UploadCard({ dataset, onUploaded }: Props) {
  const [image, setImage] = useState<File | null>(null);
  const [label, setLabel] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bust, setBust] = useState<number>(Date.now());
  const [dragOver, setDragOver] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);

  const imgInputRef = useRef<HTMLInputElement>(null);
  const labelInputRef = useRef<HTMLInputElement>(null);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    setBust(Date.now());
    setImageLoading(true);
  }, [dataset]);

  const pickImage = (file: File | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please select an image file (jpg / png / bmp / webp).");
      return;
    }
    setError(null);
    setImage(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleDrop = (ev: React.DragEvent) => {
    ev.preventDefault();
    setDragOver(false);
    const files = Array.from(ev.dataTransfer?.files ?? []);
    const img = files.find((f) => f.type.startsWith("image/"));
    if (img) pickImage(img);
  };

  const submit = async () => {
    if (!image) {
      setError("Pick an image first.");
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    setImageLoading(true);
    try {
      const res = await api.upload(dataset, image, label ?? undefined);
      const gtNote = res.label_saved
        ? " Ground-truth label saved — P/R/F1 will be computed."
        : " No label uploaded — visual results only (P/R/F1 need a YOLO .txt label).";
      setSuccess(
        `Image ready (${(res.image_size_bytes / 1024).toFixed(1)} KB).${gtNote}`
      );
      setBust(Date.now());
      setImage(null);
      setLabel(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
      if (imgInputRef.current) imgInputRef.current.value = "";
      if (labelInputRef.current) labelInputRef.current.value = "";
      onUploaded?.();
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const pickFromDataset = async () => {
    setBusy(true);
    setError(null);
    setSuccess(null);
    setImageLoading(true);
    try {
      const r = await api.randomSample(dataset);
      const fname = r.source_image.split(/[\\/]/).pop() ?? r.source_image;
      const labelPart = r.label_saved
        ? ` · loaded ${r.label_count} GT box${r.label_count === 1 ? "" : "es"}`
        : " · no matching GT label";
      const clearedPart = r.results_cleared
        ? ` · cleared ${r.results_cleared} old result file${
            r.results_cleared === 1 ? "" : "s"
          }`
        : "";
      setSuccess(
        `Picked ${fname} from ${DATASET_LABEL[dataset]} val${labelPart}${clearedPart}`
      );
      setBust(Date.now());
      setImage(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
      if (imgInputRef.current) imgInputRef.current.value = "";
      onUploaded?.();
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-ink-600/60 bg-ink-800/40 overflow-hidden">
      <div className="px-5 py-4 border-b border-ink-600/40 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-accent-500/15 grid place-items-center">
          <ImagePlus size={16} className="text-accent-300" />
        </div>
        <div>
          <div className="font-semibold flex items-center gap-2">
            Give a picture, run a pipeline
            <span className="text-[10px] uppercase tracking-widest px-1.5 py-0.5 rounded bg-accent-500/15 text-accent-300 border border-accent-500/30">
              {DATASET_LABEL[dataset]}
            </span>
          </div>
          <div className="text-[11px] text-slate-400">
            Upload any image for visual results. Add an optional YOLO label
            (.txt) for P/R/F1 metrics, or pick a random validation image (GT
            included).
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-px bg-ink-600/30">
        {/* Current image */}
        <div className="bg-ink-900/40 p-4 flex flex-col">
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-2">
            Current
          </div>
          <div className="aspect-video rounded-lg bg-ink-900/60 border border-ink-600/40 grid place-items-center overflow-hidden relative">
            {imageLoading && (
              <Skeleton className="absolute inset-0 rounded-lg" variant="shimmer" />
            )}
            <img
              src={api.originalImageUrl(dataset, bust)}
              alt="current test"
              className={`w-full h-full object-contain transition-opacity duration-300 ${
                imageLoading ? "opacity-0" : "opacity-100"
              }`}
              onLoad={() => setImageLoading(false)}
              onError={(e) => {
                setImageLoading(false);
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
            />
          </div>
          <div className="text-[11px] text-slate-500 mt-2 font-mono truncate">
            data/demo/test_image_{dataset}.jpg
          </div>
        </div>

        {/* Drop zone */}
        <div
          className={`p-4 flex flex-col gap-3 transition ${
            dragOver ? "bg-accent-500/10" : "bg-ink-900/30"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="text-[11px] uppercase tracking-widest text-slate-400">
              Automatic input
            </div>
            <Tooltip
              content={`Pick a random validation image from ${DATASET_LABEL[dataset]}; ground-truth label is loaded automatically.`}
              side="left"
            >
              <button
                onClick={pickFromDataset}
                disabled={busy}
                className="inline-flex items-center gap-1.5 text-[12px] px-2.5 py-1 rounded-md border border-accent-500/40 bg-accent-500/10 hover:bg-accent-500/20 text-accent-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {busy ? (
                  <EagleSpinner size="sm" />
                ) : (
                  <Shuffle size={12} />
                )}
                Pick random validation image
              </button>
            </Tooltip>
          </div>

          <button
            onClick={() => imgInputRef.current?.click()}
            disabled={busy}
            className={`relative aspect-video rounded-lg border-2 border-dashed grid place-items-center transition cursor-pointer ${
              dragOver
                ? "border-accent-400 bg-accent-500/10"
                : "border-ink-500 hover:border-accent-400/60 bg-ink-900/40"
            }`}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="preview"
                className="w-full h-full object-contain rounded-lg"
              />
            ) : (
              <div className="flex flex-col items-center gap-2 text-slate-400 text-sm">
                <Upload size={22} className="text-accent-400" />
                <span>
                  Drop an image here or{" "}
                  <span className="text-accent-300 underline">browse</span>
                </span>
                <span className="text-[11px] text-slate-500">
                  jpg / png / bmp / webp
                </span>
              </div>
            )}
          </button>
          <input
            ref={imgInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => pickImage(e.target.files?.[0] ?? null)}
          />

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => labelInputRef.current?.click()}
              disabled={busy}
              className="text-[12px] px-2.5 py-1 rounded-md border border-ink-500 bg-ink-700/40 hover:bg-ink-700/70 text-slate-300 disabled:opacity-50"
            >
              {label ? label.name : "Optional: YOLO label (.txt)"}
            </button>
            {label && (
              <button
                type="button"
                onClick={() => {
                  setLabel(null);
                  if (labelInputRef.current) labelInputRef.current.value = "";
                }}
                className="text-slate-500 hover:text-slate-300 text-xs"
              >
                clear
              </button>
            )}
          </div>
          <input
            ref={labelInputRef}
            type="file"
            accept=".txt,text/plain"
            className="hidden"
            onChange={(e) => setLabel(e.target.files?.[0] ?? null)}
          />

          {/* Action row */}
          <div className="flex items-center justify-end gap-2 mt-1">
            <Tooltip content="Upload this image as the active test input." side="top">
              <button
                onClick={submit}
                disabled={busy || !image}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent-500 hover:bg-accent-400 disabled:opacity-50 disabled:cursor-not-allowed text-ink-900 font-semibold transition"
              >
                {busy ? (
                  <>
                    <EagleSpinner size="sm" />
                    Uploading…
                  </>
                ) : (
                  <>
                    <Upload size={14} />
                    Use this image
                  </>
                )}
              </button>
            </Tooltip>
          </div>

          {success && (
            <div className="text-[12px] inline-flex items-start gap-1.5 text-good">
              <CheckCircle2 size={12} className="mt-0.5" />
              <span>{success}</span>
            </div>
          )}
          {error && (
            <div className="text-[12px] inline-flex items-start gap-1.5 text-bad">
              <XCircle size={12} className="mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
