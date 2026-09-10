import { useEffect, useRef, useState } from "react";
import { Shuffle, Upload } from "lucide-react";
import { api } from "../api";
import { DATASET_LABEL, type Dataset } from "../types";
import { EagleSpinner } from "./ui/EagleLoader";
import { Skeleton } from "./ui/Skeleton";
import Tooltip from "./ui/Tooltip";
import Button from "./ui/Button";
import { Eyebrow } from "./ui/TextReveal";
import { cn } from "../lib/cn";

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
    <section className="grid lg:grid-cols-2 gap-10 lg:gap-16 items-start">
      <div>
        <Eyebrow>Input</Eyebrow>
        <h2 className="mt-3 text-heading-sm font-medium tracking-[-0.038em]">
          Give a picture.
        </h2>
        <p className="mt-3 text-body text-slate-whisper">
          Upload any image for visual results. Add an optional YOLO label for
          P/R/F1, or pick a random {DATASET_LABEL[dataset]} validation frame.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Tooltip
            content={`Pick a random validation image from ${DATASET_LABEL[dataset]}; ground-truth label is loaded automatically.`}
          >
            <Button
              variant="secondary"
              onClick={pickFromDataset}
              disabled={busy}
            >
              {busy ? <EagleSpinner size="sm" /> : <Shuffle size={14} />}
              Random validation image
            </Button>
          </Tooltip>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="overflow-hidden rounded-card border border-mist bg-hailstone relative aspect-video">
          {imageLoading && (
            <Skeleton className="absolute inset-0 rounded-none" />
          )}
          <img
            src={api.originalImageUrl(dataset, bust)}
            alt="current test"
            className={cn(
              "w-full h-full object-contain transition-opacity duration-300",
              imageLoading ? "opacity-0" : "opacity-100"
            )}
            onLoad={() => setImageLoading(false)}
            onError={(e) => {
              setImageLoading(false);
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
        </div>
        <p className="text-caption text-slate-whisper">
          Current frame · data/demo/test_image_{dataset}.jpg
        </p>

        <button
          onClick={() => imgInputRef.current?.click()}
          disabled={busy}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={cn(
            "rounded-[8px] border border-dashed px-4 py-8 text-body-sm text-slate-whisper transition-colors duration-300",
            dragOver ? "border-signal-blue bg-hailstone" : "border-mist hover:border-signal-blue"
          )}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="preview" className="mx-auto max-h-40 object-contain" />
          ) : (
            <span className="inline-flex items-center gap-2">
              <Upload size={16} className="text-coal" />
              Drop an image or browse
            </span>
          )}
        </button>
        <input
          ref={imgInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => pickImage(e.target.files?.[0] ?? null)}
        />

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => labelInputRef.current?.click()}
            disabled={busy}
            className="text-caption px-2 py-1 rounded-badge border border-mist text-graphite-dim"
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
              className="text-caption text-slate-whisper"
            >
              clear
            </button>
          )}
          <Button
            className="group ml-auto"
            arrow
            onClick={submit}
            disabled={busy || !image}
          >
            {busy ? "Uploading…" : "Use this image"}
          </Button>
        </div>
        <input
          ref={labelInputRef}
          type="file"
          accept=".txt,text/plain"
          className="hidden"
          onChange={(e) => setLabel(e.target.files?.[0] ?? null)}
        />
        {success && <p className="text-caption text-good">{success}</p>}
        {error && <p className="text-caption text-bad">{error}</p>}
      </div>
    </section>
  );
}
