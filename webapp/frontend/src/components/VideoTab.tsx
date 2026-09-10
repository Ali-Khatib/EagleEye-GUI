import { useEffect, useRef, useState } from "react";
import { Play, Square, Upload } from "lucide-react";
import { api } from "../api";
import { DATASET_LABEL, type Dataset } from "../types";
import Tooltip from "./ui/Tooltip";
import Button from "./ui/Button";
import { Eyebrow } from "./ui/TextReveal";
import { cn } from "../lib/cn";

type LiveMode = "yolo_only" | "yolo_sahi";

interface Props {
  dataset: Dataset;
}

const MODES: { id: LiveMode; label: string; hint: string }[] = [
  {
    id: "yolo_only",
    label: "YOLO",
    hint: "Fast live path (~paper 70+ FPS on KITTI / VisDrone YOLO-only).",
  },
  {
    id: "yolo_sahi",
    label: "YOLO + SAHI",
    hint: "Sliced inference — slower, better small objects.",
  },
];

export default function VideoTab({ dataset }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<LiveMode>("yolo_only");
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bust, setBust] = useState(0);

  const refresh = async () => {
    try {
      const info = await api.videoInfo(dataset);
      setReady(info.ready);
    } catch {
      setReady(false);
    }
  };

  useEffect(() => {
    setPlaying(false);
    setError(null);
    refresh();
  }, [dataset]);

  const upload = async (file: File | null) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setPlaying(false);
    try {
      await api.uploadVideo(dataset, file);
      await refresh();
      setBust(Date.now());
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const start = () => {
    if (!ready) {
      setError("Upload a video first.");
      return;
    }
    setError(null);
    setBust(Date.now());
    setPlaying(true);
  };

  const streamSrc =
    playing && ready ? api.videoStreamUrl(dataset, mode, bust) : "";

  return (
    <div className="bg-horizon-navy text-paper min-h-[calc(100svh-72px)]">
      <div className="mx-auto max-w-page px-5 md:px-8 pt-28 pb-16 md:pb-24">
        <Eyebrow className="text-paper/50">
          Video · {DATASET_LABEL[dataset]}
        </Eyebrow>
        <h1 className="mt-4 text-heading-sm md:text-heading-lg font-medium tracking-[-0.038em] max-w-3xl">
          See EagleEye in motion.
        </h1>
        <p className="mt-4 max-w-2xl text-body text-paper/70">
          Live annotated frames with YOLO or YOLO + SAHI. SAM 3 is available for
          still-image evaluation — it is not in the live path (~0.02 FPS).
        </p>

        <div className="mt-12 overflow-hidden rounded-card bg-coal">
          <div className="aspect-video grid place-items-center min-h-[360px]">
            {playing && streamSrc ? (
              <img
                src={streamSrc}
                alt="Live detections"
                className="w-full h-full object-contain"
              />
            ) : (
              <p className="text-body-sm text-paper/50 p-8 text-center">
                {ready
                  ? "Press Play to stream annotated frames."
                  : "Upload an mp4 to start."}
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-4 md:gap-8 px-4 md:px-6 py-4 border-t border-white/10 text-caption">
            <Hud label="Pipeline" value={mode === "yolo_only" ? "YOLO" : "YOLO + SAHI"} />
            <Hud label="Dataset" value={DATASET_LABEL[dataset]} />
            <Hud label="Stream" value={playing ? "Live" : "Idle"} />
            <Hud label="SAM 3" value="Still images only" />
          </div>
        </div>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          {MODES.map((m) => (
            <Tooltip key={m.id} content={m.hint} side="bottom">
              <button
                onClick={() => {
                  setMode(m.id);
                  if (playing) setBust(Date.now());
                }}
                className={cn(
                  "px-4 py-3 rounded-[8px] text-body-sm font-medium border transition-colors duration-300",
                  mode === m.id
                    ? "border-signal-blue text-paper bg-signal-blue"
                    : "border-white/20 text-paper/80 hover:border-white/40"
                )}
              >
                {m.label}
              </button>
            </Tooltip>
          ))}

          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,video/x-msvideo,.mp4,.webm,.mov,.avi,.mkv"
            className="hidden"
            onChange={(e) => upload(e.target.files?.[0] ?? null)}
          />
          <Button
            variant="secondary"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
          >
            <Upload size={14} />
            {busy ? "Uploading…" : "Upload video"}
          </Button>
          {!playing ? (
            <Button className="group" arrow onClick={start}>
              <Play size={14} />
              Play
            </Button>
          ) : (
            <Button variant="secondary" onClick={() => setPlaying(false)}>
              <Square size={14} />
              Stop
            </Button>
          )}
        </div>

        {error && (
          <p className="mt-4 text-body-sm text-paper/80 border border-white/15 rounded-[8px] px-4 py-3">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

function Hud({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="uppercase tracking-[0.1em] text-paper/40">{label}</div>
      <div className="mt-0.5 text-paper">{value}</div>
    </div>
  );
}
