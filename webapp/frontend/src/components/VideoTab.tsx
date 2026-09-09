import { useEffect, useRef, useState } from "react";
import { Film, Play, Square, Upload } from "lucide-react";
import { api } from "../api";
import { DATASET_LABEL, type Dataset } from "../types";
import Tooltip from "./ui/Tooltip";

type LiveMode = "yolo_only" | "yolo_sahi";

interface Props {
  dataset: Dataset;
}

const MODES: { id: LiveMode; label: string; hint: string }[] = [
  {
    id: "yolo_only",
    label: "YOLO only",
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
    playing && ready
      ? api.videoStreamUrl(dataset, mode, bust)
      : "";

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <Film size={22} className="text-accent-400" />
          Video
          <span className="text-xs font-normal text-slate-400 border border-ink-600 rounded-full px-2 py-0.5">
            {DATASET_LABEL[dataset]}
          </span>
        </h1>
        <p className="mt-2 text-sm text-slate-400 max-w-3xl">
          Live stream with the same library toggles as the OpenCV demo.{" "}
          <b className="text-slate-200">YOLO</b> or <b className="text-slate-200">YOLO + SAHI</b>{" "}
          run on each frame. Pipeline 5 (SAM 3 text fill + masks) is not live —
          it is ~0.02 FPS; run it on a still in Pipelines.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {MODES.map((m) => (
          <Tooltip key={m.id} content={m.hint} side="bottom">
            <button
              onClick={() => {
                setMode(m.id);
                if (playing) setBust(Date.now());
              }}
              className={`px-4 py-2 rounded-lg text-sm border transition ${
                mode === m.id
                  ? "bg-accent-500/15 text-accent-400 border-accent-500/40"
                  : "text-slate-300 border-ink-600 hover:border-ink-500"
              }`}
            >
              {m.label}
            </button>
          </Tooltip>
        ))}
      </div>

      <div className="grid md:grid-cols-[1fr_280px] gap-6">
        <div className="rounded-xl border border-ink-600/60 bg-ink-800/40 overflow-hidden min-h-[360px] grid place-items-center">
          {playing && streamSrc ? (
            <img
              src={streamSrc}
              alt="Live detections"
              className="w-full h-full object-contain bg-black"
            />
          ) : (
            <div className="text-center text-slate-500 text-sm p-8">
              {ready
                ? "Press Play to stream annotated frames."
                : "Upload an mp4 to start."}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3">
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,video/x-msvideo,.mp4,.webm,.mov,.avi,.mkv"
            className="hidden"
            onChange={(e) => upload(e.target.files?.[0] ?? null)}
          />
          <button
            disabled={busy}
            onClick={() => inputRef.current?.click()}
            className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-ink-700 border border-ink-600 text-sm hover:bg-ink-600"
          >
            <Upload size={16} />
            {busy ? "Uploading…" : "Upload video"}
          </button>
          <div className="flex gap-2">
            {!playing ? (
              <button
                onClick={start}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-accent-500 text-ink-900 font-semibold"
              >
                <Play size={16} />
                Play
              </button>
            ) : (
              <button
                onClick={() => setPlaying(false)}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-ink-600 text-sm"
              >
                <Square size={16} />
                Stop
              </button>
            )}
          </div>
          <p className="text-xs text-slate-500">
            Switching YOLO / SAHI restarts the stream with that library.
          </p>
          {error && (
            <div className="text-sm text-bad border border-bad/40 bg-bad/10 rounded-lg px-3 py-2">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
