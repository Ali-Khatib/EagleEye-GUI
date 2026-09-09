import {
  ArrowRight,
  Camera,
  Layers,
  Scan,
  Sparkles,
  Target,
  Timer,
} from "lucide-react";

interface Props {
  onStart: () => void;
}

const PIPELINES = [
  { name: "YOLO", desc: "Fast object detection", icon: <Target size={16} /> },
  { name: "SAHI + YOLO", desc: "Small objects", icon: <Layers size={16} /> },
  { name: "SAM 3", desc: "Text-prompt masks", icon: <Scan size={16} /> },
  { name: "SAM 3 + YOLO", desc: "Detect + refine", icon: <Sparkles size={16} /> },
  { name: "YOLO + SAHI + SAM 3", desc: "Main pipeline", icon: <Sparkles size={16} /> },
  { name: "YOLOv8 + SAHI + SAM 3", desc: "Paper compare", icon: <Target size={16} /> },
];

export default function HomeTab({ onStart }: Props) {
  return (
    <div className="flex flex-col gap-12">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-2xl border border-ink-600/60 bg-ink-800/40 p-10 md:p-14 glow-border">
        <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-accent-500/10 blur-3xl" />
        <div className="absolute -bottom-24 -left-24 w-96 h-96 rounded-full bg-accent-600/10 blur-3xl" />

        <div className="relative">
          <div className="inline-flex items-center gap-2 text-xs uppercase tracking-widest text-accent-400 bg-accent-500/10 border border-accent-500/30 px-3 py-1 rounded-full">
            <Camera size={12} />
            EagleEye AI
          </div>

          <h1 className="mt-5 text-4xl md:text-5xl font-extrabold tracking-tight">
            6 computer vision pipelines.
            <br />
            <span className="bg-gradient-to-r from-accent-400 to-cyan-300 bg-clip-text text-transparent">
              YOLO, SAHI, and SAM 3
            </span>
          </h1>

          <p className="mt-4 text-slate-300 max-w-2xl leading-relaxed">
            Side-by-side benchmarking of <b>YOLO</b>, <b>SAHI</b>, and{" "}
            <b>SAM 3</b> on small-object detection &amp;
            segmentation tasks (VisDrone / KITTI). Run any pipeline against the
            same image and compare F1, precision, recall, accuracy, FPS, model
            size and parameter count.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button
              onClick={onStart}
              className="group inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent-500 hover:bg-accent-400 text-ink-900 font-semibold transition shadow-glow"
            >
              Choose a pipeline
              <ArrowRight
                size={16}
                className="group-hover:translate-x-0.5 transition-transform"
              />
            </button>
            <span className="text-xs text-slate-400 hidden md:inline">
              or upload your own image on the{" "}
              <span className="kbd">Pipelines</span> tab.
            </span>
          </div>

          {/* Stats */}
          <div className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Pipelines", value: "6" },
              { label: "Datasets", value: "VisDrone · KITTI" },
              { label: "Eval IoU", value: "0.50" },
              { label: "SAHI slice", value: "256 · 50%" },
            ].map((s) => (
              <div
                key={s.label}
                className="rounded-xl border border-ink-600/60 bg-ink-700/40 p-4"
              >
                <div className="text-[11px] uppercase tracking-widest text-slate-400">
                  {s.label}
                </div>
                <div className="mt-1 text-xl font-semibold">{s.value}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipelines preview grid */}
      <section>
        <div className="flex items-end justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold">Pipelines you can compare</h2>
            <p className="text-sm text-slate-400">
              Run them one by one and watch the leaderboard fill up.
            </p>
          </div>
          <button
            onClick={onStart}
            className="text-sm text-accent-400 hover:text-accent-300 inline-flex items-center gap-1"
          >
            Open Pipelines <ArrowRight size={14} />
          </button>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {PIPELINES.map((p) => (
            <div
              key={p.name}
              className="rounded-xl border border-ink-600/60 bg-ink-800/40 p-5 hover:border-accent-500/40 hover:bg-ink-700/40 transition"
            >
              <div className="flex items-center gap-2 text-accent-400">
                {p.icon}
                <span className="font-semibold text-white">{p.name}</span>
              </div>
              <p className="mt-2 text-sm text-slate-400">{p.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* What you get */}
      <section className="grid md:grid-cols-3 gap-4">
        {[
          {
            icon: <Target className="text-accent-400" />,
            title: "Per-object metrics",
            text: "Precision, recall, F1, accuracy with IoU = 0.5 matching against ground-truth YOLO labels.",
          },
          {
            icon: <Timer className="text-accent-400" />,
            title: "Runtime + model size",
            text: "Wall-clock seconds, FPS, YOLO parameter count, YOLO weight size and SAM checkpoint size on disk.",
          },
          {
            icon: <Sparkles className="text-accent-400" />,
            title: "Visual diff",
            text: "Side-by-side before / after image for every pipeline so you can eyeball detection quality fast.",
          },
        ].map((c) => (
          <div
            key={c.title}
            className="rounded-xl border border-ink-600/60 bg-ink-800/40 p-5"
          >
            <div className="w-9 h-9 rounded-lg bg-accent-500/10 grid place-items-center">
              {c.icon}
            </div>
            <div className="mt-3 font-semibold">{c.title}</div>
            <div className="text-sm text-slate-400 mt-1">{c.text}</div>
          </div>
        ))}
      </section>
    </div>
  );
}
