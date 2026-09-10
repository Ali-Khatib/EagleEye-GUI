import { ArrowRight, Layers, Scan, Target } from "lucide-react";
import type { ReactNode } from "react";
import type { Dataset } from "../types";
import FlowArt, { FlowSection } from "./ui/story-scroll";
import { cn } from "../lib/cn";

interface Props {
  dataset: Dataset;
  onStart: () => void;
  onResults: () => void;
  onVideo: () => void;
}

/** Unsplash License, free to use. */
const IMG = {
  aerialCity:
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?auto=format&fit=crop&w=2400&q=85",
  highway:
    "https://images.unsplash.com/photo-1465447142348-e9952c393450?auto=format&fit=crop&w=2400&q=85",
  drone:
    "https://images.unsplash.com/photo-1473968512647-3e447244af8f?auto=format&fit=crop&w=2400&q=85",
  urbanNight:
    "https://images.unsplash.com/photo-1519501025264-65ba15a82390?auto=format&fit=crop&w=2400&q=85",
  road:
    "https://images.unsplash.com/photo-1502481851512-e9e2529bfbf9?auto=format&fit=crop&w=2400&q=85",
};

export default function HomeTab({ onStart, onResults, onVideo }: Props) {
  return (
    <FlowArt aria-label="EagleEye story">
      <FlowSection
        aria-label="EagleEye introduction"
        className="text-paper"
        background={
          <>
            <video
              className="absolute inset-0 h-full w-full object-cover"
              autoPlay
              muted
              loop
              playsInline
              poster={IMG.aerialCity}
            >
              <source src="/media/hero-aerial.mp4" type="video/mp4" />
            </video>
            <div className="absolute inset-0 bg-gradient-to-r from-horizon-navy/55 via-horizon-navy/25 to-transparent" />
          </>
        }
      >
        <p className="text-sm md:text-base font-medium uppercase tracking-[0.12em] text-paper/80">
          01 EagleEye AI · Computer vision
        </p>
        <div className="flex flex-col gap-4 max-w-5xl mt-3">
          <h1 className="text-[clamp(2.4rem,6.5vw,5.5rem)] font-medium leading-[0.94] tracking-[-0.04em]">
            Seeing the small.
            <br />
            Understanding the scene.
          </h1>
          <p className="max-w-[42ch] text-[clamp(1.1rem,1.8vw,1.5rem)] leading-snug text-paper/90">
            Multi stage detection and segmentation with YOLO, SAHI, and SAM 3.
          </p>
          <div className="flex flex-wrap gap-3">
            <ActionButton onClick={onStart} primary>
              Explore the pipelines
              <ArrowRight size={20} aria-hidden />
            </ActionButton>
            <ActionButton onClick={onResults}>View results</ActionButton>
            <ActionButton onClick={onVideo}>Watch live video</ActionButton>
          </div>
        </div>
        <div className="mt-auto grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-paper/30">
          {[
            ["6", "Vision pipelines"],
            ["3", "Datasets"],
            ["YOLO · SAHI · SAM 3", "Stack"],
            ["IoU 0.50", "Evaluation"],
          ].map(([k, v]) => (
            <div key={v}>
              <div className="text-xs md:text-sm font-medium uppercase tracking-[0.1em] text-paper/60">
                {v}
              </div>
              <div className="mt-1 text-lg md:text-xl">{k}</div>
            </div>
          ))}
        </div>
      </FlowSection>

      <FlowSection
        aria-label="The research problem"
        style={{ backgroundColor: "#ffffff", color: "#001733" }}
      >
        <p className="text-sm md:text-base font-medium uppercase tracking-[0.12em] text-slate-whisper">
          02 The problem
        </p>
        <div className="mt-3 grid lg:grid-cols-2 gap-6 items-center flex-1 min-h-0">
          <div>
            <h2 className="text-[clamp(2.2rem,5.5vw,4.5rem)] font-medium leading-[0.94] tracking-[-0.038em]">
              Small objects
              <br />
              break
              <br />
              detectors.
            </h2>
            <p className="mt-4 max-w-[46ch] text-[clamp(1.05rem,1.7vw,1.4rem)] leading-snug">
              Conventional full frame detectors lose the pixels that matter:
              distant cars, pedestrians, and crowded scenes. EagleEye restacks
              detection so those objects get another look.
            </p>
          </div>
          <img
            src={IMG.drone}
            alt="Aerial landscape, Unsplash"
            className="w-full h-full min-h-0 max-h-[62vh] object-cover rounded-[8px]"
          />
        </div>
      </FlowSection>

      <FlowSection
        aria-label="How the stack works"
        style={{ backgroundColor: "#001733", color: "#ffffff" }}
      >
        <p className="text-sm md:text-base font-medium uppercase tracking-[0.12em] text-paper/80">
          03 The stack
        </p>
        <h2 className="mt-3 text-[clamp(2.2rem,5.5vw,4.5rem)] font-medium leading-[0.94] tracking-[-0.038em]">
          One image.
          <br />
          Six ways to see it.
        </h2>
        <div className="mt-4 grid md:grid-cols-3 gap-5">
          <StackNote
            icon={<Target size={22} />}
            title="YOLO"
            body="Fast full frame detection. The baseline everything else is measured against."
          />
          <StackNote
            icon={<Layers size={22} />}
            title="SAHI"
            body="Tiles the frame, infers at higher effective resolution, then merges boxes."
          />
          <StackNote
            icon={<Scan size={22} />}
            title="SAM 3"
            body="Open vocabulary prompts and mask refinement inside each detection."
          />
        </div>
        <img
          src={IMG.highway}
          alt="Highway interchange, Unsplash"
          className="mt-4 w-full flex-1 min-h-0 object-cover rounded-[8px]"
        />
      </FlowSection>

      <FlowSection
        aria-label="How to use EagleEye"
        style={{ backgroundColor: "#f3f4f8", color: "#001733" }}
      >
        <p className="text-sm md:text-base font-medium uppercase tracking-[0.12em] text-slate-whisper">
          04 How to run it
        </p>
        <h2 className="mt-3 text-[clamp(2.2rem,5.5vw,4.5rem)] font-medium leading-[0.94] tracking-[-0.038em]">
          Pick a scene. Run a pipeline.
        </h2>
        <div className="mt-4 grid lg:grid-cols-2 gap-6 items-stretch flex-1 min-h-0">
          <div className="grid gap-4 content-start">
            <div>
              <p className="mb-1 text-lg md:text-xl font-medium uppercase tracking-wide text-signal-blue">
                01 Choose a dataset
              </p>
              <p className="text-base md:text-lg leading-snug text-horizon-navy">
                Hover the plane, car, or image icons in the top bar, then click
                VisDrone, KITTI, or Stock (COCO).
              </p>
            </div>
            <div>
              <p className="mb-1 text-lg md:text-xl font-medium uppercase tracking-wide text-signal-blue">
                02 Open Pipelines
              </p>
              <p className="text-base md:text-lg leading-snug text-horizon-navy">
                Six architectures, same image. Run one, or run all. Pipeline 5 is
                the hybrid: YOLO + SAHI + SAM 3.
              </p>
            </div>
            <div>
              <p className="mb-1 text-lg md:text-xl font-medium uppercase tracking-wide text-signal-blue">
                03 Read Results
              </p>
              <p className="text-base md:text-lg leading-snug text-horizon-navy">
                Precision, recall, F1, FPS. Drag the comparison slider, then try
                live YOLO or YOLO + SAHI on Video.
              </p>
            </div>
          </div>
          <img
            src={IMG.urbanNight}
            alt="Urban traffic at dusk, Unsplash"
            className="w-full h-full min-h-0 object-cover rounded-[8px]"
          />
        </div>
      </FlowSection>

      <FlowSection
        aria-label="Enter the workbench"
        className="text-paper"
        background={
          <>
            <img
              src={IMG.road}
              alt="Road ahead, Unsplash"
              className="absolute inset-0 h-full w-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-horizon-navy/60 via-horizon-navy/25 to-transparent" />
          </>
        }
      >
        <p className="text-sm md:text-base font-medium uppercase tracking-[0.12em] text-paper/70">
          05 Start
        </p>
        <h2 className="mt-3 text-[clamp(2.2rem,5.5vw,4.5rem)] font-medium leading-[0.94] tracking-[-0.038em]">
          Measure the difference.
        </h2>
        <p className="mt-4 max-w-[44ch] text-[clamp(1.1rem,1.8vw,1.5rem)] leading-snug text-paper/90">
          Accuracy is only half of the problem. SAHI recovers small objects and
          costs runtime. The workbench is where those tradeoffs become visible.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <ActionButton onClick={onStart} primary>
            Open pipelines
            <ArrowRight size={20} aria-hidden />
          </ActionButton>
          <ActionButton onClick={onResults}>Open results</ActionButton>
          <ActionButton onClick={onVideo}>Open video</ActionButton>
        </div>
      </FlowSection>
    </FlowArt>
  );
}

function ActionButton({
  children,
  onClick,
  primary,
}: {
  children: ReactNode;
  onClick: () => void;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group inline-flex items-center justify-center gap-3 rounded-[8px] px-6 py-3.5 text-base md:text-lg font-medium transition-transform duration-300 hover:-translate-y-0.5",
        primary
          ? "bg-signal-blue text-paper"
          : "border border-mist bg-paper text-horizon-navy"
      )}
    >
      {children}
    </button>
  );
}

function StackNote({
  icon,
  title,
  body,
}: {
  icon: ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div>
      <div className="text-paper mb-1.5">{icon}</div>
      <p className="mb-1 text-lg md:text-xl font-medium uppercase tracking-wide">
        {title}
      </p>
      <p className="text-sm md:text-base leading-snug text-paper/80">{body}</p>
    </div>
  );
}
