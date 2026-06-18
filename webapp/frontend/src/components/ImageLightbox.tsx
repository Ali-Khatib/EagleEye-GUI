import { useCallback, useEffect } from "react";
import { X, ZoomIn } from "lucide-react";

export interface LightboxImage {
  src: string;
  title: string;
  alt?: string;
}

interface Props {
  image: LightboxImage | null;
  onClose: () => void;
}

export function ImageLightbox({ image, onClose }: Props) {
  const onKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose]
  );

  useEffect(() => {
    if (!image) return;
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [image, onKey]);

  if (!image) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex flex-col bg-ink-950/95 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={image.title}
      onClick={onClose}
    >
      <div
        className="flex items-center justify-between gap-3 px-4 py-3 border-b border-ink-600/60 shrink-0"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-sm font-medium text-slate-200 truncate">
          {image.title}
        </div>
        <p className="text-[11px] text-slate-500 hidden sm:block">
          Esc or click outside to close
        </p>
        <button
          type="button"
          onClick={onClose}
          className="p-2 rounded-lg hover:bg-ink-700 text-slate-300"
          aria-label="Close"
        >
          <X size={20} />
        </button>
      </div>

      <div
        className="flex-1 min-h-0 flex items-center justify-center p-2 sm:p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={image.src}
          alt={image.alt ?? image.title}
          className="max-w-[min(100%,96vw)] max-h-[min(100%,88vh)] w-auto h-auto object-contain select-none shadow-2xl"
          draggable={false}
        />
      </div>
    </div>
  );
}

export const lightboxImageClass =
  "cursor-zoom-in hover:opacity-95 transition-opacity";

export function LightboxHint() {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-slate-500">
      <ZoomIn size={10} />
      double-click to maximize
    </span>
  );
}
