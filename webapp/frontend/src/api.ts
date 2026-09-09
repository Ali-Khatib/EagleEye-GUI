import type {
  Dataset,
  DatasetInfo,
  Experiment,
  ResultRow,
  RunResponse,
} from "./types";

const BASE = ""; // proxied via vite to FastAPI

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

async function postJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

async function deleteJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

interface HealthResponse {
  status: string;
  dataset: Dataset;
  image_exists: boolean;
  label_exists: boolean;
  yolo_path: string;
  yolo_ready: boolean;
  inference_device?: string;
  cuda_available?: boolean;
  gpu_name?: string | null;
}

export interface UploadResponse {
  ok: boolean;
  dataset: Dataset;
  image_path: string;
  image_size_bytes: number;
  label_saved: boolean;
  label_path: string | null;
  results_cleared: number;
}

export interface RandomSampleResponse {
  ok: boolean;
  dataset: Dataset;
  source_image: string;
  source_label: string | null;
  image_path: string;
  image_size_bytes: number;
  label_saved: boolean;
  label_path: string | null;
  label_count: number;
  label_format: "yolo" | "visdrone";
  results_cleared: number;
}

async function uploadMultipart(
  path: string,
  fields: Record<string, File | string | undefined>
): Promise<UploadResponse> {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined) continue;
    fd.append(k, v as Blob | string);
  }
  const res = await fetch(`${BASE}${path}`, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

const ds = (d: Dataset) => `dataset=${encodeURIComponent(d)}`;

export const api = {
  health: (d: Dataset) => getJson<HealthResponse>(`/api/health?${ds(d)}`),
  datasets: () => getJson<DatasetInfo[]>("/api/datasets"),
  experiments: (d: Dataset) =>
    getJson<Experiment[]>(`/api/experiments?${ds(d)}`),
  results: (d: Dataset) => getJson<ResultRow[]>(`/api/results?${ds(d)}`),
  result: (d: Dataset, id: string) =>
    getJson<ResultRow>(`/api/results/${id}?${ds(d)}`),
  run: (d: Dataset, id: string) =>
    postJson<RunResponse>(`/api/experiments/${id}/run?${ds(d)}`),
  upload: (d: Dataset, image: File, label?: File) =>
    uploadMultipart("/api/upload", { image, label, dataset: d }),
  randomSample: (d: Dataset) =>
    postJson<RandomSampleResponse>(`/api/dataset/random-sample?${ds(d)}`),
  deleteLabel: (d: Dataset) =>
    deleteJson<{ ok: boolean; removed: boolean }>(`/api/label?${ds(d)}`),
  clearResults: (d: Dataset) =>
    deleteJson<{ ok: boolean; cleared: number }>(`/api/results?${ds(d)}`),
  videoInfo: (d: Dataset) =>
    getJson<{ ok: boolean; ready: boolean; path: string | null }>(
      `/api/video?${ds(d)}`
    ),
  uploadVideo: async (d: Dataset, video: File) => {
    const fd = new FormData();
    fd.append("video", video);
    const res = await fetch(`${BASE}/api/video/upload?${ds(d)}`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return res.json() as Promise<{ ok: boolean; path: string; bytes: number }>;
  },
  videoStreamUrl: (d: Dataset, mode: string, bust?: number) =>
    `/api/video/stream?${ds(d)}&mode=${encodeURIComponent(mode)}${
      bust ? `&t=${bust}` : ""
    }`,
  originalImageUrl: (d: Dataset, bust?: number | string) =>
    `/api/image/original?${ds(d)}${bust ? `&t=${bust}` : ""}`,
  resultImageUrl: (d: Dataset, id: string, bust?: number | string) =>
    `/api/image/result/${id}?${ds(d)}${bust ? `&t=${bust}` : ""}`,
};

export function fmt(value: string | number | undefined, digits = 3): string {
  if (value === undefined || value === null || value === "") return "—";
  if (value === "N/A") return "N/A";
  const n = typeof value === "number" ? value : Number(value);
  if (Number.isNaN(n)) return String(value);
  if (Number.isInteger(n)) return n.toString();
  return n.toFixed(digits);
}

export function fmtInt(value: string | number | undefined): string {
  if (value === undefined || value === null || value === "") return "—";
  if (value === "N/A") return "N/A";
  const n = typeof value === "number" ? value : Number(value);
  if (Number.isNaN(n)) return String(value);
  return Math.round(n).toLocaleString();
}

/** True when metrics.csv has valid ground-truth evaluation (not placeholder zeros). */
export function hasGtMetrics(m?: { has_reliable_gt?: string; precision?: string } | null): boolean {
  if (!m) return false;
  const flag = String(m.has_reliable_gt ?? "").toLowerCase();
  if (flag === "false") return false;
  if (flag === "true") return true;
  return m.precision !== "N/A" && m.precision !== undefined && m.precision !== "";
}

export function fmtScore(
  m: { has_reliable_gt?: string; precision?: string } | null | undefined,
  value: string | number | undefined,
  digits = 3
): string {
  if (!hasGtMetrics(m)) return "N/A";
  return fmt(value, digits);
}

export function fmtScoreInt(
  m: { has_reliable_gt?: string; precision?: string } | null | undefined,
  value: string | number | undefined
): string {
  if (!hasGtMetrics(m)) return "N/A";
  return fmtInt(value);
}
