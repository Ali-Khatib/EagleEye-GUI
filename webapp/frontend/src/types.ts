export type Dataset = "visdrone" | "kitti" | "stock";

export const DATASETS: Dataset[] = ["visdrone", "kitti", "stock"];
export const DATASET_LABEL: Record<Dataset, string> = {
  visdrone: "VisDrone",
  kitti: "KITTI",
  stock: "Stock (COCO)",
};

export interface DatasetInfo {
  id: Dataset;
  label: string;
  yolo_path: string;
  yolo_ready: boolean;
  sam_path?: string;
  sam_ready?: boolean;
  image_path: string;
  image_ready: boolean;
  label_path: string;
  label_ready: boolean;
}

export interface Experiment {
  id: string;
  key: string;
  name: string;
  subtitle: string;
  description: string;
  components: string[];
  script: string;
  basename: string;
  dataset: Dataset;
  metric_kind: "class_aware" | "class_agnostic";
  has_result: boolean;
  run_status: "idle" | "running" | "done" | "error";
  metrics?: Metrics | null;
  elapsed?: number;
  weights_ready?: boolean;
  needs_yolov8?: boolean;
  visdrone_only?: boolean;
  yolov8_path?: string;
}

export interface Metrics {
  mode_name?: string;
  dataset?: string;
  dataset_image?: string;
  count?: string;
  runtime_seconds?: string;
  fps?: string;
  yolo_model_size_mb?: string;
  yolo_parameter_count?: string;
  sam_checkpoint_size_mb?: string;
  precision?: string;
  recall?: string;
  f1_score?: string;
  accuracy?: string;
  iou_threshold?: string;
  true_positives?: string;
  false_positives?: string;
  false_negatives?: string;
  ground_truth_count?: string;
  predicted_count?: string;
  inference_device?: string;
  gpu_name?: string;
  notes?: string;
  has_reliable_gt?: string;
  mask_count?: string;
}

export interface ResultRow extends Experiment {
  metrics: Metrics | null;
  elapsed?: number;
}

export interface RunResponse {
  ok: boolean;
  exp_id: string;
  dataset?: Dataset;
  status?: "running" | "done" | "error";
  elapsed: number;
  stdout: string;
  stderr: string;
  metrics: Metrics | null;
}
