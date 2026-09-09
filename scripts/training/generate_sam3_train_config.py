#!/usr/bin/env python3
"""Write a SAM3 fine-tune Hydra config for Meta's sam3.train.train."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIST_BACKEND = "gloo" if sys.platform == "win32" else "nccl"

ROOT = Path(__file__).resolve().parents[2]
SAM3_CONFIG_DIR = ROOT / "external" / "sam3" / "sam3" / "train" / "configs" / "eagleeye"


def write_config(
    dataset: str,
    coco_root: Path,
    log_dir: Path,
    bpe_path: Path,
    checkpoint_path: Path,
    max_epochs: int,
    limit_ids: int,
    gpus: int,
    tag: str,
) -> Path:
    SAM3_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = f"eagleeye_{dataset}_{tag}.yaml"
    cfg_path = SAM3_CONFIG_DIR / cfg_name
    limit_line = f"limit_ids: {limit_ids}" if limit_ids > 0 else "limit_ids: null"
    num_images = limit_ids if limit_ids > 0 else "null"

    yaml_text = f"""# @package _global_
defaults:
  - _self_

paths:
  dataset_root: {coco_root.as_posix()}
  experiment_log_dir: {log_dir.as_posix()}
  bpe_path: {bpe_path.as_posix()}
  checkpoint_path: {checkpoint_path.as_posix()}

eagleeye_train:
  num_images: {num_images}
  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
        - _target_: sam3.train.transforms.point_sampling.RandomizeInputBbox
          box_noise_std: 0.1
          box_noise_max: 20
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${{scratch.resolution}}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${{scratch.resolution}}
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${{scratch.max_ann_per_img}}

  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${{scratch.resolution}}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: False
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}

  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    normalization: local
    matcher: ${{scratch.matcher}}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict:
          loss_bbox: 5.0
          loss_giou: 2.0
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict:
          loss_ce: 20.0
          presence_loss: 20.0
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 50
        pad_scale_pos: 1.0
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${{scratch.scale_by_find_batch_size}}

scratch:
  enable_segmentation: False
  d_model: 256
  use_presence_eval: True
  original_box_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    use_original_ids: true
    use_original_sizes_box: true
    use_presence: ${{scratch.use_presence_eval}}
  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True
  resolution: 1008
  consistent_transform: False
  max_ann_per_img: 50
  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]
  num_train_workers: 0
  num_val_workers: 0
  max_data_epochs: 20
  target_epoch_size: 1500
  hybrid_repeats: 1
  context_length: 2
  gather_pred_via_filesys: false
  lr_scale: 0.1
  lr_transformer: ${{times:8e-4,${{scratch.lr_scale}}}}
  lr_vision_backbone: ${{times:2.5e-4,${{scratch.lr_scale}}}}
  lr_language_backbone: ${{times:5e-5,${{scratch.lr_scale}}}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20
  val_batch_size: 1
  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: eagleeye
    with_seg_masks: ${{scratch.enable_segmentation}}
  gradient_accumulation_steps: 1
  train_batch_size: 1
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: all
    with_seg_masks: ${{scratch.enable_segmentation}}

trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: False
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: True
  max_epochs: {max_epochs}
  accelerator: cuda
  seed_value: 123
  val_epoch_freq: 5
  mode: train
  gradient_accumulation_steps: ${{scratch.gradient_accumulation_steps}}
  distributed:
    backend: {_DIST_BACKEND}
    find_unused_parameters: True
    gradient_as_bucket_view: True
  loss:
    all: ${{eagleeye_train.loss}}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        {limit_line}
        transforms: ${{eagleeye_train.train_transforms}}
        load_segmentation: ${{scratch.enable_segmentation}}
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
        img_folder: ${{paths.dataset_root}}/train/images
        ann_file: ${{paths.dataset_root}}/train/_annotations.coco.json
      shuffle: True
      batch_size: ${{scratch.train_batch_size}}
      num_workers: ${{scratch.num_train_workers}}
      pin_memory: True
      drop_last: True
      collate_fn: ${{scratch.collate_fn}}
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        load_segmentation: ${{scratch.enable_segmentation}}
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: 2
          _partial_: true
        img_folder: ${{paths.dataset_root}}/val/images
        ann_file: ${{paths.dataset_root}}/val/_annotations.coco.json
        transforms: ${{eagleeye_train.val_transforms}}
        max_ann_per_img: 100000
        multiplier: 1
        training: false
      shuffle: False
      batch_size: ${{scratch.val_batch_size}}
      num_workers: ${{scratch.num_val_workers}}
      pin_memory: True
      drop_last: False
      collate_fn: ${{scratch.collate_fn_val}}
  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${{paths.bpe_path}}
    checkpoint_path: ${{paths.checkpoint_path}}
    load_from_HF: False
    device: cpus
    eval_mode: false
    enable_segmentation: ${{scratch.enable_segmentation}}
  optim:
    amp:
      enabled: True
      amp_dtype: float16
    optimizer:
      _target_: torch.optim.AdamW
    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2
    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${{scratch.lrd_vision_backbone}}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0
    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_transformer}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_vision_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_language_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.language_backbone.*'
      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${{scratch.wd}}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']
  checkpoint:
    save_dir: ${{launcher.experiment_log_dir}}/checkpoints
    save_freq: 0
  logging:
    tensorboard_writer:
      _target_: sam3.train.utils.logger.make_tensorboard_logger
      log_dir: ${{launcher.experiment_log_dir}}/tensorboard
      flush_secs: 120
      should_log: True
    wandb_writer: null
    log_dir: ${{launcher.experiment_log_dir}}/logs
    log_freq: 10

launcher:
  num_nodes: 1
  gpus_per_node: {gpus}
  experiment_log_dir: ${{paths.experiment_log_dir}}
  multiprocessing_context: spawn

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 72
  use_cluster: False
  cpus_per_task: 4
  port_range: [10000, 65000]
  constraint: null
"""
    cfg_path.write_text(yaml_text, encoding="utf-8")

    # Mirror a copy under runs/ for reference
    mirror = log_dir / cfg_name
    mirror.write_text(yaml_text, encoding="utf-8")
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["visdrone", "kitti"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--tag", default="", help="Config suffix (default: smoke or full)")
    parser.add_argument("--sam3-repo", default=str(ROOT / "external" / "sam3"))
    args = parser.parse_args()

    tag = args.tag or ("smoke" if args.limit > 0 else "full")
    coco_root = ROOT / "data" / "sam3_coco" / args.dataset
    log_dir = ROOT / "runs" / "sam3_finetune" / args.dataset / tag
    bpe = Path(args.sam3_repo) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    checkpoint = ROOT / "sam3.pt"
    if not bpe.is_file():
        print(f"[WARN] BPE not found yet: {bpe}")
    if not checkpoint.is_file():
        print(f"[WARN] Base checkpoint not found: {checkpoint}")

    cfg = write_config(
        args.dataset, coco_root, log_dir, bpe, checkpoint,
        args.epochs, args.limit, args.gpus, tag,
    )
    hydra_name = f"configs/eagleeye/{cfg.name}"
    print(f"[OK] Config: {cfg}")
    print(f"[OK] Hydra name: {hydra_name}")


if __name__ == "__main__":
    main()
