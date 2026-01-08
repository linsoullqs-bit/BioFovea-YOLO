#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级改进模型训练脚本 - 基于7篇小目标检测论文的系统化改进方案
支持多种注意力机制、P2检测头及其组合

理论依据：
1. SE/ECA/CoordAtt: 通道注意力增强特征表达 (SO-YOLOv8, MAE-YOLOv8等)
2. P2检测头: 提升小目标检测能力 (多篇文献证实对小目标最有效)
3. 组合策略: P2+注意力双重增强 (SOD-YOLO, SMA-YOLOv8等)

作者: AI Assistant
日期: 2025-10-31
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# 添加models目录到Python路径
models_dir = Path(__file__).parent / 'models'
sys.path.insert(0, str(models_dir))

from ultralytics import YOLO
import torch
from ultralytics.utils.loss import v8DetectionLoss, VarifocalLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.metrics import bbox_iou

# 导入自定义注意力模块
try:
    from models.cbam import CBAM, ECA, SE, CoordAtt
    print("✅ 已加载自定义注意力模块: CBAM, ECA, SE, CoordAtt")
except ImportError as e:
    print(f"⚠️ 注意力模块导入失败: {e}")
    print("将尝试在训练时动态注册...")

# 模型配置字典
MODEL_CONFIGS = {
    # ========== 基线模型 ==========
    'baseline': {
        'yaml': None,  # 使用默认yolov8s.pt
        'name': 'bubble_yolov8s_baseline',
        'description': 'YOLOv8s基线模型（无改进）',
        'theory': '作为对照组，评估其他改进的效果',
        'expected_gain': '基准：约0.44-0.47 mAP50-95',
        'vram': '约7GB',
        'speed': '基准速度',
        'disable_augment': False  # 使用标准增强
    },
    'baseline-800': {
        'yaml': None,  # 使用默认yolov8m.pt
        'name': 'baseline_800_simulation',
        'description': 'YOLOv8m (Baseline 8.0.0 Simulation)',
        'theory': '模拟Ultralytics 8.0.0版本baseline（关闭auto_augment和erasing）',
        'expected_gain': '基准：约0.45-0.48 mAP50-95',
        'vram': '约10GB',
        'speed': '0.8x基准',
        'disable_augment': True,  # 关闭特定增强
        'base_model': 'yolov8m.pt'  # 使用YOLOv8m
    },
    
    # ========== 仅注意力机制 ==========
    'cbam': {
        'yaml': 'models/yolov8s-cbam.yaml',
        'name': 'bubble_yolov8s_cbam',
        'description': 'YOLOv8s + CBAM (通道+空间注意力)',
        'theory': 'CBAM结合通道和空间注意力，全面增强特征表达',
        'expected_gain': '+1.5~2.5% mAP50-95',
        'vram': '约7.5GB',
        'speed': '0.9x基准',
        'disable_augment': False
    },
    'eca': {
        'yaml': 'models/yolov8s-eca.yaml',
        'name': 'bubble_yolov8s_eca',
        'description': 'YOLOv8s + ECA (高效通道注意力)',
        'theory': 'ECA是CBAM的轻量化版本，计算开销更小但效果相近',
        'expected_gain': '+1.0~2.0% mAP50-95',
        'vram': '约7GB',
        'speed': '0.98x基准'
    },
    'se': {
        'yaml': 'models/yolov8s-se.yaml',
        'name': 'bubble_yolov8s_se',
        'description': 'YOLOv8s + SE (挤压激励注意力)',
        'theory': 'SE是经典通道注意力，被SO-YOLOv8论文采用',
        'expected_gain': '+1.0~2.0% mAP50-95',
        'vram': '约7GB',
        'speed': '0.97x基准'
    },
    'coordatt': {
        'yaml': 'models/yolov8s-coordatt.yaml',
        'name': 'bubble_yolov8s_coordatt',
        'description': 'YOLOv8s + CoordAtt (坐标注意力)',
        'theory': 'CoordAtt编码通道和位置信息，对小目标空间定位更精准',
        'expected_gain': '+1.5~2.5% mAP50-95',
        'vram': '约7.2GB',
        'speed': '0.95x基准'
    },
    
    # ========== P2检测头（轻量版）==========
    'p2-lite': {
        'yaml': 'models/yolov8s-p2-lite.yaml',
        'name': 'bubble_yolov8s_p2_lite',
        'description': 'YOLOv8s + P2轻量化检测头',
        'theory': 'P2检测头（stride=4）对小目标最有效，Lite版减少显存占用',
        'expected_gain': '+1.5~3.0% mAP50-95',
        'vram': '约9-10GB',
        'speed': '0.75x基准'
    },
    'p2': {
        'yaml': 'models/yolov8s-p2.yaml',
        'name': 'bubble_yolov8s_p2',
        'description': 'YOLOv8s + P2完整检测头',
        'theory': 'P2完整版，提供最强小目标检测能力',
        'expected_gain': '+2.0~4.0% mAP50-95',
        'vram': '约11-12GB',
        'speed': '0.7x基准'
    },
    
    # ========== P2 + 注意力组合 ==========
    'p2-cbam': {
        'yaml': 'models/yolov8s-p2-cbam.yaml',
        'name': 'bubble_yolov8s_p2_cbam',
        'description': 'YOLOv8s + P2 + CBAM',
        'theory': 'P2小目标检测 + CBAM特征增强，双重提升',
        'expected_gain': '+2.5~4.5% mAP50-95',
        'vram': '约12GB',
        'speed': '0.65x基准'
    },
    'p2-eca': {
        'yaml': 'models/yolov8s-p2-eca.yaml',
        'name': 'bubble_yolov8s_p2_eca',
        'description': 'YOLOv8s + P2 + ECA',
        'theory': 'P2检测头 + 轻量ECA注意力，性能与速度平衡',
        'expected_gain': '+2.0~4.0% mAP50-95',
        'vram': '约10-11GB',
        'speed': '0.7x基准'
    },
    'p2-se': {
        'yaml': 'models/yolov8s-p2-se.yaml',
        'name': 'bubble_yolov8s_p2_se',
        'description': 'YOLOv8s + P2 + SE (复现SO-YOLOv8)',
        'theory': '复现SO-YOLOv8论文的核心方案',
        'expected_gain': '+2.0~4.0% mAP50-95',
        'vram': '约10-11GB',
        'speed': '0.7x基准'
    },
    'p2-se-varifocal': {
        'yaml': 'models/yolov8s-p2-se.yaml',
        'name': 'bubble_yolov8s_p2_se_varifocal',
        'description': 'YOLOv8s + P2 + SE + VarifocalLoss',
        'theory': 'P2检测头 + SE注意力，分类损失替换为VarifocalLoss以强调高质量正样本',
        'expected_gain': '+2.0~4.0% mAP50-95',
        'vram': '约10-11GB',
        'speed': '0.7x基准',
        'use_varifocal': True,
        'varifocal_alpha': 0.75,
        'varifocal_gamma': 2.0
    },
    'p2-coordatt': {
        'yaml': 'models/yolov8s-p2-coordatt.yaml',
        'name': 'bubble_yolov8s_p2_coordatt',
        'description': 'YOLOv8s + P2 + CoordAtt',
        'theory': 'P2检测 + 坐标注意力，空间定位最精准',
        'expected_gain': '+2.5~4.5% mAP50-95',
        'vram': '约11GB',
        'speed': '0.68x基准'
    },
}


def register_custom_modules():
    """
    注册自定义模块到ultralytics命名空间
    确保YAML文件能正确加载自定义注意力模块
    """
    try:
        import ultralytics.nn.modules as modules
        import ultralytics.nn.tasks as tasks
        
        # 导入自定义模块
        from models.cbam import CBAM, ECA, SE, CoordAtt
        
        # 注册到两个命名空间
        for module_name, module_class in [
            ('CBAM', CBAM), ('ECA', ECA), ('SE', SE), ('CoordAtt', CoordAtt)
        ]:
            setattr(modules, module_name, module_class)
            setattr(tasks, module_name, module_class)
        
        print("✅ 自定义注意力模块已成功注册")
        return True
    except Exception as e:
        print(f"⚠️ 模块注册失败: {e}")
        return False




class VarifocalDetectionLoss(v8DetectionLoss):
    """Varifocal-version of YOLOv8 detection loss."""

    def __init__(self, model, alpha=0.75, gamma=2.0):
        super().__init__(model)
        self.alpha = alpha
        self.gamma = gamma
        self.varifocal = VarifocalLoss(gamma=gamma, alpha=alpha)
        print(f"[INFO] VarifocalDetectionLoss enabled (alpha={alpha}, gamma={gamma})")

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores = target_scores.to(dtype)
        target_scores_sum = max(target_scores.sum(), 1)
        cls_labels = (target_scores > 0).float()
        vfl_scores = torch.zeros_like(target_scores, dtype=dtype)

        target_bboxes_grid = target_bboxes / stride_tensor
        if fg_mask.sum():
            ious = bbox_iou(pred_bboxes[fg_mask], target_bboxes_grid[fg_mask], xywh=False, CIoU=True).clamp_(0)
            vfl_scores[fg_mask] = cls_labels[fg_mask] * ious.unsqueeze(-1)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes_grid,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[1] = self.varifocal(pred_scores, vfl_scores, cls_labels) / target_scores_sum

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()

def train_model(model_type, epochs, batch_size, patience, init_weights=None, 
                imgsz=640, optimizer='SGD', lr0=0.01, close_mosaic=10):
    """
    训练指定类型的模型
    
    Args:
        model_type: 模型类型（见MODEL_CONFIGS）
        epochs: 训练轮数
        batch_size: 批次大小
        patience: 早停耐心值
        init_weights: 初始化权重路径（None则从头训练）
        imgsz: 输入图像尺寸
        optimizer: 优化器类型
        lr0: 初始学习率
        close_mosaic: 最后几轮关闭mosaic增强
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"未知模型类型: {model_type}. 可选: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    print(f"\n{'='*80}")
    print(f"🚀 开始训练: {config['description']}")
    print(f"📋 理论依据: {config['theory']}")
    print(f"📈 预期收益: {config['expected_gain']}")
    print(f"💾 显存需求: {config['vram']}")
    print(f"⚡ 训练速度: {config['speed']}")
    print(f"{'='*80}\n")
    
    # 注册自定义模块（重试机制）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            register_custom_modules()
            
            # 创建模型
            if config['yaml'] is None:
                # baseline使用预训练权重
                base_model = config.get('base_model', 'yolov8s.pt')
                model = YOLO(base_model)
                print(f"✅ 加载{base_model}预训练模型")
            else:
                # 自定义模型
                if init_weights:
                    # 热启动：从预训练权重加载
                    print(f"🔥 热启动模式: 从 {init_weights} 初始化")
                    model = YOLO(config['yaml'])
                    model.load(init_weights)
                    print(f"✅ 成功加载预训练权重")
                else:
                    # 从头训练
                    print(f"🆕 从头训练模式")
                    model = YOLO(config['yaml'])
            
            if config.get('use_varifocal'):
                alpha = config.get('varifocal_alpha', 0.75)
                gamma = config.get('varifocal_gamma', 2.0)
                model.model.loss_function = VarifocalDetectionLoss(model.model, alpha=alpha, gamma=gamma)

            break  # 成功则跳出重试循环
            
        except KeyError as e:
            if attempt < max_retries - 1:
                print(f"⚠️ 第{attempt+1}次尝试失败，正在重试...")
                time.sleep(1)
                register_custom_modules()
            else:
                raise RuntimeError(f"❌ 模型加载失败（尝试{max_retries}次）: {e}")
    
    # 检查是否需要禁用数据增强（用于baseline-800模拟）
    disable_augment = config.get('disable_augment', False)
    
    # 训练参数
    train_args = {
        'data': 'data.yaml',
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': patience,
        'device': 0,
        'workers': 8,
        'project': 'runs/train',
        'name': config['name'],
        'exist_ok': True,
        'pretrained': False,  # 我们手动控制权重加载
        'optimizer': optimizer,
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': True,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': close_mosaic,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': lr0,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        # 根据配置决定是否使用高级数据增强
        'auto_augment': None if disable_augment else 'randaugment',
        'erasing': 0.0 if disable_augment else 0.4,
        'crop_fraction': 1.0,
        'save': True,
        'save_period': -1,
        'cache': True,  # 缓存数据集，加速训练
        'plots': True,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
    }
    
    print("\n📊 训练参数:")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 输入尺寸: {imgsz}x{imgsz}")
    print(f"  - 优化器: {optimizer}")
    print(f"  - 学习率: {lr0} -> {lr0*0.01} (cosine)")
    print(f"  - 早停耐心: {patience}")
    print(f"  - 损失权重: box={train_args['box']}, cls={train_args['cls']}, dfl={train_args['dfl']}")
    if disable_augment:
        print(f"  - 数据增强: ⚠️ 禁用高级增强（auto_augment=None, erasing=0.0）")
        print(f"  - 增强模式: 仅基础增强（hsv/translate/scale/flip/mosaic）")
    else:
        print(f"  - 数据增强: 完整增强（hsv/translate/scale/flip/mosaic/randaugment/erasing）")
    print(f"  - Close Mosaic: 最后{close_mosaic}轮")
    if init_weights:
        print(f"  - 初始权重: {init_weights}")
    print()
    
    # 开始训练
    start_time = time.time()
    results = model.train(**train_args)
    train_time = time.time() - start_time
    
    # 保存结果
    save_dir = Path(f'runs/train/{config["name"]}')
    
    # 查找最佳模型
    best_model = save_dir / 'weights' / 'best.pt'
    last_model = save_dir / 'weights' / 'last.pt'
    
    if best_model.exists():
        print(f"\n✅ 训练完成！最佳模型: {best_model}")
        
        # 使用最佳模型进行验证
        model_best = YOLO(str(best_model))
        val_results = model_best.val(data='data.yaml', batch=batch_size, imgsz=imgsz)
        
        # 提取指标
        metrics = {
            'model_type': model_type,
            'description': config['description'],
            'theory': config['theory'],
            'expected_gain': config['expected_gain'],
            'train_time_hours': train_time / 3600,
            'epochs_trained': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'metrics/precision(B)': float(val_results.box.p[0]) if hasattr(val_results.box, 'p') else 0.0,
            'metrics/recall(B)': float(val_results.box.r[0]) if hasattr(val_results.box, 'r') else 0.0,
            'metrics/mAP50(B)': float(val_results.box.map50),
            'metrics/mAP50-95(B)': float(val_results.box.map),
            'best_model_path': str(best_model),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # 保存JSON结果
        json_path = save_dir / 'results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 最终性能指标:")
        print(f"  - Precision: {metrics['metrics/precision(B)']:.4f}")
        print(f"  - Recall: {metrics['metrics/recall(B)']:.4f}")
        print(f"  - mAP50: {metrics['metrics/mAP50(B)']:.4f}")
        print(f"  - mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
        print(f"  - 训练时长: {metrics['train_time_hours']:.2f} 小时")
        print(f"\n💾 结果已保存:")
        print(f"  - 最佳模型: {best_model}")
        print(f"  - JSON结果: {json_path}")
        
        return metrics
    else:
        print(f"\n⚠️ 未找到最佳模型，请检查训练日志")
        return None


def compare_all_results():
    """
    对比所有训练结果
    生成性能对比表格
    """
    runs_dir = Path('runs/train')
    if not runs_dir.exists():
        print("❌ 未找到训练结果目录")
        return
    
    results = []
    for model_type, config in MODEL_CONFIGS.items():
        result_dir = runs_dir / config['name']
        json_path = result_dir / 'results.json'
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
    
    if not results:
        print("❌ 未找到任何训练结果")
        return
    
    # 按mAP50-95排序
    results.sort(key=lambda x: x['metrics/mAP50-95(B)'], reverse=True)
    
    print("\n" + "="*120)
    print("📊 所有模型性能对比（按mAP50-95排序）")
    print("="*120)
    print(f"{'排名':<6}{'模型':<25}{'mAP50-95':<12}{'mAP50':<12}{'Precision':<12}{'Recall':<12}{'训练时长(h)':<15}")
    print("-"*120)
    
    baseline_map = None
    for idx, r in enumerate(results, 1):
        if r['model_type'] == 'baseline':
            baseline_map = r['metrics/mAP50-95(B)']
        
        gain_str = ""
        if baseline_map and r['model_type'] != 'baseline':
            gain = (r['metrics/mAP50-95(B)'] - baseline_map) / baseline_map * 100
            gain_str = f"(+{gain:.1f}%)"
        
        print(f"{idx:<6}"
              f"{r['description'][:24]:<25}"
              f"{r['metrics/mAP50-95(B)']:.4f}{gain_str:<6}"
              f"{r['metrics/mAP50(B)']:.4f}      "
              f"{r['metrics/precision(B)']:.4f}      "
              f"{r['metrics/recall(B)']:.4f}      "
              f"{r['train_time_hours']:.2f}")
    
    print("="*120)
    
    # 保存对比结果
    comparison_path = runs_dir / 'model_comparison.json'
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 对比结果已保存至: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 高级改进模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用模型类型及说明:
  基线模型:
  baseline      - YOLOv8s基线（完整数据增强）
  baseline-800  - YOLOv8m基线（模拟8.0.0版本，禁用auto_augment和erasing）
  
  注意力机制:
  cbam          - CBAM (通道+空间注意力，全面增强)
  eca           - ECA (轻量级通道注意力)
  se            - SE (SO-YOLOv8论文采用)
  coordatt      - CoordAtt (坐标注意力，空间定位精准)
  
  P2检测头:
  p2-lite       - P2轻量版（显存友好）
  p2            - P2完整版（最强小目标检测）
  
  组合方案:
  p2-cbam       - P2 + CBAM（双重增强）
  p2-eca        - P2 + ECA（性能速度平衡）
  p2-se         - P2 + SE（复现SO-YOLOv8）
  p2-se-varifocal - P2 + SE + VarifocalLoss (cls loss only)
  p2-coordatt   - P2 + CoordAtt（空间定位最强）

训练建议:
  1. 先训练baseline建立基准
  2. 测试单一改进（注意力或P2）
  3. 验证组合方案（P2+注意力）
  4. 使用--compare查看所有结果对比
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (默认: 200)')
    parser.add_argument('--batch', type=int, default=12,
                        help='批次大小 (默认: 12)')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值 (默认: 50)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--init-weights', type=str, default=None,
                        help='初始化权重路径（热启动，默认: yolov8s.pt for custom models）')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型 (默认: SGD)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率 (默认: 0.01)')
    parser.add_argument('--close-mosaic', type=int, default=10,
                        help='最后几轮关闭mosaic增强 (默认: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='⚡ 快速测试模式：epochs=50, patience=10, batch=16')
    parser.add_argument('--compare', action='store_true',
                        help='对比所有训练结果（不训练）')
    parser.add_argument('--show-config', action='store_true',
                        help='显示指定模型的详细配置')
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick:
        print("\n⚡ 快速测试模式已启用")
        args.epochs = 50
        args.patience = 10
        args.batch = 16
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Patience: {args.patience}")
        print(f"   - Batch: {args.batch}")
        print("   - 适合快速验证改进方案是否有效\n")
    
    # 仅显示配置
    if args.show_config:
        config = MODEL_CONFIGS[args.model]
        print(f"\n{'='*80}")
        print(f"📋 {config['description']}")
        print(f"{'='*80}")
        print(f"理论依据: {config['theory']}")
        print(f"预期收益: {config['expected_gain']}")
        print(f"显存需求: {config['vram']}")
        print(f"训练速度: {config['speed']}")
        if config['yaml']:
            print(f"配置文件: {config['yaml']}")
        print(f"{'='*80}\n")
        return
    
    # 仅对比结果
    if args.compare:
        compare_all_results()
        return
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到CUDA，将使用CPU训练（速度会很慢）")
    else:
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 为自定义模型自动设置热启动
    config = MODEL_CONFIGS[args.model]
    init_weights = args.init_weights
    if init_weights is None and config['yaml'] is not None:
        # 自定义模型默认热启动
        init_weights = 'yolov8s.pt'
        print(f"ℹ️ 自定义模型将从 yolov8s.pt 热启动（可通过--init-weights修改）")
    
    # 开始训练
    result = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        init_weights=init_weights,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        close_mosaic=args.close_mosaic
    )
    
    if result:
        print(f"\n{'='*80}")
        print(f"🎉 训练成功完成！")
        print(f"{'='*80}")
        print(f"\n💡 提示:")
        print(f"  1. 查看训练曲线: runs/train/{config['name']}/results.png")
        print(f"  2. 对比所有结果: python {sys.argv[0]} --compare")
        print(f"  3. 使用最佳模型推理: python detect_track.py --model {result['best_model_path']}")
        print()


if __name__ == '__main__':
    main()
