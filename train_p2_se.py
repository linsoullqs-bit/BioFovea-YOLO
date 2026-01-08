#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8s-P2-SE 训练脚本
复现SO-YOLOv8论文方案：P2检测头 + SE注意力机制

改进点：
1. P2检测头（stride=4）- 提升小气泡检测能力
2. SE注意力 - 通道注意力增强特征表达
3. SO-YOLOv8论文验证有效的组合方案

作者: AI Assistant
日期: 2025-10-31
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加models目录到Python路径
models_dir = Path(__file__).parent / 'models'
sys.path.insert(0, str(models_dir))

from ultralytics import YOLO
import torch


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


def train_p2_se(epochs=200, batch_size=12, patience=50, imgsz=640, 
                init_weights: Optional[str]='yolov8s.pt', data='data.yaml', 
                optimizer='SGD', lr0=0.01, close_mosaic=10):
    """
    训练 YOLOv8s-P2-SE 模型
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        patience: 早停耐心值
        imgsz: 输入图像尺寸
        init_weights: 初始化权重路径（默认从yolov8s.pt热启动）
        data: 数据配置文件路径
        optimizer: 优化器类型
        lr0: 初始学习率
        close_mosaic: 最后几轮关闭mosaic增强
    """
    print(f"\n{'='*80}")
    print(f" 开始训练: YOLOv8s + P2 + SE (复现SO-YOLOv8)")
    print(f" 理论依据: P2检测头提升小目标检测，SE注意力增强特征表达")
    print(f" 预期收益: +2.0~4.0% mAP50-95")
    print(f" 显存需求: 约10-11GB")
    print(f" 训练速度: 0.7x基准")
    print(f"{'='*80}\n")
    
    # 注册自定义模块
    max_retries = 3
    for attempt in range(max_retries):
        try:
            register_custom_modules()
            
            # 创建模型
            model = YOLO('models/yolov8s-p2-se.yaml')
            
            # 热启动：从预训练权重加载
            if init_weights and Path(init_weights).exists():
                print(f"🔥 热启动模式: 从 {init_weights} 初始化")
                model.load(init_weights)
                print(f" 成功加载预训练权重")
            else:
                print(f" 从头训练模式")
            
            break  # 成功则跳出重试循环
            
        except KeyError as e:
            if attempt < max_retries - 1:
                print(f" 第{attempt+1}次尝试失败，正在重试...")
                time.sleep(1)
                register_custom_modules()
            else:
                raise RuntimeError(f" 模型加载失败（尝试{max_retries}次）: {e}")
    
    # 训练参数
    train_args = {
        'data': data,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': patience,
        'device': 0,
        'workers': 8,
        'project': 'runs/train',
        'name': 'bubble_yolov8s_p2_se_retrain',
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
        'auto_augment': 'randaugment',
        'erasing': 0.4,
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
    
    print("\n 训练参数:")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 输入尺寸: {imgsz}x{imgsz}")
    print(f"  - 优化器: {optimizer}")
    print(f"  - 学习率: {lr0} -> {lr0*0.01} (cosine)")
    print(f"  - 早停耐心: {patience}")
    print(f"  - 损失权重: box={train_args['box']}, cls={train_args['cls']}, dfl={train_args['dfl']}")
    print(f"  - 数据增强: 完整增强（hsv/translate/scale/flip/mosaic/randaugment/erasing）")
    print(f"  - Close Mosaic: 最后{close_mosaic}轮")
    if init_weights:
        print(f"  - 初始权重: {init_weights}")
    print()
    
    # 开始训练
    start_time = time.time()
    model.train(**train_args)
    train_time = time.time() - start_time
    
    # 保存结果
    save_dir = Path('runs/train/bubble_yolov8s_p2_se')
    
    # 查找最佳模型
    best_model = save_dir / 'weights' / 'best.pt'
    
    if best_model.exists():
        print(f"\n 训练完成！最佳模型: {best_model}")
        
        # 使用最佳模型进行验证
        model_best = YOLO(str(best_model))
        val_results = model_best.val(data=data, batch=batch_size, imgsz=imgsz)
        
        # 提取指标
        metrics = {
            'model_type': 'p2-se',
            'description': 'YOLOv8s + P2 + SE (复现SO-YOLOv8)',
            'theory': 'P2检测头提升小目标检测，SE注意力增强特征表达',
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
        
        print(f"\n 最终性能指标:")
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
        print(f"\n 未找到最佳模型，请检查训练日志")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8s-P2-SE 训练脚本（复现SO-YOLOv8）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数训练
  python train_p2_se.py

  # 自定义训练参数
  python train_p2_se.py --epochs 300 --batch 16 --patience 50

  # 从头训练（不使用预训练权重）
  python train_p2_se.py --init-weights None

  # 快速测试模式
  python train_p2_se.py --quick
        """
    )
    
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (默认: 200)')
    parser.add_argument('--batch', type=int, default=12,
                        help='批次大小 (默认: 12)')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值 (默认: 50)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--init-weights', type=str, default='yolov8s.pt',
                        help='初始化权重路径（默认: yolov8s.pt，设为None则从头训练）')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='数据配置文件路径 (默认: data.yaml)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'AdamW'],
                        help='优化器类型 (默认: SGD)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率 (默认: 0.01)')
    parser.add_argument('--close-mosaic', type=int, default=10,
                        help='最后几轮关闭mosaic增强 (默认: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='⚡ 快速测试模式：epochs=50, patience=10, batch=16')
    
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
    
    # 处理init_weights参数
    init_weights = None if args.init_weights.lower() == 'none' else args.init_weights
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print(" 警告: 未检测到CUDA，将使用CPU训练（速度会很慢）")
    else:
        print(f" 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 开始训练
    result = train_p2_se(
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        imgsz=args.imgsz,
        init_weights=init_weights,
        data=args.data,
        optimizer=args.optimizer,
        lr0=args.lr0,
        close_mosaic=args.close_mosaic
    )
    
    if result:
        print(f"\n{'='*80}")
        print(f" 训练成功完成！")
        print(f"{'='*80}")
        print(f"\n 提示:")
        print(f"  1. 查看训练曲线: runs/train/bubble_yolov8s_p2_se/results.png")
        print(f"  2. 使用最佳模型推理: python detect_track.py --model {result['best_model_path']}")
        print()


if __name__ == '__main__':
    main()

