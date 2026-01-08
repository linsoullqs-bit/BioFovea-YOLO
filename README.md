# 气泡检测与跟踪 - YOLOv8改进研究

> 基于论文 "Bubble feature extraction in subcooled flow boiling using AI-based object detection and tracking techniques" 的代码复现与改进

---

## 🎯 项目简介

本项目是气泡检测论文的完整复现，并在此基础上进行了系统化的YOLOv8模型改进研究。通过引入P2检测头和多种注意力机制，在小目标检测性能上取得了显著提升。

### 📊 主要成果

| 模型 | mAP50-95 | vs 论文基线 | 改进策略 |
|------|----------|-------------|---------|
| **论文基线** | 41.8% | - | YOLOv8x |
| **YOLOv8s-P2-SE** ⭐ | **50.78%** | **+8.98%** | P2检测头 + SE注意力 |
| **YOLOv8s-P2-CoordAtt** | 50.18% | +8.38% | P2检测头 + 坐标注意力 |
| **YOLOv8s-P2-ECA** | 48.30% | +6.50% | P2检测头 + ECA注意力 |

**关键发现**：
- ✅ **所有改进模型均超越论文基线**
- ✅ 最高相对提升：**+21.5%**
- ✅ P2检测头是核心改进（平均提升 +3.02%）
- ✅ 注意力机制进一步提升性能（+1~3%）

---

## 📁 项目结构

```
.
├── date/                          # 数据集（训练/验证集）
│   ├── images/
│   │   ├── train/                 # 495张训练图像
│   │   └── val/                   # 54张验证图像
│   └── labels/
│       ├── train/                 # 训练标签
│       └── val/                   # 验证标签
│
├── models/                        # 模型配置文件
│   ├── cbam.py                    # 注意力机制实现（SE/ECA/CBAM/CoordAtt）
│   ├── yolov8s-p2-*.yaml          # YOLOv8s改进配置
│   └── yolov8m-p2-*.yaml          # YOLOv8m改进配置
│
├── paper/                         # 论文资料
│   ├── baseline/                  # 原论文
│   └── yolov8小目标检测/           # 参考文献（7篇）
│
├── train_all.py                   # 批量训练脚本 ⭐
├── train_yolov8s.py              # YOLOv8s基线训练
├── train_yolov8m.py              # YOLOv8m基线训练
├── detect_track.py               # 检测与追踪
├── extract_parameters.py         # 热液参数提取
├── test_environment.py           # 环境验证
│
├── 快速开始.md                    # 快速开始指南 📘
├── 模型改进指南.md                 # 改进方案详细说明 📘
├── 实验结果分析.md                 # 实验结果分析报告 📘
├── 进一步优化方向指南.md           # 优化方向建议 📘
└── README.md                     # 本文件
```

---

## 🚀 快速开始

### 1. 环境配置

使用项目提供的虚拟环境：

```bash
# Windows PowerShell
.\yolov8\Scripts\activate

# 验证环境
python test_environment.py
```

### 2. 训练模型

#### 方案A：批量训练所有改进模型（推荐）

```bash
# 训练所有模型
python train_all.py --model all --epochs 200

# 训练特定模型
python train_all.py --model p2-se --epochs 200
python train_all.py --model p2-eca --epochs 200

# 对比结果
python train_all.py --compare
```

#### 方案B：训练基线模型

```bash
# YOLOv8m基线
python train_yolov8m.py

# YOLOv8s基线（速度快，适合快速验证）
python train_yolov8s.py
```

### 3. 检测与追踪

```bash
# 基础用法
python detect_track.py --source video.mp4

# 使用改进模型
python detect_track.py \
    --source video.mp4 \
    --model runs/train/yolov8s_p2_se/weights/best.pt \
    --conf 0.5
```

### 4. 提取热液参数

```bash
python extract_parameters.py \
    --csv runs/detect_track/bubble_tracking_results.csv \
    --fps 5000 \
    --pixel-to-meter 1e-4
```

---

## 📚 文档导航

| 文档 | 内容 | 适合人群 |
|------|------|---------|
| [快速开始.md](快速开始.md) | 详细的使用教程 | 所有用户 |
| [模型改进指南.md](模型改进指南.md) | 改进方案设计、训练方法、论文写作 | 研究人员 |
| [实验结果分析.md](实验结果分析.md) | 8个模型的详细对比分析 | 研究人员 |
| [进一步优化方向指南.md](进一步优化方向指南.md) | 未来优化方向 | 进阶用户 |

---

## 🔬 改进方案说明

### 核心改进

1. **P2检测头**：添加stride=4的检测层，专门针对小目标（<32×32像素）
2. **注意力机制**：SE/ECA/CBAM/CoordAtt，增强特征表达能力
3. **组合策略**：P2 + 注意力，实现最大化性能提升

### 理论依据

基于7篇顶会/顶刊论文的研究：
- SO-YOLOv8 (SE注意力)
- MAE-YOLOv8 (P2检测头)
- SOD-YOLO, IMCMD-YOLOv8, SMA-YOLOv8等

所有方法均经过学术界验证，适合作为论文创新点。

### 实验验证

已完成8个模型配置的完整训练和对比：
- ✅ 所有模型均超越论文基线
- ✅ 最佳模型：YOLOv8s-P2-SE（mAP50-95: 50.78%）
- ✅ 详细结果见 [实验结果分析.md](实验结果分析.md)

---

## 🎯 主要功能

### 1. 气泡检测
- 基于YOLOv8的目标检测
- 多种改进方案可选
- 支持YOLOv8s/m/l/x等不同尺寸

### 2. 气泡追踪
- Strongsort追踪算法（论文最优）
- 支持多种追踪器：BoT-SORT、ByteTrack等
- 输出完整的轨迹信息

### 3. 参数提取
- 长宽比（Aspect Ratio）
- Sauter平均直径（SMD）
- 脱离直径（Departure Diameter）
- 生长时间（Growth Time）
- 气泡寿命（Bubble Lifetime）

---

## 📊 性能对比

### 检测性能（mAP50-95）

| 模型 | mAP50-95 | Precision | Recall | 训练时间 |
|------|----------|-----------|--------|---------|
| 论文基线 (YOLOv8x) | 41.8% | - | 80.6% | - |
| **YOLOv8s-P2-SE** | **50.78%** ⭐ | 86.04% | 82.15% | 0.98h |
| YOLOv8s-P2-CoordAtt | 50.18% | 83.53% | **82.17%** | 1.17h |
| YOLOv8s-P2-ECA | 48.30% | 83.38% | 80.43% | 0.66h |
| YOLOv8m | 48.36% | **84.54%** | 75.31% | 5.73h |
| YOLOv8s | 47.24% | 83.98% | 74.88% | 0.59h |

### 关键发现

1. **P2检测头效果显著**：P2系列模型平均mAP50-95达到49.75%，比非P2系列高3.02%
2. **SE注意力最优**：在P2基础上，SE注意力达到最高性能（50.78%）
3. **性价比最高**：YOLOv8s-P2-ECA，训练时间0.66h，性能48.30%

---

## 🛠️ 使用场景

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **生产部署（精度优先）** | YOLOv8s-P2-SE | 最高精度（50.78%） |
| **召回率优先（不容漏检）** | YOLOv8s-P2-CoordAtt | 最高召回率（82.17%） |
| **快速验证** | YOLOv8s-P2-ECA | 训练快（0.66h），性能好 |
| **资源受限** | YOLOv8s | 训练最快（0.59h） |
| **论文对比** | YOLOv8s-P2-SE | 最显著提升（+8.98%） |

---

## 📖 论文引用

如果使用本项目代码或方法，请引用原论文：

```bibtex
@article{zhou2024bubble,
  title={Bubble feature extraction in subcooled flow boiling using AI-based object detection and tracking techniques},
  author={Zhou, Wen and Miwa, Shuichiro and Tsujimura, Ryoma and Nguyen, Thanh-Binh and Okawa, Tomio and Okamoto, Koji},
  journal={International Journal of Heat and Mass Transfer},
  year={2024}
}
```

改进方案基于以下研究（参见 `paper/yolov8小目标检测/` 目录）：
- SO-YOLOv8 (SE attention)
- MAE-YOLOv8 (P2 detection head)
- SOD-YOLO, IMCMD-YOLOv8, 等

---

## 🤝 贡献

欢迎贡献改进建议！包括但不限于：
- 新的改进方案
- 性能优化
- 文档完善
- Bug修复

---

## 📝 更新日志

### v2.0 (2025-11-09)
- ✅ 完成8个模型的完整训练和对比
- ✅ 所有模型均超越论文基线
- ✅ 最佳模型：YOLOv8s-P2-SE（mAP50-95: 50.78%）
- ✅ 整理优化所有文档

### v1.0 (2024)
- ✅ 论文基线复现
- ✅ 基础训练和检测功能
- ✅ 热液参数提取

---

## 📞 技术支持

遇到问题时：
1. 查看相关文档（快速开始.md、模型改进指南.md等）
2. 运行 `python test_environment.py` 检查环境
3. 查看训练日志和错误信息

---

## 📄 许可证

本项目代码基于论文复现，仅供学术研究使用。

---

## 🌟 特色

✅ 完整复现论文方法  
✅ 系统化的改进方案（基于7篇论文）  
✅ 所有模型均超越论文基线  
✅ 详细的中文文档  
✅ 开箱即用的脚本  
✅ 完整的实验对比分析  
✅ 论文写作指南  

---

**最后更新**: 2025-11-09  
**项目状态**: ✅ 稳定版本  
**推荐用于**: 学术研究、论文复现、模型改进研究

---

<p align="center">
  <strong>🎓 适合作为目标检测研究的参考项目</strong><br>
  <strong>🚀 已验证的改进方案，可直接用于论文</strong>
</p>
