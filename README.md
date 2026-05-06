# Spectral Calculation - 光谱重建算法项目

**版本**: v1.0  
**作者**: UNS-JeromeWei  
**日期**: 2026-05-06

---

## 项目概述

本项目是一个光谱重建算法的实现，主要用于计算光谱成像系统中的光谱重建问题。核心功能是通过响应矩阵和透射曲线，利用优化算法重建原始光谱分布。

---

## 项目结构

```
spectral calculation/
│
├── README.md                           # 项目说明文档
├── spectral_cal/                       # 核心算法目录
│   ├── rebuild_curves_cwl_fwhm_loop.py           # 主算法实现文件
│   ├── rebuild_curves_cwl_fwhm_loop_20260503.py  # 算法版本备份
│   ├── rebuild_curves_cwl_fwhm_loop_analysis.md  # 算法详细解析文档
│   ├── tmp/                                      # 临时测试文件目录
│   │   ├── rebuild_curves_2_peak_new_uc450_260417.py
│   │   ├── rebuild_curves_2_peak_400_1000nm.py
│   │   └── rebuild_curves_2_peak_400_700nm.py
│   ├── data analysis/                            # 数据分析结果目录
│   │   ├── 20260503_232523-...                   # 分析结果示例
│   │   ├── 20260503_232153-...
│   │   ├── 20260503_170351-...
│   │   ├── 20260503_164126-...
│   │   └── ...
│   ├── .kilo/                                    # Kilo配置目录
│   │   └── agent-manager.json
│   └── .vscode/                                  # VSCode配置
│       └── settings.json
│
├── data analysis/                       # 全局数据分析结果目录
│   ├── 20260430_112954-...              # UDE450 MEMS分析结果
│   ├── 20260416143643-...               # U500 MEMS Drift分析结果
│   └── ...
│
└── .kilo/                               # Kilo项目配置
    ├── command/
    ├── agent/
    └ kilo.json
    └ AGENTS.md
```

---

## 核心功能

### 1. 光谱重建算法

**文件**: `spectral_cal/rebuild_curves_cwl_fwhm_loop.py`

**主要功能**:
- 响应矩阵加载与处理
- 光谱重建计算（最小二乘法、正则化优化）
- FWHM（半高全宽）计算与分析
- CWL（中心波长）循环测试
- 结果可视化与保存

**核心算法**:
- 非负最小二乘法 (NNLS)
- 增广正则化最小二乘法
- PyTorch优化器方法
- 二阶差分平滑约束

### 2. 数学模型

#### 线性模型
```
y = A · x
```
其中:
- `y`: 测量的透射曲线（通道响应）
- `A`: 响应矩阵（每个通道对不同波长的响应）
- `x`: 待重建的原始光谱

#### 正则化优化
```
min_x ||Ax - y||² + λ||D₂x||²,  x ≥ 0
```
其中:
- `λ`: 正则化参数，控制平滑约束强度
- `D₂`: 二阶差分矩阵，用于平滑约束

---

## 主要函数说明

### 核心重建函数

#### `lst_with_aug_reg_cwl_fwhm_loop()`
**功能**: CWL和FWHM双重循环的光谱重建  
**参数**:
- `src_dir`: 数据目录路径
- `cwl_range`: CWL波长范围 (起始, 结束, 步长)
- `fwhm_range`: FWHM范围 (起始, 结束, 步长)
- `amplitude`: 高斯峰幅度
- `lam`: 正则化参数
- `save_results`: 是否保存结果

**返回**: 包含所有循环结果的字典列表

#### `lst_with_aug_reg_cwl_loop()`
**功能**: 仅CWL循环的光谱重建（FWHM固定）

#### `lst_with_aug_reg_fwhm_loop()`
**功能**: 仅FWHM循环的光谱重建（CWL固定）

### 辅助函数

#### `gaussian_beam(x, mu, sig)`
**功能**: 高斯光束模型  
**公式**: `G(x, μ, σ) = exp(-(x - μ)² / (2σ²))`

#### `find_fwhm_normal(in_x, in_y)`
**功能**: 计算光谱峰的FWHM  
**方法**: 三次样条插值 + 半高位置查找

#### `build_D2(L)`
**功能**: 构建二阶差分矩阵  
**用途**: 用于正则化平滑约束

#### `load_matrix_from_file(src_dir)`
**功能**: 从文件加载波长和透过率数据  
**处理**: 波长范围筛选 (350-950nm)，透过率归一化

---

## 数据分析结果

### 结果文件说明

每个分析结果目录包含:

1. **spectrum_results.csv**: CSV格式的汇总结果
   - Input_CWL_nm: 输入中心波长
   - Output_CWL_nm: 输出中心波长
   - Input_FWHM_nm: 输入FWHM
   - Output_FWHM_nm: 输出FWHM
   - MSE: 重建误差

2. **rebuild_summary.npy**: Numpy格式的完整结果数据

3. **response_matrix.png**: 响应矩阵热图

4. **FWHM={value}/**: 不同FWHM值的重建对比图
   - cwl_{value}_fwhm_{value}.png: 光谱对比图

---

## 使用示例

### 示例1: CWL和FWHM双重循环
```python
results = lst_with_aug_reg_cwl_fwhm_loop(
    data_dir,
    cwl_range=(440, 800, 40),   # CWL: 440nm到800nm，步长40nm
    fwhm_range=(10, 30, 5),      # FWHM: 10nm到30nm，步长5nm
    amplitude=0.6,
    lam=0.01,
    save_results=True
)
```

### 示例2: 仅CWL循环
```python
results = lst_with_aug_reg_cwl_loop(
    data_dir,
    cwl_range=(440, 650, 10),   # CWL范围
    fwhm=8,                     # 固定FWHM: 8nm
    save_results=True
)
```

### 示例3: 仅FWHM循环
```python
results = lst_with_aug_reg_fwhm_loop(
    data_dir,
    cwl=550,                    # 固定CWL: 550nm
    fwhm_range=(10, 30, 5),
    save_results=True
)
```

---

## 依赖库

```python
numpy              # 数值计算
torch              # PyTorch深度学习框架
scipy              # 科学计算（插值、优化、滤波）
matplotlib         # 可视化
pandas             # 数据处理
spectral           # 光谱数据处理
```

---

## 物理背景

### 光谱成像系统

光谱成像系统通过滤光片阵列将不同波长的光分配到不同通道:
- 每个通道对特定波长范围有响应
- 响应矩阵描述各通道的波长响应特性

### 重建原理

**已知**:
- 响应矩阵 `A`（系统特性）
- 测量值 `y`（各通道输出）

**求解**:
- 原始光谱 `x`（入射光光谱分布）

### 应用场景

- 计算光谱成像
- 光谱仪校准
- 薄膜设计验证
- MEMS可调滤光片评估

---

## 版本历史

### v1.0 (2026-05-06)
- 初始版本发布
- 实现核心光谱重建算法
- 支持CWL和FWHM双重循环测试
- 完善结果保存与可视化功能
- 添加详细算法解析文档

---

## GitHub仓库

**仓库地址**: https://github.com/UNS-JeromeWei/spectral-calculation.git

---

## 许可证

本项目仅供研究和学习使用。

---

## 联系方式

**作者**: UNS-JeromeWei  
**GitHub**: https://github.com/UNS-JeromeWei