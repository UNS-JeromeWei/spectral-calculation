# rebuild_curves_cwl_fwhm_loop.py 代码功能详细解析

## 一、文件概述

本文件是一个光谱重建算法的实现，主要用于计算光谱成像系统中的光谱重建问题。核心功能是通过响应矩阵和透射曲线，利用优化算法重建原始光谱分布。

---

## 二、依赖库

```python
import numpy as np              # 数值计算
import torch                    # PyTorch深度学习框架
import torch.optim as optim     # PyTorch优化器
import matplotlib.pyplot as plt # 可视化
from scipy.interpolate import CubicSpline, interp1d  # 插值
from scipy.optimize import lsq_linear                 # 线性最小二乘
from scipy.signal import savgol_filter               # Savitzky-Golay滤波
from scipy.ndimage import gaussian_filter1d          # 高斯滤波
import pandas as pd             # 数据处理
```

---

## 三、核心数学概念

### 3.1 高斯光束模型 (Gaussian Beam)

**函数**: `gaussian_beam(x, mu, sig)` (第436-437行)

**数学表达式**:
$$G(x, \mu, \sigma) = \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**参数说明**:
- $x$: 波长位置（索引或波长值）
- $\mu$: 中心波长位置（峰值位置）
- $\sigma$: 标准差，控制峰宽度

**物理意义**:
模拟理想单色光的光谱分布，用于生成测试光谱。高斯分布是光谱仪响应的典型模型。

### 3.2 FWHM与标准差的关系

**公式**:
$$\text{FWHM} = 2.355 \times \sigma$$

**推导**:
半高宽定义为峰值高度一半处的宽度。对于高斯函数：
$$G(\mu \pm \frac{\text{FWHM}}{2}) = \frac{1}{2} G(\mu)$$

解得：
$$\exp\left(-\frac{(\text{FWHM}/2)^2}{2\sigma^2}\right) = \frac{1}{2}$$

$$\frac{\text{FWHM}^2}{8\sigma^2} = \ln(2)$$

$$\text{FWHM} = 2\sqrt{2\ln(2)}\sigma \approx 2.355\sigma$$

---

### 3.3 FWHM计算函数

**函数**: `find_fwhm_normal(in_x, in_y)` (第283-308行)

**算法流程**:
1. 使用三次样条插值加密数据点
2. 找到峰值位置 `peak_id = argmax(y)`
3. 计算半峰值 `half_peak = peak_value / 2`
4. 在峰值左侧和右侧分别找到半高位置
5. 计算FWHM = `x[right] - x[left]`

**数学意义**:
通过插值提高精度，准确测量光谱峰的实际宽度。

---

### 3.4 三次样条插值 (Cubic Spline Interpolation)

**函数**: 使用 `scipy.interpolate.CubicSpline`

**数学原理**:
三次样条是在每两个数据点之间用一个三次多项式连接，满足：
1. 在节点处函数值连续
2. 在节点处一阶导数连续
3. 在节点处二阶导数连续
4. 边界条件（natural: 二阶导数为0）

**表达式**:
$$S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$$

**作用**:
将稀疏的光谱数据点加密为密集曲线，提高FWHM计算的精度。

---

### 3.5 平滑度损失函数

#### 一阶平滑度损失

**函数**: `smoothness_loss_1d(y)` (第311-313行)

**数学表达式**:
$$L_1 = \frac{1}{n-1}\sum_{i=1}^{n-1}(y_{i+1} - y_i)^2$$

**物理意义**:
惩罚相邻点之间的突变，使曲线更平滑。这是对一阶导数的L2约束。

#### 二阶平滑度损失

**函数**: `smoothness_loss_2nd(y)` (第316-318行)

**数学表达式**:
$$L_2 = \frac{1}{n-2}\sum_{i=1}^{n-2}(y_{i+2} - 2y_{i+1} + y_i)^2$$

**物理意义**:
这是二阶中心差分的L2范数，近似二阶导数。惩罚曲线的曲率变化，使曲线更加"平直"。

---

### 3.6 二阶差分矩阵 (Second Difference Matrix)

**函数**: `build_D2(L)` (第783-790行)

**数学形式**:
$$D_2 = \begin{bmatrix}
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & & \ddots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & 1 & -2 & 1
\end{bmatrix}_{(L-2) \times L}$$

**作用**:
对向量 $x$ 应用此矩阵：$D_2 x$ 得到二阶差分向量。

**在正则化中的应用**:
$$\|D_2 x\|^2 = \sum_{i}(x_{i+2} - 2x_{i+1} + x_i)^2$$

这作为Tikhonov正则化的平滑约束。

---

## 四、核心算法解析

### 4.1 线性最小二乘重建

**问题模型**:
$$y = A \cdot x$$

其中：
- $y$: 测量的透射曲线（通道响应）
- $A$: 响应矩阵（每个通道对不同波长的响应）
- $x$: 待重建的原始光谱

**目标**: 已知 $y$ 和 $A$，求解 $x$

**数学表达**:
$$\min_x \|Ax - y\|^2$$

### 4.2 非负最小二乘 (NNLS)

**函数**: `lst_no_reg()` (第713-742行)

**使用**: `scipy.optimize.lsq_linear`

**数学模型**:
$$\min_x \|Ax - y\|^2, \quad x \geq 0$$

**物理约束**:
光谱强度不能为负值，这是物理上的合理约束。

---

### 4.3 增广正则化最小二乘

**函数**: `lst_with_aug_reg()` (第793-846行)

**数学模型**:
$$\min_x \left\|\begin{bmatrix} A \\ \sqrt{\lambda} D_2 \end{bmatrix} x - \begin{bmatrix} y \\ 0 \end{bmatrix}\right\|^2, \quad x \geq 0$$

展开为：
$$\min_x \|Ax - y\|^2 + \lambda \|D_2 x\|^2, \quad x \geq 0$$

**正则化项解释**:
- $\lambda$: 正则化参数，控制平滑约束强度
- $\|D_2 x\|^2$: 二阶差分的平方和，即平滑度度量

**物理意义**:
真实光谱通常是平滑的，不会有剧烈的跳变。正则化约束防止重建结果出现不物理的高频振荡。

**增广矩阵构造**:
```python
A_aug = np.vstack([new_coef_matrix, np.sqrt(lam) * D2])
y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0])])
```

---

### 4.4 PyTorch优化器方法

**函数**: `pytorch_optimizor(in_matrix, in_vector)` (第321-369行)

**目标函数**:
$$J(x) = \|Ax - C\|^2 + \lambda_1 \|x\|^2 + \lambda_2 \text{ReLU}(-x) + \lambda_3(L_1(x) + L_2(x))$$

**各项含义**:
1. $\|Ax - C\|^2$: 数据拟合项（主要目标）
2. $\lambda_1 \|x\|^2$: L2正则化（防止过拟合）
3. $\lambda_2 \text{ReLU}(-x)$: 非负惩罚（软约束）
4. $\lambda_3(L_1 + L_2)$: 平滑度约束

**优化算法**: Adam优化器

**参数配置**:
- `lambda_reg = 0.00001`: L2正则化系数
- `penalty_weight = 0.0001`: 非负惩罚系数
- `learning_rate = 0.01`: 学习率
- `num_iterations = 20000`: 迭代次数

---

## 五、辅助函数解析

### 5.1 区域划分函数

**函数**: `make_sum_regions()` (第36-46行)

**功能**: 将650个点划分为70个区域

**划分规则**:
- 第一个区域: [0, 56] (57个点)
- 中间68个区域: 每个区域8个点
- 最后一个区域: 剩余点到650

**用途**: 用于计算光谱的部分能量分布。

### 5.2 能量分布计算

**函数**: `get_part_sum_list(in_curve, part_list)` (第176-183行)

**数学表达**:
$$s_i = \frac{\sum_{j \in \text{region}_i} y_j}{\sum_{j} y_j}$$

**意义**: 计算每个区域的能量占总能量的比例。

### 5.3 响应矩阵加载

**函数**: `load_matrix_from_file(src_dir)` (第405-433行)

**功能**: 从文件加载波长和透过率数据

**数据处理**:
1. 读取每个文件的波长和透过率
2. 筛选波长范围 350-950nm
3. 透过率转换为0-1范围（除以100）
4. 返回波长列表和透过率矩阵

---

### 5.4 随机光谱生成函数

#### 三次样条随机曲线
**函数**: `cubic_spline_sim_1(wavelengths)` (第440-452行)

生成方法：在波长范围内选取8个固定点，赋予随机值，用三次样条插值连接。

#### 多峰高斯叠加
**函数**: `random_peak_sum_with_noise(wavelengths)` (第455-466行)

$$y = \sum_{k=1}^{5} A_k \exp\left(-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right) + \epsilon$$

#### 高斯平滑随机曲线
**函数**: `gaussian_smooth_curve(wavelengths)` (第469-473行)

#### 傅里叶曲线
**函数**: `fourier_curve(wavelengths)` (第476-485行)

$$y = \sum_{k=1}^{9}(a_k \sin(\frac{k}{100}x) + b_k \cos(\frac{k}{100}x)) + 0.8$$

#### Savitzky-Golay滤波曲线
**函数**: `savgol_filter_curve(wavelengths)` (第488-492行)

---

## 六、可视化函数

### 6.1 曲线绘制

**函数**: `draw_curves(x, ys, title)` (第49-76行)

特点：使用三次样条加密显示，多条曲线叠加。

### 6.2 2D矩阵热图

**函数**: `draw_map_of_2d_angle(data, axis)` (第157-166行)

显示响应矩阵的2D热图。

### 6.3 简单曲线绘制

**函数**: `plot_simple_curve(x, y, in_title)` (第372-380行)

**函数**: `plot_simple_multi_curves(x, ys, in_label)` (第383-394行)

---

## 七、主要测试函数

### 7.1 单曲线重建测试

**函数**: `rebuild_test_one_curve(src_dir)` (第495-598行)

**流程**:
1. 加载响应矩阵
2. 生成模拟输入光谱（多峰高斯）
3. 计算透射曲线（加入噪声）
4. 使用PyTorch优化器重建
5. 高斯滤波平滑结果
6. 可视化对比

### 7.2 无正则化最小二乘测试

**函数**: `lst_no_reg(src_dir)` (第713-742行)

使用非负最小二乘，无平滑约束。

### 7.3 增广正则化测试

**函数**: `lst_with_aug_reg(src_dir)` (第793-846行)

使用二阶差分正则化，加入噪声测试。

---

## 八、核心功能：CWL-FWHM循环重建

### 8.1 CWL和FWHM双重循环

**函数**: `lst_with_aug_reg_cwl_fwhm_loop()` (第1184-1366行)

**参数**:
- `src_dir`: 数据目录
- `cwl_range`: (起始CWL, 结束CWL, 步长)
- `fwhm_range`: (起始FWHM, 结束FWHM, 步长)
- `amplitude`: 高斯峰幅度
- `lam`: 正则化参数
- `save_results`: 是否保存结果

**算法流程**:
```
for each CWL in cwl_range:
    peak_idx = argmin(|wave_len_list - CWL|)
    for each FWHM in fwhm_range:
        sig = FWHM / 2.355
        input_spectrum = amplitude * gaussian_beam(index, peak_idx, sig)
        y_measured = A @ input_spectrum
        y_aug = [y_measured, zeros]
        x_reconstructed = lsq_linear(A_aug, y_aug, bounds=(0, inf))
        calculate MSE and FWHM
        save results
```

**数学模型**:
对于每个(CWL, FWHM)组合：
$$\min_x \|Ax - y\|^2 + \lambda \|D_2 x\|^2, \quad x \geq 0$$

### 8.2 仅CWL循环

**函数**: `lst_with_aug_reg_cwl_loop()` (第1369-1407行)

固定FWHM，仅循环CWL。

### 8.3 仅FWHM循环

**函数**: `lst_with_aug_reg_fwhm_loop()` (第1410-1448行)

固定CWL，仅循环FWHM。

---

## 九、结果保存功能

### 9.1 时间戳目录创建

**函数**: `get_timestamped_output_dir(src_dir, base_dir)` (第1024-1052行)

创建格式：`YYYYMMDD_HHMMSS-源目录名`

### 9.2 光谱对比图保存

**函数**: `save_spectrum_comparison_plot_with_fwhm()` (第1055-1130行)

保存内容：
- 输入光谱与重建光谱对比
- FWHM信息文本框
- CWL和FWHM标注

### 9.3 CSV结果保存

**函数**: `save_results_to_csv()` (第1133-1177行)

保存字段：
- Input_CWL_nm
- Output_CWL_nm
- Input_FWHM_nm
- Output_FWHM_nm
- MSE

---

## 十、数学总结

### 10.1 核心问题

光谱重建是一个**逆问题**（Inverse Problem）：

$$y = Ax \Rightarrow x = A^{-1}y$$

由于响应矩阵 $A$ 通常不是满秩或病态的，直接求逆不可行。

### 10.2 解决方案

1. **最小二乘法**: $\min_x \|Ax - y\|^2$
2. **正则化**: 加入平滑约束 $\lambda \|D_2 x\|^2$
3. **非负约束**: $x \geq 0$（物理约束）

### 10.3 正则化的必要性

响应矩阵的条件数通常较大，导致：
- 噪声放大
- 解不稳定
- 不物理的高频振荡

正则化通过引入先验知识（光谱平滑）来稳定解。

### 10.4 FWHM的意义

FWHM是衡量光谱分辨能力的核心指标：
- 输入FWHM: 原始光谱的峰宽
- 输出FWHM: 重建光谱的峰宽
- 差值: 评估重建质量

---

## 十一、代码结构图

```
rebuild_curves_cwl_fwhm_loop.py
│
├── 基础工具函数
│   ├── gaussian_beam()          # 高斯光束模型
│   ├── find_fwhm_normal()       # FWHM计算
│   ├── smoothness_loss_1d()     # 一阶平滑损失
│   ├── smoothness_loss_2nd()    # 二阶平滑损失
│   ├── build_D2()               # 二阶差分矩阵
│   └── load_matrix_from_file()  # 数据加载
│
├── 可视化函数
│   ├── draw_curves()
│   ├── draw_map_of_2d_angle()
│   ├── plot_simple_curve()
│   └── plot_simple_multi_curves()
│
├── 重建算法
│   ├── pytorch_optimizor()      # PyTorch优化器方法
│   ├── lst_no_reg()             # 无正则化最小二乘
│   ├── lst_with_aug_reg()       # 增广正则化最小二乘
│   └── lst_with_aug_reg_cwl_fwhm_loop()  # CWL-FWHM循环重建
│
├── 结果保存
│   ├── get_timestamped_output_dir()
│   ├── save_spectrum_comparison_plot_with_fwhm()
│   └── save_results_to_csv()
│
└── 测试函数
    ├── rebuild_test_one_curve()
    ├── lst_with_aug_reg_with_loop()
    └── main()
```

---

## 十二、使用示例

```python
# CWL和FWHM双重循环
results = lst_with_aug_reg_cwl_fwhm_loop(
    data_dir,
    cwl_range=(440, 800, 40),   # CWL范围
    fwhm_range=(10, 30, 5),     # FWHM范围
    amplitude=0.6,
    lam=0.01,
    save_results=True
)

# 仅CWL循环
results = lst_with_aug_reg_cwl_loop(
    data_dir,
    cwl_range=(440, 650, 10),
    fwhm=8,                     # 固定FWHM
    save_results=True
)

# 仅FWHM循环
results = lst_with_aug_reg_fwhm_loop(
    data_dir,
    cwl=550,                    # 固定CWL
    fwhm_range=(10, 30, 5),
    save_results=True
)
```

---

## 十三、物理背景

### 13.1 光谱成像系统

光谱成像系统通过滤光片阵列将不同波长的光分配到不同通道：
- 每个通道对特定波长范围有响应
- 响应矩阵描述各通道的波长响应特性

### 13.2 重建原理

已知：
- 响应矩阵 $A$（系统特性）
- 测量值 $y$（各通道输出）

求解：
- 原始光谱 $x$（入射光光谱分布）

### 13.3 应用场景

- 计算光谱成像
- 光谱仪校准
- 薄膜设计验证
- MEMS可调滤光片评估