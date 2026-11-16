# AI4THMC C74-P5：物理信息分配 Spline Flow 的强基线、跨物理条件与非线性后验诊断

**日期：2026-11-11；研究定位：C73论文的独立增强证据，不直接覆盖C73正式稿。**

## 1. 固定科学问题与目标

仍研究真实 M1-linked Biot PDE 下，有限压力、位移观测对完整512维（256维 log-k +256维 log-E）空间材料场条件概率分布的生成式反演；原始 C73 H/M 双侧信息几何保持不变。C74-P4 新机制是每个512维可逆耦合层在更新半块的24个高物理信息方向使用K=5 Rational-Quadratic Spline，在其余232个方向使用仿射耦合。

**C74-P5 不新增候选创新层。** 目的为：(A) 补足同参数量和大参数量的全维 Spline 强基线；(B) 验证不同信息分配；(C) 小规模实际非线性后验检查；(D) 考察固定模型在物理耦合参数变化下的分布外表现。

## 2. 实验可比性

训练样本仍为C64原始1600 Biot材料—观测对；开发验证240例。C73、C74-P4模型冻结；新增 WeakSpline（每半块24个弱信息方向，303,280参数）、FullSplineMatched（每半块全部256方向Spline、hidden=18，298,736参数）、FullSplineWide（全部256方向Spline、hidden=62，1,004,848参数）。全维大模型额外在开发集选择较低学习率2e−4，对照原8e−4，最终采用更低学习率版本。三随机训练种子641133/641134/641135、相同RQS K=5、条件观测90维、数据、先验和基准优化目标。全维网络的条件器宽度与P4不同，因此容量/参数量对齐不等于完全相同的表达类。所有分布选择仅在开发验证进行；确认性测试为第二批全新1024例 `fresh1024_tuned_confirm`。

| 网络 | 可训练参数 | 三初始化平均开发NLL |
|:---|---:|---:|
| 原C73双侧几何Affine | 302848 | 约240.533（旧开发记录） |
| C74-P4 FisherSpline | 303280 | 238.219（既有P4记录） |
| WeakSpline | 303280 | 241.557 |
| 全维Spline—匹配容量 | 298736 | 240.050 |
| 全维Spline—大容量、原学习率 | 1004848 | 240.934 |
| 全维Spline—大容量、调优学习率 | 1004848 | 240.125 |

## 3. 第一批新测试（新1024例，alpha=.7）

`fresh1024_nominal` 在第一批强基线训练后冻结测试，P4 FisherSpline平均NLL约237.527，C73约240.029，全维参数量匹配Spline约239.506；相对匹配全维Spline降低1.979 NLL，95%配对CI [1.815,2.168]（以改善量为正）。这批结果不作为新增学习率调优方案的最终确认数据，因为调优是在之后开展的。

## 4. 第二批全新确认性测试：1024例 alpha=.7

固定开发选出的低学习率全维Spline checkpoint，新生成1024组Biot合成材料—观测对，种子744521，原始C32 PDE，90个观测，噪声同开发分布。每个材料真实场为统计独立单位；三次网络初始化在个体内平均。

| 模型 | 平均条件NLL |
|---|---:|
| C73 Affine Flow | 240.560 |
| 全维Spline—匹配参数 | 240.040 |
| 全维Spline—大容量+调优学习率 | 239.997 |
| 随机24方向Spline | 241.758（见原始CSV精确值） |
| 弱信息24方向Spline | 241.608（见原始CSV精确值） |
| **Fisher强信息24方向Spline** | **238.162** |
| H/M互补24方向Spline | 238.149 |

主要配对优势（FisherSpline减对照，负值更好）：

- C73：−2.398，95% CI [−2.703,−2.086]；
- 全维匹配Spline：−1.878，CI [−2.035,−1.726]；
- 全维大容量、调优Spline：−1.835，CI [−2.003,−1.670]；
- 弱信息方向Spline：−3.446，CI [−3.783,−3.114]；
- 随机方向Spline：−3.596，CI [−3.947,−3.251]。

五项对比的5000次配对sign-flip检验，Holm校正后p≤0.0012（蒙特卡洛精度限制）。H/M互补方向相对一般Fisher方向差值−0.013，CI跨零，不能主张额外优势。**实验支持物理信息驱动的非线性容量分配，在当前条件下优于均匀全维Spline及非物理方向控制；还不足以证明H/M独特分配优于普通Fisher信息分配。**

## 5. 冻结模型的分布外物理条件测试（alpha=.4 和 1.0）

各256个全新真实C32 Biot forward材料—观测对。材料先验和观测噪声保持相同，输入信号由alpha不同的真实PDE产生；但模型及Gaussian物理锚定均**不重新训练或重新计算适配alpha的切线/Jacobian**。因此仅为“名义模型错配敏感性”检查，不是适配后的跨条件验证。

- `alpha=.4`：FisherSpline 413.364，C73 469.162；
- `alpha=1.0`：FisherSpline 427.140，C73 451.220。

与名义条件下238左右的NLL相比，分布外NLL极高，显示模型及物理锚定不适合直接用于这些新条件的定量后验校准。相对排名虽然通常有利于FisherSpline，但**不能宣称跨alpha概率反演已经成功**。

## 6. 真实非线性Biot后验：严格区分条件密度预测和实际后验采样

取新1024例独立材料真值中的case 3/17/71/143，每例使用全90维观测求联合 Nonlinear MAP，完整90×512 Biot Jacobian构造同一MAP切线Gaussian。MAP的最终梯度范数依次约2.5e−6、2.0e−6、6.8e−6、1.9e−5。各方法使用同一批MAP Gaussian基础扰动、96个新的非线性Biot forward和精确可逆的变换密度，算单组观测的Importance ESS。Flow作MAP-centered位移变换，已考虑Jacobian和归一化权重。该MAP-centered变换属于单独的物理校准策略，不是网络原生训练时的名义Gaussian条件分布。

| 方法 | 4例平均Importance ESS/96 | 加权log-E RMSE均值 |
|---|---:|---:|
| MAP-tangent Gaussian | 12.74 | 0.3631 |
| 原C73 | 4.74 | 0.3880 |
| C74-P4 FisherSpline | 2.53 | 0.4146 |
| 全维匹配Spline | 1.85 | 0.4379 |

这些仅是4个观测、每种96次PDE的**重要性采样诊断**，不是收敛的后验参考，也不能直接解释成方法泛化的显著性结论。它们说明提高held-out conditional NLL并未自动改善真实非线性后验提案匹配。在这组试验中，MAP Gaussian仍然是强基线。

## 7. 论文论证决策

1. 保持原C73论文与C73冻结果不变。
2. C74-P4通过**条件生成密度**强对照（同参数全维、三倍参数调优全维、弱信息、随机方向），具备增加一项结构化模型创新贡献的数值证据。
3. 当前不能主张C74-P4在真实非线性后验Importance ESS、后验覆盖率或材料RMSE上超越MAP Gaussian。
4. 若升级稿件，应以“物理信息引导的非线性容量分配提高完整512维材料场条件密度学习”为主创新，并显式将exact likelihood诊断列为边界；下一项真正能增强原创性的工作是跨alpha重新计算对应物理几何并训练/适配模型，而不应把本轮名义模型分布外失配视为通过。
5. Spline、Fisher/LIS、条件Normalizing Flow均有既有文献，原创性还需要与采用相似信息分配机制的最新研究核对。

## 8. 复现命令

ZIP根目录保持三个相邻文件夹 `C73_HM_Information_Routing_20261111`、`C74_P4_PhysicsAllocated_Spline_20261111`、`C74_P5_StrongBaselines_20261111`，Python需求numpy, scipy, pandas, torch, matplotlib。

```bash
cd C74_P5_StrongBaselines_20261111
python c74_p5_analysis.py                           # 无训练、无PDE，复核统计和图
python c74_p5_strong_baselines.py --test           # 重新打分，若数据已存在不重生成PDE
# 重新训练可能覆盖checkpoint；先备份当前权重
python c74_p5_strong_baselines.py --train --epochs 8
python c74_p5_tuned_full_spline.py 641133
C74_P5_CASES=3 python c74_p5_exact_posterior_pilot.py # 重新做96x4真实PDE评价和MAP
```

- `fresh1024_tuned_confirm_*`：确认队列逐材料真值评分、配对统计。
- `cohort_fresh1024_tuned_confirm.npz`：独立原始Biot数据；其余cohort保存名义初测、alpha错配测试。
- `checkpoints/`：所有新增控制模型及学习率调优checkpoint和开发历史。
- `exact_posterior_pilot_all_logweights.csv`、`exact_posterior_pilot_cases.csv`：每个真实物理似然权重与ESS。
- `primary_confirmation_full_statistics.csv`：1024材料真值级的Bootstrap、signflip和Holm统计。
- `P5_summary.json`：集中汇总。

**学术定位：此轮加强了C74-P4的条件概率密度模型创新证据，未证明精确Biot后验采样优势。**
