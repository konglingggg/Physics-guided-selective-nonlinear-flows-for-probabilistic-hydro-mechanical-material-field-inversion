# AI4THMC C74-P4 — 物理信息分配的条件 Spline Flow（2026-11-11）

## 研究状态

**已完成：代码实现、原始 C32 Biot 合成 PDE 数据生成、数值可逆性审计、三次独立网络初始化、2,048 个全新材料场—观测对的冻结测试、配对统计和复现归档。**

本轮保持 C73 的 512 维联合 `log-permeability` 和 `log-Young's modulus` 材料反演任务、同一 Biot M1-linked 求解器、`alpha=0.7`、`pulse` 载荷、42+48 H/M 观测和 C73 双侧物理几何观测条件表示。C73 论文不因本实验自动改动。

## 核心模型方法

- 基础分布：当前观测对应的 *参考点线性化* Biot tangent Gaussian，而不是精确的非线性后验。
- 完整 512 维先验白化材料向量经固定 Fisher 正交基和 *rank-balanced* 坐标排列，以便每个 256 维更新半块含有可观测信息方向。
- 四层交替可逆耦合层，保持所有 512 个坐标参与变换：每层在更新半块的 **24 个**指定方向上使用 K=5 的 Monotone Rational-Quadratic Spline，其余 **232 个**方向使用仿射耦合更新。
- `FisherSpline`：依据联合 Fisher 信息特征值，选择每半块信息最强的24个方向。
- `HMInfoSpline`：每半块选择12个 H 主导和12个 M 主导的物理信息方向。
- `RandomSpline`：网络容量和 spline 数量完全相同，随机选择每半块24个位置。
- `BalancedAffine`：相同 rank-balanced 坐标、C73 H/M 条件输入的纯仿射 RealNVP。
- `C73_HMBlockSVDWhiten`：**冻结原始 C73 checkpoint**，绝不重新训练，直接作为关键论文基线。

具体可逆层表达式为

\[
\begin{aligned}
b_j'&=\mathrm{RQS}_{\theta_j(a,c)}(b_j),&&j\in\mathcal I_{\rm phys},\\
b_j'&=e^{s_j(a,c)} b_j+t_j(a,c),&&j\notin\mathcal I_{\rm phys}.
\end{aligned}
\]

其 Jacobian 行列式由逐坐标 spline 导数和仿射尺度精确求得，所有原始材料维度保留。RQS 是 Durkan et al. (NeurIPS 2019) 的已有技术；物理信息子空间有 Cui et al. (JCP 2016) 等先例。**候选方法增量是物理敏感度驱动的可逆层非线性容量分配，而不是 spline 或 Fisher 本身的发明。**

## 开发数据和模型预算

- 训练 1,600 组、开发验证 240 组，完全复用 C64 原始 C32 PDE 生成的材料—观测配对；没有把旧 C64 的另外240组测试当作新的确认性证据。
- 训练初始化：`641133`、`641134`、`641135`；各模型相同优化器、32个 epoch 上限（首轮）或8个 epoch 上限，其最佳 checkpoint 均在第2到第4轮出现，8轮以后验证 NLL 持续变差。对照采用相同的验证损失及 checkpoint 选择规则。
- C73 模型参数 `302,848`；Spline 方案 `303,280`，仅相差 `+0.143%`。
- 模型的 `forward/inverse` 数值误差和精确密度已单独审计；详见 `architecture_audit.json`。

| 模型 | 三次初始化平均验证 NLL |
|---|---:|
| BalancedAffine | 240.266 |
| FisherSpline | **238.219** |
| HMInfoSpline | 238.238 |
| RandomSpline | 241.605 |

## 冻结独立测试：2,048组新 Biot 材料场与 H/M 观测

- 冻结数据随机种子 `742480`，实际通过原始 C32 Biot forward 生成 2,048 组材料场与观测，并重新加入对应噪声。此队列此前未用于训练/验证/选型。
- 每组独立物理真实场是**一个**统计个体；三次网络训练结果先在每个个体上平均，不将三次初始化当作3倍独立物理样本。
- 指标为测试材料真值的条件负对数密度 `NLL`（省略所有方法共有的 Gaussian 常数），越低越好。

| 方法 | 平均冻结 NLL |
|---|---:|
| 物理 Gaussian | 255.709 |
| C73 冻结原方法 | 240.399 |
| BalancedAffine | 240.087 |
| RandomSpline | 241.535 |
| FisherSpline | 237.811 |
| HMInfoSpline | **237.802** |

主比较（`候选−对照`）及 95% case-level paired bootstrap 区间：

- FisherSpline − C73：`−2.588`，CI `[-2.869, −2.327]`。
- FisherSpline − RandomSpline：`−3.724`，CI `[-4.021, −3.443]`。
- FisherSpline − BalancedAffine：`−2.276`，CI `[-2.526, −2.034]`。
- HMInfoSpline − FisherSpline：`−0.0087`，CI `[-0.0208, +0.0034]`，**尚无独立增量证据**。

五项预定比较执行成对 sign-flip permutation 与 Holm 校正，前三项 Holm p ≤ 0.001（5000次符号翻转，p 精度受模拟次数限制）。三次训练初始化的优势方向保持一致。`frozen2048_seed_replicates.csv` 保存逐种子的对照。

## 方法解释与创新性边界

**本轮正结果：** 在同一固定 Biot 合成物理制度、同一 H/M 条件输入、几乎相同参数量下，给物理 Fisher 高信息方向分配单调非仿射 spline 变换，比普通 C73 仿射流和随机方向 spline 流产生更准确的512维材料场条件对数密度。

**未成立的主张：** (a) H/M 互补方向在普通 Fisher 方向之上有额外增量；(b) 单次观测下精确非线性 Biot 后验采样/覆盖率更优；(c) 在不同载荷、噪声、alpha、真实现场数据条件下通用有效；(d) 新发明了 spline/Fisher 数学机制。这些均不作为论文已经通过的创新宣言。

**额外强基线仍值得补充：** 容量和训练预算匹配的全维 Neural Spline Flow、以及多种先验和物理条件下的独立重复。相关2025年论文已在706维地下水反演中使用条件 spline 流，需在投稿前检查原创性重叠。

## 文件

- `c74_p4_spline_flow.py`：可逆部分 spline、物理方向分配、条件训练和数值审计的完整源码。
- `c74_p4_frozen_biot_test.py`：全新原始 Biot 2,048 真值数据生成、冻结模型测试、配对 Bootstrap、sign-flip 和 Holm 校正。
- `architecture_audit.json`：参数量、可逆性、Jacobian 与方向选择。
- `checkpoints/`：3次初始化 × 4种训练模型的权重及每轮验证日志；冻结 C73 原模型由兄弟目录 C73 依赖提供。
- `development_validation_all_seeds.csv`：12次训练开发统计。
- `frozen_independent2048.npz`：独立 PDE 物理材料场真值和观测对。
- `frozen2048_all_seeds.csv` / `frozen2048_mean_seeds.csv`：逐案例完整评分及三初始化均值。
- `frozen2048_paired_statistics.csv` / `frozen2048_seed_replicates.csv`：完整的统计和稳定性检查。
- `C74_P4_frozen_NLL_comparison.png` / `C74_P4_paired_CI.png`：最终图表。

## 独立解压后复现

ZIP 包中有两个兄弟目录：`C73_HM_Information_Routing_20261111/` 和 `C74_P4_PhysicsAllocated_Spline_20261111/`。需 Python + NumPy, SciPy, Pandas, PyTorch, Matplotlib；CPU 可以运行。

```bash
cd C74_P4_PhysicsAllocated_Spline_20261111
python c74_p4_spline_flow.py --audit-only
python c74_p4_frozen_biot_test.py   # 使用已保存的全新Biot真值与冻结权重，重新核算全部评分
# 以下选项会重新训练，请先保留当前 checkpoint：
# python c74_p4_spline_flow.py --seed 641133 --epochs 8 --modes BalancedAffine,FisherSpline,HMInfoSpline,RandomSpline
```

`c74_p4_frozen_biot_test.py` 在已有 `frozen_independent2048.npz` 的情况下不会调用 PDE 重新生成数据，但评分仍严格使用原始训练所得的物理模型参数与冻结网络；删除该 `.npz` 会按固定独立种子重新使用原始求解器生成2,048个全新的 Biot forward 样本。

## 必读文献

- Durkan et al., *Neural Spline Flows*, NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html
- Cui, Marzouk & Willcox, *Scalable posterior approximations for large-scale Bayesian inverse problems via likelihood-informed parameter and state reduction*, JCP 2016. https://doi.org/10.1016/j.jcp.2016.03.055
- Zeng et al., *Solving high-dimensional inverse problems using amortized likelihood-free inference with noisy and incomplete data*, CMAME 2025. https://doi.org/10.1016/j.cma.2025.118064

## 研究决策

C74-P4 相对 C73 的条件 NLL 增量**达到本轮开发和独立测试通过标准**。其独立方法新颖性、精确后验适用性仍需进一步论证。保留 C73 论文原版，建议把 P4 列为可升级论文主方法的候选，补做对应强方法基线和后验验证之后再正式替换。
