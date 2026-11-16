# AI4THMC C73 — H/M Dual-Geometry Conditional Flow

**研究任务**：固定“利用异质压力与位移观测及物理敏感度，提高512维非均质材料场的条件概率生成准确性”的论文科学问题；C73研究如何让 Hydro-Mechanical 物理几何进入 Normalizing Flow 的**内部条件表示**，而不是仅改变模型层数或反演未知量维度。

**实验状态（2026-11-11）**：实际使用原始 C32 M1-linked nonlinear transient Biot PDE 产生训练和三个后续独立评估队列；全部对照都保留16×16 log-k 和16×16 log-E（512维）、三时间步、42压力+48位移（90观测）。预训练原始数据：1,600训练、240验证、240历史测试；三个不同NN初始化种子641133–641135；所有主要Flow均为四层Conditioned RealNVP、302,848个可训练参数。

## A. C73 第一候选：H/M 信息路由门控（未通过）

从 C72 Fisher 材料坐标出发：(i)将H/M主导的敏感方向按固定物理规则组织在耦合层两侧；(ii)把H、M观测梯度投影为每个方向的数据驱动正值门控，控制每层仿射耦合变换。比较 Fisher、H/M固定分组、H/M动态门控、打乱物理投影的动态门控。

在256组新Biot材料场上，H/M动态门控平均NLL 242.197，普通Fisher 242.911，但配对区间跨零。**随后没有修改模型，直接用2,048组不同随机种子的全新Biot材料场独立复验**：动态门控242.571、普通Fisher242.664、仅H/M静态分组242.490、打乱门控242.492。动态门控相对Fisher差值−0.094，95% CI [−0.347,+0.151]；与两组强消融相比甚至没有平均优势。

**结论**：第一种动态门控不能用作已经成立的方法创新；保留负结果和全部模型、种子、原始数据。

## B. C73 第二候选：H/M 双侧物理信息几何条件生成（已在条件NLL上通过）

对联合压力–位移线性化Biot响应构造先验白化Jacobian：

`A_H=Gamma_H^{-1/2} J_H T_M`（42×512），`A_M=Gamma_M^{-1/2} J_M T_M`（48×512）。

其中 `T_M` 是C32已有的材料先验白化转化，`z` 为完整512维先验标准正态坐标。材料侧沿用C72由 `A^T A` 得到的**完整512维 Fisher 正交坐标**，不删去弱敏感方向。相比于C72只变换材料空间，C73将观测侧也做几何对齐：

`A_H = U_H Sigma_H V_H^T`、`A_M = U_M Sigma_M V_M^T`。

对当前观测残差 `r_H=Gamma_H^{-1/2}(y_H-G_H(m0))`，条件编码变为

`c_H = r_H U_H (I+Sigma_H^2)^{-1/2}`，相应地 `c_M = r_M U_M (I+Sigma_M^2)^{-1/2}`。

输入Conditional RealNVP的上下文是 `[c_H,c_M]` 共90维，输出仍是 **全512维 log-k/log-E 联合分布**。两个模态没有复杂的专家消息网络，简单拼接即可；本轮创新不声称H/M神经协同。Flow保持可逆、显式可算密度；上下文变换只依赖冻结的原始Biot参考Jacobain和当前观测，**绝不使用测试真值**。此处的 `I+Sigma^2` 是线性先验预测的观测协方差谱，它不是非线性后验精确协方差。

### 物理与非物理强基线

- `Fisher`: C72完整512维Fisher材料坐标Flow，原始90维观测残差条件；
- `JointSVDWhiten`: 90维 H+M 整体SVD并按联合预测方差白化，不尊重H/M分块；
- `SensorDiagNorm`: 对每个原始传感器按其预测方差归一化，无物理模态特征重排；
- `RandomBlockWhiten`: H/M两模态分别随机正交旋转，并按每个旋转坐标的真实线性预测方差归一化，与本方法容量及输入维数一致；
- `HMBlockSVDWhiten`: 本次建议方法。

### 冻结新队列 N=2,048 的结果（seed731960，与前述2,048队列seed731900不同）

所有数值为三训练种子对每个材料真值的平均条件NLL（越低越好）；各模型保持相同物理参数、相同训练/验证划分、相同观测与网络容量。

| 模型 | 2048真值平均 NLL |
|---|---:|
| Joint physics Gaussian | 254.8725 |
| C69原始Generic Joint Flow | 246.7562 |
| Fisher | 242.2004 |
| JointSVDWhiten | 242.0995 |
| SensorDiagNorm | 241.8014 |
| RandomBlockWhiten | 239.9584 |
| **HMBlockSVDWhiten** | **239.5774** |

主比较为各观测真值下对三个训练种子取均值后做配对检验，避免将三个初始化误当作三倍独立物理样本。

| HMBlockSVDWhiten相对对照 | 平均NLL差值 | 95%配对Bootstrap CI | Holm校正p（6000次符号翻转，精度下限） |
|---|---:|---|---:|
| Fisher | −2.6230 | [−2.9321,−2.3167] | 0.000667 |
| JointSVDWhiten | −2.5221 | [−2.8362,−2.2209] | 0.000667 |
| RandomBlockWhiten | **−0.3809** | **[−0.4509,−0.3102]** | **0.000667** |
| SensorDiagNorm | −2.2240 | [−2.5100,−1.9542] | 0.000667 |

**最重要的可归因效应是对随机H/M分块白化的 −0.3809 NLL**，而不是对弱物理高斯的差距。随机对照说明，归一化/分块本身贡献了不少改善，但真实Biot模态奇异向量又额外提供了可重复、可检验的增量。

### 三次训练随机初始化的独立重复

`dual_fresh2048_seed_stability.csv` 记录三种不同初始化分别的完整比较；每次的 HMBlockSVDWhiten 均优于同种子 `Fisher` 和 `RandomBlockWhiten`。数据只来自一组2048个独立材料场/观测的物理条件，不是跨不同载荷/噪声/耦合强度的重复。

### 数值准确性与成本

1. 512维材料场和90维观测均保持原始维度；物理Jacobian按原始C32求解器计算；所有网络参数数目302,848。
2. 正规化条件后验密度直接用可逆Flow Jacobian 行列式与同一解析物理切线 Gaussian 计算；`c=(c_H,c_M)`只作为确定的条件输入，不涉及对材料密度的额外坐标Jacobian。
3. C73双侧方法的前向/逆向误差小于1e−6（float32）；每个独立测试材料真值直接调用原始非线性C32 solver生成H/M观测；全部原始条件NLL保留。
4. 预训练数据复用C64已经生成的1600/240；每套新网络仅进行神经优化，无额外的训练PDE标签生成。固定参考的Fisher/SVD基构造需要一次原始物理Jacobian，推断新的观测时无需重新计算Jacobian；完整端到端预算需另行统计（尤其与MAP-centered后验基线比较）。

### 必须保留的研究边界

- **通过的是独立生成材料场对应的条件 Log Score 准确性**，不是每个指定观测下“精确非线性Biot后验”的KL或MCMC ESS。C64–C70先前观察到的后验提案退化不能因为这次NLL改善而忽略。
- H/M块SVD加预测方差白化借用了成熟线性代数与贝叶斯信息几何思想。要把它表述为**大的AI算法原创性**，仍需要与现有物理对齐生成模型、普通LIS/SVD-Flow等文献严谨对照，并证明不可由成熟预处理直接替代。
- 当前是**固定参考材料/α=0.7/pulse载荷/噪声定义的原始C32合成问题**。还没有确认跨噪声、载荷、材料先验或非线性后验采样的优势。
- 此处没有产生现场观测证据，不把PDE合成数据误称为真实场地测量。

## 复现入口

- `c73_route_flow.py`：原先H/M材料方向门控候选及基线训练；
- `c73_frozen_test.py`、`c73_frozen_confirm2048.py`：先前否定的门控方案256/2048独立检查；
- `c73_dual_geometry_flow.py`：H/M双侧模态SVD、联合SVD、随机分块、逐传感器归一化的等容量模型训练；
- `c73_dual_frozen_confirm.py`：**主结果**：重新生成（若NPZ不存在）或直接用冻结的2048 Biot队列复算各模型条件密度、Bootstrap和Holm；
- `c73_analysis_package.py`：图表、C69原始checkpoint对照、训练日志总审计和可逆性检查；
- `checkpoints/`和`dual_checkpoints/`：所有模型冻结checkpoint及验证日志；
- `dual_independent2048.npz`、`dual_fresh2048_all_seeds.csv`、`dual_fresh2048_primary_statistics.csv`：主试验完整数据；
- `deps/C72_InformationGeometry_Pilot_20261111/`：原始C32 solver、C64数据、C69 pretrained Joint Flow、C72 Fisher基。无需之前会话的外部下载链接。

运行（解压后于本目录执行）：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python c73_dual_frozen_confirm.py
python c73_analysis_package.py
```

本研究最适合的下一步：围绕该**已验证的双侧几何条件表征**，进一步构建真正物理结构化的可逆生成模块，并在强Generic/LIS/MAP等同预算基线下验证真实非线性后验准确性；同时对照已有文献核查原创性。不要把第一版失败的动态门控包装成新的贡献。
