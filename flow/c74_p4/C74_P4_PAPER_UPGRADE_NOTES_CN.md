# C73→C74-P4：论文方法升级建议（暂不修改C73正式稿）

## 建议的潜在论文标题

**Physics-Allocated Nonlinear Transport for High-Dimensional Generative Material-Field Inversion in Biot Poroelasticity**

物理信息分配的非线性生成传输：Biot孔弹性高维材料场反演。

## 最核心、已验证的模型机制

非线性容量预算不平均地分给所有512个潜变量，而是依据 Biot Jacobian 与材料先验构造的 Fisher 信息方向选择可逆 spline 更新位置，其余位置仍保留可训练仿射变换；完整材料场的概率密度精确可算。

## 四个紧密衔接的论证层次

1. C73：H/M 双侧几何条件对齐能改善同容量 Conditional Flow 的条件 NLL。
2. C74-P4：在该物理条件表示上增加相同参数预算的非仿射 spline 层时，**选择哪些物理方向非仿射**对独立 NLL 至关重要；与随机分配比较可以归因。
3. C74-P4：以冻结原始 C73 与相同 balanced Fisher 坐标的仿射流作为双重强对照，新方法在三初始化、2,048例新的真实 Biot 正演合成任务中稳定领先。
4. 未验证：该收益对新物理条件、与全维等预算 Spline Flow 的比较及单次观测的精确非线性后验质量。投稿前按目标期刊决定哪些属于必需验证。

## 当前不应声称

- 不是发明了 Rational-Quadratic Spline；
- 不是发明了 Fisher/LIS；
- H/M互补采样没有显著超越简单 Fisher 采样；
- 不能用 conditional NLL 的改善直接宣称真实非线性 posterior KL、coverage 或 ESS 已改善。

## 建议论文贡献措辞（英文）

"We introduce an information-allocated invertible generative architecture that reserves nonlinear spline transformations for selected Biot likelihood-informed material directions, while retaining full-dimensional affine updates and tractable conditional density. Under matched network capacities, this physical allocation yields lower held-out conditional log scores than affine and randomly allocated spline controls on independent high-dimensional hydro-mechanical simulations."
