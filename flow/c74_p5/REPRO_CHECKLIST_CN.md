# C74-P5 实验复现与论文证据检查清单

## 计算环境
- Python 3 + numpy/scipy/pandas/matplotlib/PyTorch CPU。
- 所有训练在 C64原始1600训练/240验证对上进行；冻结测试新队列分别为1024+1024+256+256原始C32 Biot forward。
- 真正的材料后验采样必须报告：方法预训练、MAP、Jacobian、采样所用PDE分别的成本；目前ESS表只按每种提案96次后验PDE评价比较，不能代表端到端成本。

## 复算
```bash
cd C74_P5_StrongBaselines_20261111
python c74_p5_analysis.py
python c74_p5_strong_baselines.py --test
# 如需逐一独立重新做4个MAP和96x4后验似然：
C74_P5_CASES=3 python c74_p5_exact_posterior_pilot.py
C74_P5_CASES=17 python c74_p5_exact_posterior_pilot.py
C74_P5_CASES=71 python c74_p5_exact_posterior_pilot.py
C74_P5_CASES=143 python c74_p5_exact_posterior_pilot.py
```

## 统计独立性
- `fresh1024_tuned_confirm`: 1024 个**材料真实场**为独立单位；3随机初始化结果在同一个材料场内平均，不能计为3倍样本数。
- 四个MAP后验pilot仅有4个观测问题，Importance ESS不证明收敛后验。
- alpha=.4/1.0检验直接施加固定alpha=.7的名义物理几何，因此是**模型错配压力测试**而不是适配后的泛化证明。

## 原创性限制
- RQS、LIS、Fisher信息、条件Flow均为已有技术。
- 目前支持的结构机制是**非线性容量应如何在物理信息方向间分配**，不声称发明了Spline或有普适后验校准保证。
