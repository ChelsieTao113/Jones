# Jones 矩阵扩散：epsilon 与 v_prediction 配对实验

本仓库保存单任务 `train_000252` 的受控实验源码、冻结输入、逐通道数据和图表。E 使用 epsilon 预测与普通 MSE；V 使用 `v_prediction` 与普通 MSE。两组的网络、条件编码、噪声日程、优化器和 DDIM100 采样设置一致。

三次独立训练种子各训练 2000 次更新，每次用六个相同的纯噪声生成种子评价。V 在三次训练中均改善成像，但两组最终都没有生成满足原全部门槛的样本，不能据此开展全量训练。

| 训练种子 | E/V 最差通道 PCC 中位数 | E/V 最大 `e_abs` 中位数 | 合格数 E/V |
| --- | ---: | ---: | ---: |
| 20261124 | 0.092 / 0.689 | 0.964 / 0.500 | 0/6 / 0/6 |
| 20261125 | 0.083 / 0.722 | 0.987 / 0.460 | 0/6 / 0/6 |
| 20261126 | 0.008 / 0.728 | 3.415 / 0.484 | 0/6 / 0/6 |

## 阅读结果

- [中文主报告](生成式模型/Diffusion/physics_loss_ablation/reports/PREDICTION_TYPE_ABLATION_REPORT.md)
- [公式与训练检查](生成式模型/Diffusion/physics_loss_ablation/reports/PREDICTION_TYPE_TEST_REPORT.md)
- [一组纯噪声生成的三通道对照图](生成式模型/Diffusion/physics_loss_ablation/figures/prediction_type_ablation/seed_20261201_focused_comparison.png)
- [全部六个生成种子的接触表](生成式模型/Diffusion/physics_loss_ablation/figures/prediction_type_ablation/contact_absolute.png)
- [逐通道 CSV、分噪声 CSV、采样轨迹及来源哈希](生成式模型/Diffusion/physics_loss_ablation/results/prediction_type_ablation)

## 复现

实验在 Windows、Python 3.12.7、PyTorch 2.14.1+cu126、Diffusers 0.43.0 和 RTX 4050 Laptop GPU 上运行。固定任务的 NPZ 包含在 `生成式模型/Training_Set/runs/analyzer_formal_train_part4/solutions/`。所需项目源码位于 `生成式模型/Diffusion/Diffusion_prep/`，执行入口位于 `生成式模型/Diffusion/physics_loss_ablation/scripts/`。

在 `生成式模型/Diffusion/physics_loss_ablation` 中运行：

```powershell
$python = 'C:\Users\david\anaconda3\envs\metasurface\python.exe'
& $python scripts\prediction_type_ablation.py --check
& $python scripts\prediction_type_ablation.py --run-paired --train-seed 20261124
& $python scripts\prediction_type_ablation.py --evaluate --train-seed 20261124
& $python scripts\prediction_type_ablation.py --report
```

其余两个训练种子为 `20261125` 和 `20261126`，以相同命令分别替换 `--train-seed`。完整命令见[实验目录说明](生成式模型/Diffusion/physics_loss_ablation/README.md)。

**模型 checkpoint 未上传。** 它们包含优化器和随机状态，保存在本地实验目录；重新训练后才能用 `--evaluate` 重算生成图。仓库提供运行代码、固定输入、种子、配置、来源 SHA256 和已算好的结果数据。`results/.../artifact_sizes.json` 记录的是原实验本地运行目录，包含未上传的 checkpoint，作为本地空间审计留存。
