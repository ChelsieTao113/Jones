# prediction_type 公式与训练检查

任务 `train_000252`，输入 SHA256 `829ee01518c3e3f04f81a7ec677707f58836338b49c7b7d1b47948b2b9ad69ca`。
E/V 仅训练目标与 scheduler prediction_type 不同；两者初始模型和条件编码器哈希相同，20 步训练 t/噪声摘要逐次相同。
20 步耗时（秒/更新）E/V：0.552/0.518。
两组梯度、参数更新、checkpoint 恢复与一次 DDIM100 生成均通过。
float32 v→x0 最大相对 L2 9.17e-08，get_velocity 对显式公式最大绝对差 0。E/V 的 betas、alphas_cumprod 和 100 步 t 列表完全相同；DDPM/DDIM 配置差异均仅为 prediction_type，详见 [scheduler_comparison.json](../results/prediction_type_ablation/scheduler_comparison.json)。
在 t=990/500/0 用真实 epsilon/v 的独立 DDIM scheduler.step，pred_original_sample 最大绝对差 1.65e-05，prev_sample 最大绝对差 7.15e-07。t=990 的 epsilon 反解误差受 float32 消减与除小 a 影响，不套用中低噪声阈值。
epsilon 旧代码单次更新回归：四个微批次 t/噪声完全相同，loss 1.12556985/1.12556985，更新后最大参数差 0；此项单独启用确定性 CUDA 算法，正式训练保持原设置。详见 [epsilon_regression.json](../results/prediction_type_ablation/epsilon_regression.json)。
公式误差与各 t 的 SNR 见 [formula.csv](../results/prediction_type_ablation/formula.csv)，检查记录见 [smoke.json](../results/prediction_type_ablation/smoke.json)。
float64 和 float32 均验证 get_velocity、x0/epsilon 恢复；epsilon 高噪声反解按实际浮点误差单列。
smoke checkpoint 仅用于恢复检查，正式训练重新从共同随机种子初始化。
