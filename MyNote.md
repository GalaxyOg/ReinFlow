


显存占用主要由以下三个参数的 乘积 决定：

1. denoising_steps (配置文件 L27, 值为 40) ：
   
   - 影响：线性 。
   - 原因 ：这是流匹配（Flow Matching）的积分步数。在计算 PPO Loss 时（ ppoflow.py 中的 get_logprobs ），代码会执行一个 for 循环，循环次数为 40。PyTorch 需要保存这 40 次前向传播的所有中间激活值以进行反向传播。
   - 显存压力 ：这是导致显存爆炸的主要原因之一。40 步意味着网络深度相当于增加了 40 倍。
2. batch_size (配置文件 L78, 值为 50000) ：
   
   - 影响：线性 。
   - 原因 ：这是每次梯度下降更新时送入 GPU 的样本数量。
   - 代码陷阱 ：在 train_ppo_flow_agent.py L386 中，如果开启 repeat_samples: true ，数据量会被硬编码放大 10 倍 ( duplicate_multiplier = 10 )。
   - 计算 ：
     - 总样本数 = n_steps (500) * n_envs (8) * duplicate_multiplier (10) = 40,000。
     - 配置中的 batch_size: 50000 大于 40,000，这意味着 所有样本会在一个 Batch 中一次性进入 GPU 。
   - 后果 ：一次性对 40,000 个样本进行 40 步的展开反向传播，极易导致 OOM (Out Of Memory)。
3. n_envs (配置文件 L32, 值为 8) ：
   
   - 影响：线性 。
   - 原因 ：直接增加了收集到的数据总量。数据总量 = n_envs * n_steps 。