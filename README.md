# CNN Snake

使用 MaskablePPO 和图像输入学习贪吃蛇。实验默认 8×8 棋盘、初始蛇长 3，吃满棋盘才计为通关；可用 `--board-size 12` 恢复原棋盘规模。棋盘缩小减少了通关所需食物数，但后期仍然需要避免把自己困住。

## 运行环境

已在现有 `SnakeAI` Conda 环境验证：Python 3.8.16、PyTorch 2.0.0、Gym 0.21.0、Stable-Baselines3 1.8.0、sb3-contrib 1.8.0、pygame 2.3.0。当前保留旧 Gym 接口，请勿单独把 SB3 升级到 2.x；迁移时需要一并调整 Gymnasium 的 reset/step 接口。

在项目根目录运行：

```bash
conda activate SnakeAI
python -m src.train_cnn
```

默认 8 个环境、200 万环境步。每次训练在 `trained_models_cnn_easy/` 创建独立目录，包含 `config.json`、`progress.csv`、`evaluations.npz`、`best_model.zip` 和最终模型。训练进度同时显示在终端；Ctrl+C 会保存 `interrupted_model.zip`。原有模型目录不受影响。

自定义训练或先跑小规模流程验证：

```bash
python -m src.train_cnn --timesteps 2000000 --num-envs 8 --board-size 8
python -m src.train_cnn --timesteps 256 --num-envs 1 --n-steps 64 --eval-every 128 --eval-episodes 2 --device cpu --save-root /tmp/snakeai-smoke
```

`--device auto` 按 CUDA、MPS、CPU 的顺序选择可用设备。相同种子可复现环境随机数；不同硬件或依赖版本下，训练出的权重不保证逐位相同。

## 演示与评估

```bash
# 自动选择最新一次新实验中的最佳模型，读取同目录的棋盘配置
python -m src.test_cnn

# 无窗口评估；通关率定义为吃满棋盘的局数占比
python -m src.test_cnn --headless --episodes 100 --device cpu

# 使用与训练中定期评估相同的保留种子（默认训练种子 + 10000）
python -m src.test_cnn --headless --episodes 20 --seed 20270927 --device cpu

# 指定模型：新模型自动读取 config.json；旧模型需要明确原棋盘大小
python -m src.test_cnn --model trained_models_cnn_mps/ppo_snake_3000000_steps.zip --board-size 12

# 每局显式重置相同种子，便于复现同一策略的表现
python -m src.test_cnn --repeat-seed --episodes 5
```

窗口中可按 Escape 或关闭按钮退出。评估输出平均食物数、步数、奖励，以及成功、碰撞、超时的局数。最佳模型由平均评估奖励选择，训练日志另外记录 `eval/success_rate`；平均奖励提高不等于通关率一定提高。

## 随机种子与公平比较

训练与演示共用 `src/config.py` 中的种子 `20260927`。训练环境使用 `seed + 环境编号`，评估使用未参与训练的 `seed + 10000`。每批训练评估都会重启同一个随机数序列，便于比较模型；批内后续局继续推进随机数序列。

`env.reset(seed=20260927)` 会重启游戏随机数；普通 `env.reset()` 继续当前序列。食物从排序后的空闲格中抽取，因此要复现完整食物序列，棋盘、重置次数和蛇的动作也必须相同。不同动作会改变空闲格，即使随机数相同也可能产生不同的后续食物位置。

修改了奖励、食物采样顺序和棋盘大小后，建议新开训练；旧模型仍可指定原棋盘加载，但不能据此声称新旧训练过程完全相同。先比较固定条件下的平均食物数，再以足够多局的真实通关率判断是否改善。

## 回归检查

```bash
python -m unittest discover -s tests -v
```

覆盖种子重放、掩码与实际碰撞一致性、失败时蛇长保持不变、满盘终局、超时边界，以及多种棋盘大小的观察空间。
