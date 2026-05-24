# Agent 快速编码与运行指南

> 本文档面向 AI Agent，基于 `code-ref/` 参考代码的运行逻辑与工程结构编写。
> **核心原则**：Agent 应理解参考代码的设计思想后自主重新实现，不可直接复制粘贴 `code-ref/` 中的代码到 `code/` 目录。

---

## 0. 实验失败教训总结（必须优先阅读）

以下教训来自连续 5 次实验的失败报告，是 Agent 生成代码时**最容易忽视但最致命**的问题。

### 0.1 CLI 参数接口不匹配

实验调度器会以固定格式调用 `train.py` 和 `infer.py`。如果 Agent 生成代码参数接口与 runner 的调用格式不兼容，程序会在 `argparse` 阶段直接退出。

三任务推荐兼容命令：

```bash
python code/train.py --task task1 --output_dir output/task1/iter_N --data_dir ./data/task1
python code/train.py --task task2 --output_dir output/task2/iter_N --data_dir ./data/task2
python code/train.py --task task3 --output_dir output/task3/iter_N --data_dir ./data/task3

python code/infer.py --task task1 --checkpoint output/task1/iter_N/best_checkpoint.pt --output output/task1/iter_N/pred.hdf5 --data_dir ./data/task1
python code/infer.py --task task2 --checkpoint output/task2/iter_N/best_checkpoint.pt --output output/task2/iter_N/pred.hdf5 --data_dir ./data/task2
python code/infer.py --task task3 --checkpoint output/task3/iter_N/best_checkpoint.pt --output output/task3/iter_N/pred.hdf5 --data_dir ./data/task3
```

**失败模式**：

| 实验轮次 | 错误信息                                                     | 原因                                                   |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| Exp 1-4  | `error: the following arguments are required: --data`        | runner 未传入 `--data`，且 Agent 脚本将其设为 required |
| Exp 5    | `error: unrecognized arguments: --task task1 --output_dir output/task1/iter_5` | Agent 脚本不认识 `--task` 和 `--output_dir`            |

**解决方案**：

1. `train.py` 和 `infer.py` 必须显式支持 `--task`、`--output_dir`、`--data_dir`

2. 使用 argparse 别名同时支持下划线和横线版本：

   ```python
   parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output")
   parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="")
   ```

3. 所有参数必须有合理的默认值，runner 不会传入所有参数

4. **checkpoint 必须保存为 `best_checkpoint.pt`**，因为 runner 硬编码寻找此文件

### 0.2 数据路径缺失

当前 PDEAgent 已确认的数据目录结构是固定的：

```python
TASK_DATA_DIRS = {
    "task1": "./data/task1",
    "task2": "./data/task2",
    "task3": "./data/task3",
}
```

默认路径建议：

```python
if not args.data_dir:
    args.data_dir = TASK_DATA_DIRS[args.task]
```

不要再引用旧的 `data_and_sample_submission/train_val_test_init` 或 `task3_data_sample_submission/train_val_test_init` 目录。

### 0.3 验证必须匹配真实推理方式

- Task 1 / Task 2：验证时必须从前 10 步完整预测未来 190 步。
- Task 3：验证时必须从前 20 步完整预测未来 380 步，并拼成 400 步轨迹。

不得在 validation 中使用 teacher forcing 得到虚高分数。

### 0.4 输入窗口一致性检查不可省略

Task 1 / Task 2：

```python
assert np.allclose(full[:, :10, :], raw_test[:, :10, :], atol=1e-6)
```

Task 3：

```python
assert np.allclose(full[:, :20, :], raw_test[:, :20, :], atol=1e-6)
```

### 0.5 GPU 自动检测

不要硬编码为 CPU。训练和推理都应自动使用 CUDA：

```python
device = torch.device(args.device)
model.to(device)
```

---

## 1. 项目目标速览

Agent 需要为 PDE 神经算子预测任务生成完整可运行代码，并支持三项任务：

| 任务 | 数据 | 输入 → 输出 | 特殊要求 |
|------|------|-------------|----------|
| Task 1 | `1D_Burgers_Sols_Nu0.001.hdf5` / `task1_val.hdf5` / `task1_test.hdf5` | 10 步 → 200 步 | $\nu=0.001$，可用官方 checkpoint；训练/微调应基于 PDEBench 大型训练集，`task1_val.hdf5` 仅作验证 |
| Task 2 | `task2_part{0,1,2}_train.h5` / `task2_test.h5` | 10 步 → 200 步 | 测试不提供 $\nu$，必须从头训练 |
| Task 3 | `KS_train.hdf5` / `KS_val.hdf5` / `KS_test.hdf5` | 20 步 → 400 步 | 测试不提供 $\lambda_2$，必须从头训练 |

提交要求：

- Task 1 / Task 2：`task{N}_pred.hdf5` shape `(N,200,256)`，前 10 步一致。
- Task 3：当前本地设置下 `task3_pred.hdf5` shape `(100,400,256)`，前 20 步一致；代码实现仍应通过 `raw_test.shape[0]` 动态处理样本数。
- 每个任务推理时间必须小于 120 秒。

---

## 2. 你必须生成的文件清单

```text
code/
├── model.py      # 神经网络模型定义
├── dataset.py    # 数据加载与归一化
├── train.py      # 训练入口脚本
├── infer.py      # 推理与提交生成脚本
└── utils.py      # 评分计算、损失和辅助工具
```

所有文件必须由 Agent 通过允许的文件写入工具生成，不能由人工预置。

---

## 3. dataset.py — 数据加载与归一化

### 3.1 核心设计思想

- 对速度场使用全局 scalar mean/std 归一化。
- 归一化参数必须在训练集上计算，并保存到 checkpoint。
- Task 2 的 `nu` 和 Task 3 的 `lambda2` 训练时可用，测试时不可用。
- 测试集 loader 不能假设存在 `nu` 或 `lambda2` 字段。

### 3.2 必须实现的类与函数

```python
class Normalizer:
    def __init__(self, mean: float, std: float): ...
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def decode(self, x: torch.Tensor) -> torch.Tensor: ...
    def as_dict(self) -> dict: ...

class BurgersDataset(torch.utils.data.Dataset):
    """Task 1 / Task 2 数据集。返回 input, target, optional nu。"""

class KSDataset(torch.utils.data.Dataset):
    """
    Task 3 数据集。
    train/val 返回:
      x: [20, 256]
      y: [380, 256] 或 full: [400, 256]
      lambda2: [1]
    test 返回:
      x: [20, 256]
      y: None
      lambda2: None
    """

class WindowedBurgersDataset(torch.utils.data.Dataset): ...

class WindowedKSDataset(torch.utils.data.Dataset):
    """
    可选：用于 Task 3 chunked 训练。
    从完整 400 步轨迹中生成短窗样本，如 input=20, target=chunk_size。
    注意验证仍必须完整 rollout 380 步。
    """

def get_dataloaders(data_dir, task="task1", batch_size=16, val_fraction=0.2,
                    model_type="chunked", chunk_size=10, t_in=None, t_out=None, **kwargs): ...

def get_test_loader(data_dir, task="task1", batch_size=64, normalizer=None, t_in=None): ...

def load_initial_tensor(data_dir, task): ...
```

### 3.3 HDF5 key 约定

Task 1：

```text
tensor, x-coordinate, t-coordinate
```

Task 2：

```text
tensor, x_coordinate, t_coordinate, nu
```

Task 3：

```text
# train/val
tensor, x-coordinate, t-coordinate, lambda2

# test
tensor, x-coordinate, t-coordinate
```

### 3.4 Task 3 数据形状

- `KS_train.hdf5`: `tensor` `(2000,400,256)`, `lambda2` `(2000,)`。
- `KS_val.hdf5`: `tensor` `(100,400,256)`, `lambda2` `(100,)`。
- `KS_test.hdf5`: `tensor` expected `(100,20,256)`，不含 `lambda2`。

代码应通过 `N = raw_test.shape[0]` 动态确定样本数，避免因本地小样例与正式测试集大小不一致而崩溃。

---

## 4. model.py — 模型定义

### 4.1 推荐主干

本项目推荐以 FNO 或 FNO-hybrid 作为默认主干，因为三项任务均是一维规则网格预测，频域卷积推理快，适合 120 秒推理限制。

提供三类模型模式：

- **Direct 模式**：一次性输出所有未来步。推理快，但长时稳定性弱。
- **Chunked 模式**：每次输出一个 chunk，自回归 rollout。更符合长时预测，但推理步数增加。
- **Conditioned Chunked 模式**：从输入窗口估计隐含物理参数，并用 FiLM / AdaLN 注入主干。

### 4.2 必须实现或支持的类

```python
class SpectralConv1d(nn.Module): ...
class FNOBlock1d(nn.Module): ...
class FiLM(nn.Module): ...

class DynamicsEncoder(nn.Module):
    """
    从短观测窗口估计隐含动力学参数或 latent code。
    Task 2 可估计 nu；Task 3 可估计 lambda2 或 latent dynamics embedding。
    输入 shape: [B, t_in, 256]
    输出 shape: [B, cond_dim]
    """

class FNOForecast1d(nn.Module):
    """输入 [B,t_in,Nx]，输出 [B,t_out,Nx]。"""

class ResidualFNO1d(FNOForecast1d): ...

class ChunkedFNO1d(nn.Module):
    def forward(self, x, cond=None): ...
    def rollout(self, x, horizon, cond=None, detach_between_chunks=False): ...
    @torch.no_grad()
    def rollout_no_grad(self, x, horizon, cond=None): ...

class KSChunkedFNO1d(ChunkedFNO1d):
    """
    Task 3 推荐模型。
    默认 t_in=20, horizon=380, chunk_size=10 或 20。
    内置 lambda2_encoder；训练时可用真实 lambda2 做条件，也可用 encoder 输出。
    推理时必须仅从 x 推断条件。
    """

def build_model(cfg, task="task1") -> nn.Module: ...
```

### 4.3 关键实现细节

1. 坐标通道：将 `linspace(0,1,Nx)` 作为额外 channel 拼接。
2. 残差输出：预测相对最后一帧的增量通常更稳定。
3. Task 2 / Task 3 条件化：训练时可用真实参数监督 encoder，但推理时不能读取测试参数。
4. Task 3 chunked rollout：输入窗口为 20，未来 horizon 为 380。
5. 避免过慢 ensemble：当前本地 Task 3 推理规模为 100 条轨迹 × 400 步 × 256 点；代码仍应把 rollout 次数和模型规模控制在足够快的范围内。

### 4.4 物理残差函数

```python
def burgers_residual(u, dx=1.0/256.0, dt=1.0, nu=1e-3):
    """u_t + u*u_x - nu*u_xx"""

def ks_residual(u, dx=1.0/256.0, dt=0.5, lambda2=None):
    """u_t + u*u_x + lambda2*u_xx + u_xxxx"""
```

Task 3 中四阶导数对噪声敏感，`ks_residual` 应只作为小权重辅助损失或诊断指标，不应主导训练。

---

## 5. utils.py — 评分、损失与工具

### 5.1 通用函数

```python
def set_seed(seed: int) -> None: ...
def compute_rel_mse(pred, gt, eps=1e-10) -> float: ...
def compute_rmse(pred, gt) -> float: ...
def compute_frechet_distance(u1, u2) -> float: ...
def save_hdf5(pred: np.ndarray, save_path: str) -> None: ...
def save_metrics(metrics: dict, save_path: str) -> None: ...
class Timer: ...
class Logger: ...
```

### 5.2 Task 1 / Task 2 分段评分

```python
def compute_segment_scores_burgers(pred, gt):
    """
    pred/gt shape: [B,190,256]
    Segment 1: [:48]
    Segment 2: [48:96]
    Segment 3: [96:]
    """
```

### 5.3 Task 3 分段评分

```python
def compute_segment_scores_ks(pred_future, gt_future):
    """
    pred_future/gt_future shape: [B,380,256]
    对应完整轨迹步 20-399。

    Segment 1: future[0:30]    -> full steps 20-49
    Segment 2: future[30:180]  -> full steps 50-199
    Segment 3: future[180:380] -> full steps 200-399
    total = 0.25*s1 + 0.25*s2 + 0.50*s3
    """
```

注意：Task 3 的第 2 段不是 `[30:200]`，而是 `[30:180]`，因为完整步 50–199 共 150 步；第 3 段从 future index 180 开始。

### 5.4 推荐辅助损失

```python
def spectral_energy_loss(pred, gt): ...
def spatial_gradient_loss(pred, gt): ...
def temporal_difference_loss(pred, gt): ...
def parameter_regression_loss(pred_param, true_param): ...
```

Task 3 可将谱能量损失用于长时统计段，使模型不仅追逐逐点 MSE，也保持合理的吸引子统计。

---

## 6. train.py — 训练入口

### 6.1 必须支持的参数

```python
--task {task1,task2,task3}
--data_dir / --data-dir
--output_dir / --output-dir
--model_type {direct,chunked}
--chunk_size
--epochs
--batch_size
--lr
--weight_decay
--modes
--width
--depth
--dropout
--scheduler
--patience
--seed
--num_workers
--device
--amp
--grad_clip
--t_in
--t_out
--use_film
--cond_dim
--use_param_loss
--param_loss_weight
--use_physics_loss
--physics_weight
--spectral_weight
--time_diff_weight
--grad_weight
--unroll_chunks
--resume
```

默认值建议：

- Task 1 / Task 2：`t_in=10`, `t_out=190`。
- Task 3：`t_in=20`, `t_out=380`。

### 6.2 训练主流程

```text
parse_args
→ set_seed
→ create output_dir
→ load train/val data
→ compute normalizer on training data
→ build_model(args, task=args.task)
→ optimizer / scheduler / AMP scaler
→ train_epoch + validate loop
→ save best_checkpoint.pt
→ write metrics.json and time.json
```

### 6.3 Task 3 训练要点

1. 训练时可读取 `lambda2`，但推理时不可读取。
2. 若使用 `lambda2_encoder`，训练时可加入参数监督损失。
3. validation 必须完整 rollout 380 步。
4. 不要仅用 1-step 或 chunk teacher forcing 指标作为最终选择标准。
5. 早停指标应使用 `compute_segment_scores_ks(...)["total"]`。
6. 训练时间虽不计精度分，但正式 session 仍需小于 12 小时。

### 6.4 训练伪代码

```python
def validate(model, loader, args, normalizer):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, cond = unpack(batch)
            x = x.to(args.device)
            y = y.to(args.device)
            if args.task == "task3":
                pred_norm = model.rollout(x, horizon=380, cond=cond_if_allowed)
                pred = normalizer.decode(pred_norm)
                gt = normalizer.decode(y)
                metrics = compute_segment_scores_ks(pred, gt)
            else:
                pred_norm = model.rollout(x, horizon=190, cond=cond_if_allowed)
                pred = normalizer.decode(pred_norm)
                gt = normalizer.decode(y)
                metrics = compute_segment_scores_burgers(pred, gt)
    return aggregate(metrics)
```

---

## 7. infer.py — 推理与提交生成

### 7.1 通用逻辑

```python
args = parse_args()
ckpt = torch.load(args.checkpoint, map_location=args.device)
cfg = rebuild_cfg_with_safe_defaults(ckpt["args"])
model = build_model(cfg, task=args.task).to(args.device)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()
normalizer = Normalizer.from_dict(ckpt["normalizer"])
loader, test_dataset = get_test_loader(args.data_dir, task=args.task, batch_size=args.batch_size, normalizer=normalizer)
raw_test = load_initial_tensor(args.data_dir, args.task)
```

### 7.2 Task 1 / Task 2 输出

```python
future = model.rollout(x, horizon=190, cond=None_or_estimated)
full = np.empty((N, 200, 256), dtype=np.float32)
full[:, :10, :] = raw_test[:, :10, :]
full[:, 10:, :] = future
assert np.allclose(full[:, :10, :], raw_test[:, :10, :], atol=1e-6)
save_hdf5(full, args.output)
```

### 7.3 Task 3 输出

```python
future = model.rollout(x, horizon=380, cond=None)  # cond must be inferred from x internally if needed
full = np.empty((N, 400, 256), dtype=np.float32)
full[:, :20, :] = raw_test[:, :20, :]
full[:, 20:, :] = future
assert np.allclose(full[:, :20, :], raw_test[:, :20, :], atol=1e-6)
save_hdf5(full, args.output)
```

**禁止行为**：Task 3 推理时不得读取测试 `lambda2`，不得从验证集选择与测试相近的轨迹拼接，不得调用数值求解器延拓测试轨迹。

---

## 8. 快速验证清单（Agent 自检用）

### 8.1 环境检查

```bash
python -c "import torch, h5py, numpy; print(torch.cuda.is_available())"
```

### 8.2 数据加载验证

```python
from code.dataset import get_dataloaders
loader, val_loader = get_dataloaders("./data/task3", task="task3", batch_size=2)
for batch in loader:
    print([x.shape if hasattr(x, "shape") else x for x in batch])
    break
```

Expected Task 3 shapes:

```text
x: [B,20,256]
y: [B,380,256] or full target [B,400,256]
lambda2: [B,1]
```

### 8.3 模型前向验证

```python
from code.model import KSChunkedFNO1d
model = KSChunkedFNO1d(modes=24, width=64, depth=4, t_in=20, chunk_size=10)
x = torch.randn(2, 20, 256)
pred = model.rollout(x, horizon=380)
assert pred.shape == (2, 380, 256)
```

### 8.4 训练一个 dummy epoch

```bash
python code/train.py --task task3 --epochs 1 --batch_size 2 --output_dir output/task3/test_run --data_dir ./data/task3
```

### 8.5 推理验证

```bash
python code/infer.py --task task3 --checkpoint output/task3/test_run/best_checkpoint.pt --output output/task3/test_run/task3_pred.hdf5 --data_dir ./data/task3
python -c "import h5py; f=h5py.File('output/task3/test_run/task3_pred.hdf5','r'); print(f['tensor'].shape); f.close()"
```

---

## 9. 常见失败模式与排查

| 现象 | 可能原因 | 修复方法 |
|------|---------|----------|
| `unrecognized arguments: --task task3` | CLI 未加入 task3 | `choices` 加入 `task3` |
| 找不到 KS 数据 | `data_dir` 默认仍指向旧目录或错误任务目录 | Task 3 默认路径改为 `./data/task3`，并确认 `KS_train.hdf5` / `KS_val.hdf5` / `KS_test.hdf5` 存在 |
| `KeyError: lambda2` 出现在测试集 | test loader 假设存在参数字段 | 测试集不读取 `lambda2` |
| Task 3 输出 shape 为 `(N,380,256)` | 忘记拼接前 20 步 | 保存前构造 `(N,400,256)` |
| 前 20 步不一致 | infer 中用模型重构了观测步 | 直接复制 `raw_test[:, :20, :]` |
| 验证 score 虚高 | validation 使用 teacher forcing | 必须完整 rollout 380 步 |
| 训练不稳定 / NaN | KS 四阶残差权重过大 | 降低 physics weight 或改用谱统计损失 |
| 推理超时 | chunk 太小或 ensemble 太多 | 增大 chunk_size、减小模型、批量推理 |

---

## 10. 推荐的首次实验配置

```bash
# Task 1 — Chunked FNO
python code/train.py \
  --task task1 \
  --model_type chunked \
  --chunk_size 10 \
  --modes 24 --width 64 --depth 4 \
  --epochs 220 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 \
  --scheduler cosine --patience 35 \
  --grad_weight 0.05 --time_diff_weight 0.02 \
  --augment_shift \
  --output_dir output/task1_baseline \
  --data_dir ./data/task1

# Task 2 — Chunked FNO with parameter encoder / FiLM
python code/train.py \
  --task task2 \
  --model_type chunked \
  --chunk_size 10 \
  --modes 24 --width 64 --depth 4 \
  --epochs 200 --batch_size 32 --lr 1e-3 \
  --use_film --use_param_loss \
  --output_dir output/task2_baseline \
  --data_dir ./data/task2

# Task 3 — KS Chunked FNO with lambda2 encoder
python code/train.py \
  --task task3 \
  --model_type chunked \
  --chunk_size 20 \
  --t_in 20 --t_out 380 \
  --modes 32 --width 96 --depth 4 \
  --epochs 160 --batch_size 16 --lr 8e-4 --weight_decay 1e-4 \
  --use_film --use_param_loss --param_loss_weight 0.05 \
  --spectral_weight 0.02 --time_diff_weight 0.02 \
  --scheduler cosine --patience 30 \
  --output_dir output/task3_baseline \
  --data_dir ./data/task3
```

这些配置只是第一轮 baseline。Agent 应根据显存、训练时间和验证结果自主调整。

---

## 11. 代码生成过程的日志记录要求

Agent 在编写代码时，应在日志中记录类似内容：

```text
[思考] 我计划为 Task 3 实现一个 KSChunkedFNO1d。
       该模型使用前 20 步观测估计 lambda2 或 latent dynamics code，
       并通过 FiLM 注入 FNO block。推理时不读取测试 lambda2。
[行动] 正在生成 code/model.py，包含 SpectralConv1d, FNOBlock1d,
       DynamicsEncoder, KSChunkedFNO1d 等类。
[验证] 我将检查 task3_pred.hdf5 shape 是否为当前本地期望的 (100,400,256)，
       并确认前 20 步与 KS_test.hdf5 完全一致。
```

日志必须能追溯 `code/` 中每个文件的生成过程。

---

## 12. 总结：Agent 编码执行路线图

```text
Step 1: 读取任务文档，确认当前 task 和数据路径
Step 2: inspect_hdf5 检查对应 HDF5 key/shape/dtype
Step 3: 生成 utils.py（评分、保存、计时、loss）
Step 4: 生成 dataset.py（Task1/2/3 数据加载与 normalizer）
Step 5: 生成 model.py（FNO、条件化、rollout）
Step 6: 生成 train.py（训练、验证、checkpoint）
Step 7: 生成 infer.py（加载 checkpoint、推理、保存 HDF5）
Step 8: dummy 训练和推理 smoke test
Step 9: 正式训练，保存 best_checkpoint.pt
Step 10: 正式推理，验证 shape 和前缀一致性
Step 11: 导出 time.csv 和 logs.log
Step 12: final_check 与 package_final
```

对于 Task 3，最重要的提交前检查是：

```text
task3_pred.hdf5/tensor shape == (N, 400, 256)
pred[:, :20, :] == KS_test/tensor[:, :20, :] within 1e-3
推理时间 < 120 秒
训练代码没有加载任何公开预训练权重或 Task1/2 checkpoint
日志解释了 KS 混沌、lambda2 未知处理和模型选择依据
```
