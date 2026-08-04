# CRL_CW Cluster Launcher

简单的工具，用于将本地命令提交到 cluster 上运行。

## 快速开始

### 1. 配置 cluster 信息

编辑 `launcher/cluster_config.yaml`，填入你的 cluster 信息：

```yaml
cluster:
  host: your.cluster.edu
  user: your_username
  remote_dir: ~/crl_cw
  python_exec: /path/to/python
```

### 2. 测试命令（dry run）

先不带 `--submit` 运行，检查生成的命令是否正确：

```bash
python launcher/submit.py \
  "caffeinate -dimsu env PYTHONPATH=src python scripts/run.py --mode continual --method full_bc --seed 0"
```

这会打印出：
- rsync 同步命令
- 生成的 Slurm 脚本内容
- sbatch 提交命令

但**不会真正执行**。

### 3. 真正提交

确认无误后，加上 `--submit` 和 `--job-name`：

```bash
python launcher/submit.py \
  "caffeinate -dimsu env PYTHONPATH=src python scripts/run.py --mode continual --method full_bc --seed 0" \
  --submit \
  --job-name full_bc_seed0
```

这会：
1. rsync 同步整个项目到 cluster
2. 生成 Slurm 脚本并保存到 `scripts/slurm/full_bc_seed0.sh`
3. 通过 SSH 提交 sbatch job

## 使用示例

### 提交完整 CW10 实验

```bash
# Full BC seed 0
python launcher/submit.py \
  "env PYTHONPATH=src python scripts/run.py --mode continual --method full_bc --env-version v3 --reward-function-version cw10_v1 --steps-per-task 500000 --seed 0 --device cpu" \
  --submit --job-name full_bc_cw10_seed0

# ClonEx seed 1
python launcher/submit.py \
  "env PYTHONPATH=src python scripts/run.py --mode continual --method clonex_sac --env-version v3 --reward-function-version cw10_v1 --steps-per-task 500000 --seed 1 --device cpu" \
  --submit --job-name clonex_seed1
```

### 批量提交多个 seed

用 shell script：

```bash
#!/bin/bash
for seed in 0 1 2; do
  python launcher/submit.py \
    "env PYTHONPATH=src python scripts/run.py --mode continual --method full_bc --seed $seed --device cpu" \
    --submit --job-name full_bc_seed${seed}
  sleep 2  # 避免并发冲突
done
```

## 日志和结果下载

### 方式1：自动下载（推荐）

Job 完成后，用 fetch 脚本下载结果回本地：

```bash
# 下载所有结果和日志
.venv/bin/python launcher/fetch.py

# 下载特定 job 的结果
.venv/bin/python launcher/fetch.py --job-name full_bc_seed0

# 先预览要下载什么（dry run）
.venv/bin/python launcher/fetch.py --dry-run
```

### 方式2：手动查看（SSH）

```bash
# SSH 到 cluster
ssh your_username@your.cluster.edu

# 查典型工作流

```bash
# 1. 提交 job
.venv/bin/python launcher/submit.py \
  "env PYTHONPATH=src python scripts/run.py --mode continual --method full_bc --seed 0 --device cpu" \
  --submit --job-name full_bc_seed0

# 2. 检查状态（过一段时间）
.venv/bin/python launcher/fetch.py --status

# 3. Job 完成后下载结果
.venv/bin/python launcher/fetch.py --job-name full_bc_seed0

# 4. 分析本地结果
# outputs/ 和 logs/ 目录下已经有 cluster 的结果了
```

## 常见问题

**Q: 如何检查 job 状态？**

```bash
.venv/bin/python launcher/fetch.py --status
# 或手动 SSH
ssh your_username@your.cluster.edu 'squeue -u your_username'
```

**Q: 如何取消 job？**

```bash
ssh your_username@your.cluster.edu 'scancel JOB_ID'
```

**Q: 如何只下载特定目录？**

```bash
.venv/bin/python launcher/fetch.py --paths outputs/cw10_continual logs/cluster
```

**Q: 本地和 cluster Python 版本不同怎么办？**

在 `cluster_config.yaml` 里指定 cluster 的 Python 路径：
```yaml
python_exec: /home/user/anaconda3/envs/myenv/bin/python
```

**Q: 下载太慢怎么办？**

fetch 脚本使用 rsync，会自动跳过已存在且未修改的文件。如果文件很大，可以：
1. 只下载特定 job：`--job-name xxx`
2. 只下载日志先看结果：`--paths logs/cluster`
3. 手动 SSH 到 cluster 上压缩后再下载大文件

## 注意事项

1. **自动替换 Mac 命令**：脚本会自动移除 `caffeinate -dimsu`，并替换 python 路径
2. **rsync 排除规则**：在 `cluster_config.yaml` 中配置，避免同步不必要的文件
3. **时间限制**：默认 48 小时，在 config 里可以调整
4. **内存设置**：默认每核 8G，根据实际需求调整

## 常见问题

**Q: 如何检查 job 状态？**

```bash
ssh your_username@your.cluster.edu 'squeue -u your_username'
```

**Q: 如何取消 job？**

```bash
ssh your_username@your.cluster.edu 'scancel JOB_ID'
```

**Q: 本地和 cluster Python 版本不同怎么办？**

在 `cluster_config.yaml` 里指定 cluster 的 Python 路径：
```yaml
python_exec: /home/user/anaconda3/envs/myenv/bin/python
```
