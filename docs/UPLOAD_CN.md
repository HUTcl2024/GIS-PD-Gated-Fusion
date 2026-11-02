# 将服务器代码同步到 GitHub

目标仓库：https://github.com/HUTcl2024/GIS-PD-Gated-Fusion

## 当前快照的来源

仓库初始版本来自本机保存的开发代码副本。由于配置的服务器 SSH 连接超时，不能据此声称已完整核对服务器版本。`SOURCE_PROVENANCE.json` 保存八个源文件的 SHA256；完成下面的服务器同步后，它会标记为 `server_verified: true`。这里的 verified 仅表示从服务器指定文件复制并记录哈希，不表示完成科学实验验证。

仓库统一使用 LF 换行。版本记录中的 `source_sha256` 是原始文件字节哈希，`sha256` 是统一 LF 后的仓库文件哈希；仅换行差异不代表网络或算法变化。

## 在服务器操作

不要在包含原始数据的 `/home/aa/cl-2` 中直接执行 `git add .`。将仓库放在独立目录：

```bash
cd /home/aa
git clone https://github.com/HUTcl2024/GIS-PD-Gated-Fusion.git
cd GIS-PD-Gated-Fusion

# 第一步仅核对源文件哈希，不复制、不提交。
python -B scripts/sync_from_server.py --source /home/aa/cl-2

# 第二步复制明确列出的代码，提交并推送。
python -B scripts/sync_from_server.py \
  --source /home/aa/cl-2 --apply --commit --push
```

仓库已存在时，先进入该独立仓库并在工作区干净时执行 `git pull --ff-only`，无需再次 clone。

脚本会检查核心函数、自导入错误和工作区状态；同步以下代码及版本来源记录，其中 `revision_temporal.py` 是可选文件，服务器没有时不会阻止同步：

```text
revision_data.py
revision_train.py
revision_reliability.py
revision_temporal.py
generate_prpd_wfm.py
mso44_wfm.py
mso44_csv.py
recovered_models.py
```

新增功能不是同步的前置条件。旧版服务器文件没有 `region_resplit` 或 `interval_gate_loss` 也可以原样同步；脚本在终端和 `SOURCE_PROVENANCE.json` 的 `capabilities` 中记录可用功能。此时 README 中相应新命令可能不适用，但不要为上传代码而覆盖正在使用的实验脚本。可选时序文件不在服务器上时，仓库中的原副本会保留，并明确记录它没有经过服务器核对。

如果报错缺少核心函数，错误会列出具体名称；这可能仍是上传错文件的问题。不能关闭所有检查，因为之前确实发生过同名脚本内容混淆、自导入的情况。

如果提示工作区有未提交改动，先用 `git status` 检查并处理自己的修改；脚本不会替你丢弃文件。若 push 提示认证失败，使用服务器已有的 GitHub 登录或 SSH 配置；也可以在已安装 GitHub CLI 的环境运行 `gh auth login`。不要将密码、令牌或私钥发到聊天中。

## 哪些内容没有上传

- 原始 WFM、CSV、PRPD 数据图像和数据压缩包。
- `data/`、`first_paper_data/`、`revision_output*/`、`runs/`。
- 模型权重、训练日志、预测文件、私人记录清单。
- 论文、审稿意见、回复信和个人电脑配置。

本仓库采用明确的代码白名单和 `.gitignore`，同步脚本不递归复制实验目录。

## README 与实验结果

README 参考 Fourier Angle Alignment 的章节组织，内容根据当前 GIS 局放代码重写。没有复制对方的方法图、论文成果、会议信息或许可证。

旧表结果是 **划分 seed42、训练 seed3407、158条验证记录**。最新要求的 **划分 seed3407** 尚未提供重跑结果，README 明确写为 pending。换划分之后须重拟合训练参考幅值、重新绘图和重训；不能直接沿用旧表数字。

当前仓库没有发布数据、预训练模型或最终论文 DOI。代码公开不等于实验已经得到独立复现。
