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

脚本会检查必需函数、自导入错误和工作区状态；仅同步以下八个文件以及它自动生成的版本来源记录：

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

如果提示缺少 `region_resplit` 或 `interval_gate_loss`，服务器文件版本落后于 README 描述，或曾上传错文件名。先核对相应脚本，不要删除检查代码强行提交。

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
