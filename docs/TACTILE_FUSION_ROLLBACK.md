# LingBot V2 触觉融合回退说明

## 已冻结的 wrist-only 基线

```text
commit: 43ba7421feedaf6b7a08213096694e67be170c01
tag: tacthru-umi-v2-wrist-only-20260724
branch: backup/tacthru-umi-v2-wrist-only-20260724
```

触觉开发位于 `feature/tactile-fusion-v1`。旧 dataset、norm、checkpoint、protocol v1
和 `scripts/real_insert_ethernet.sh` 都没有被触觉入口覆盖。

## 最安全的代码回退：独立 worktree

保留当前触觉工作目录，同时建立一个只包含基线的运行目录：

```bash
cd /mnt/models/VTLA-RDT/lingbot-vla-v2

git worktree add ../lingbot-vla-v2-wrist-only \
  tacthru-umi-v2-wrist-only-20260724
```

之后从 `../lingbot-vla-v2-wrist-only` 启动旧服务即可，不需要删除或覆盖触觉代码。

## 在当前目录切换

先确认没有未保存改动：

```bash
git status --short
git switch backup/tacthru-umi-v2-wrist-only-20260724
```

不要使用 `git reset --hard`。若触觉分支已经产生提交，优先切换分支或对指定提交
执行 `git revert`。

## 文件级定向快照

创建快照：

```bash
bash scripts/create_tactile_fusion_snapshot.sh
```

快照保存在 `.rollback/tactile_fusion_<timestamp>/`，其中的 `restore.sh` 默认只预览：

```bash
bash .rollback/tactile_fusion_<timestamp>/restore.sh --dry-run
```

逐项检查输出后，才可显式应用：

```bash
bash .rollback/tactile_fusion_<timestamp>/restore.sh --force
```

该工具只处理 `scripts/tactile_fusion_paths.txt` 中列出的触觉融合文件，恢复 tag 中的
基线内容，并删除基线中不存在的触觉新文件。快照同时记录创建时的 working-tree SHA；
如果某个文件后来又被人工修改，普通 `--force` 会拒绝整次恢复，不会部分覆盖：

```bash
bash .rollback/tactile_fusion_<timestamp>/restore.sh --force
```

只有逐项审阅冲突后，才可使用显式破坏性选项：

```bash
bash .rollback/tactile_fusion_<timestamp>/restore.sh --force-conflicts
```

Git tag/backup branch 仍是第一层回退，文件级快照只是脏工作树下的第二层保障。

## 运行时回退

1. 停止 tactile v2 客户端和 18082 服务。
2. 使用旧 wrist-only checkpoint 与旧 norm stats 启动原服务：

```bash
bash scripts/run_tacthru_umi_v2_server.sh \
  --host 127.0.0.1 \
  --port 18081 \
  --use-compile \
  --warmup
```

3. 使用原 `scripts/real_insert_ethernet.sh`，不要使用 tactile 真机脚本。
4. 检查 health 的 checkpoint、norm、protocol v1 和 `[50,8]` action contract。
5. 重新做 synthetic 和 dry-run，通过后才恢复真机执行。

真正回退必须成套切换：

```text
旧 wrist-only checkpoint
+ 旧 wrist-only norm stats
+ protocol v1 server/client
+ 原 real_insert_ethernet.sh
```

仅关闭 `force_mask` 或把触觉开关改成 false，不能把已经训练好的 tactile checkpoint
变成 wrist-only checkpoint。
