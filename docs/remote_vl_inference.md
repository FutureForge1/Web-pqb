# 远程 VLM 预标使用说明

这套方案的目标是：

- 数据留在你本地电脑
- `Qwen2.5-VL-7B-Instruct` 跑在 AutoDL 的 GPU 上
- 本地脚本负责读轨迹、读截图、递推 `O_t`
- AutoDL 只负责做单步推理

## 1. 在 AutoDL 上启动推理服务

```bash
cd /root/autodl-tmp/Web-pqb

python scripts/vl_judge_server.py \
  --model-path /root/autodl-tmp/web-models/models/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8008 \
  --local-files-only
```

健康检查：

```bash
curl http://127.0.0.1:8008/health
```

## 2. 在本地把端口转回来

如果你是通过 SSH 连 AutoDL，常用做法是本地做端口转发：

```bash
ssh -L 8008:127.0.0.1:8008 root@你的-autodl-地址 -p 端口
```

这样你本地就能通过：

- `http://127.0.0.1:8008/predict`

访问 AutoDL 上的模型服务。

## 3. 在本地运行预标

```bash
cd /home/lenovo/code/NIPS2026

python scripts/prelabel_vl.py \
  --sources vwa_gpt4v_som \
  --remote-url http://127.0.0.1:8008/predict \
  --output predictions/qwen25vl7b_vwa_remote_debug.jsonl \
  --limit 20 \
  --resume
```

注意：

- 这时本地不需要模型权重
- 本地只需要自己的 `data/canonical` 和对应 raw 数据
- `O_t` 仍然在本地脚本里自动递推

## 4. 输出字段

输出 JSONL 里关键字段还是：

- `pred_V_t`
- `pred_y_t`
- `pred_O_t`
- `pred_R_t`
- `pred_confidence`
- `rationale_short`
- `cot_text`

其中：

- `pred_V_t` 对应你之前说的 `K`
- `pred_O_t` 是根据 `pred_y_t` 自动递推出来的

