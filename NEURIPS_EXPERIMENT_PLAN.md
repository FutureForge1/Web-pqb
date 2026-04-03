# Web-PQB / Web-PRM NeurIPS 实验计划

## 1. 文档目的

这份文档的目标是把我们现在已经拥有的资产，系统地推进到一篇具备 `NeurIPS / 顶会投稿水准` 的论文实验方案。

这里的核心不是“再想更多 idea”，而是把当前已有的：

- 任务定义
- benchmark 构建脚本
- 标注系统
- 轨迹数据
- 本地与 AutoDL 的模型运行能力

组织成一条清晰、可信、可复现、可写进论文的证据链。

---

## 2. 我们现在已经有什么

### 2.1 已有资产

- 论文核心问题已经明确：
  - 现有 Web Agent 评测过于依赖 `Success Rate`
  - 缺少对过程质量的刻画
- 方法定义已经基本成形：
  - step-level `V_t`
  - step-level optimality indicator `y_t`
  - trajectory-level `O_t`
  - `TPQ`
- 已有 benchmark v1：
  - 基于 `VisualWebArena`
  - 三类任务：
    - `multi_path`
    - `high_distraction`
    - `recovery`
- 已有 benchmark v2 pipeline 雏形：
  - task card 生成
  - recovery candidate 生成
  - hard constraint filtering
  - page capture
  - VLM screening
  - review sheet 生成
- 已有轨迹数据：
  - `VWA GPT-4V + SoM`
  - `Mind2Web`
  - `Human Playwright`
- 已有人工标注与模型预标基础设施：
  - Streamlit 标注工具
  - VLM 预标脚本
  - `V_t / y_t / O_t` 递推逻辑
- 已有 AutoDL 多模态模型环境：
  - `Qwen2.5-VL-7B-Instruct`
  - `Qwen3-8B`

### 2.2 当前距离顶会还差什么

当前最缺的不是“idea”，而是下面四类证据：

- benchmark 的可信度：
  - 这些任务真的在测这三类过程挑战吗
- 标注与 judge 的可信度：
  - `V_t / y_t` 是否能被稳定标注和预测
- 指标有效性：
  - `TPQ` 是否比 `SR` 更符合人类判断
- 实用价值：
  - 新分数是否能真正帮助 agent 选择或 reranking

一句话说：

> 现在我们已经有“一个很好的研究方向 + 第一版系统”，但还没有形成“顶会级别的完整实验闭环”。

---

## 3. 顶会版本最终要回答的四个研究问题

整篇论文最好围绕四个核心 research questions 展开。

### RQ1. Benchmark validity

`Web-PQB` 中的三类任务，是否真的分别对应三种不同的过程挑战？

- `multi_path`：路径选择与最优性
- `high_distraction`：抗干扰与候选辨别
- `recovery`：状态识别、纠错与恢复

### RQ2. Human alignment

`TPQ / Web-PRM` 是否比 `Success Rate`、步数、简单启发式分数更贴近人类对“过程质量”的判断？

### RQ3. Judge quality

多模态 judge 是否能够学习并逼近人类对 `V_t / y_t` 的标注？

### RQ4. Utility

这个过程质量分数是否不仅能“评”，还能真正帮助：

- reranking
- best-of-k selection
- agent selection

---

## 4. 顶会水准的整体实验路线

我建议整个工作分成六个阶段推进。

## 阶段 A. Benchmark v2 构建与冻结

### A1. 生成统一 task card 层

**做什么**

把 `VisualWebArena` 原始任务统一转换成 `task card`。

**怎么做**

使用已经写好的脚本：

- [generate_benchmark_v2_task_cards.py](/home/lenovo/code/NIPS2026/scripts/generate_benchmark_v2_task_cards.py)

输出：

- `data/benchmark_tasks_v2/vwa_task_cards.jsonl`

**为什么做**

因为顶会版本的 benchmark 不能再直接依赖某个 agent 的轨迹来定义类别。  
我们需要一个 **task-centric** 的母集。

**当前状态**

这一步已经完成，已有 `910` 条 task cards。

---

### A2. 为 recovery 生成多 wrong-start 候选

**做什么**

不给每个 recovery task 只造一个 wrong start，而是给每个 base task 生成多个 wrong-start candidate。

**怎么做**

使用：

- [generate_benchmark_v2_recovery_candidates.py](/home/lenovo/code/NIPS2026/scripts/generate_benchmark_v2_recovery_candidates.py)

输出：

- `data/benchmark_tasks_v2/recovery_task_cards.jsonl`

**为什么做**

第三类是论文的亮点。  
如果每个 task 只拍脑袋指定一个 wrong start，会被质疑主观、偶然、不可复现。  
多候选生成 + 后续筛选，会让 recovery 设计更扎实。

**当前状态**

这一步已经完成，已有 `1629` 条 recovery 候选。

---

### A3. 做硬约束过滤

**做什么**

先用程序过滤掉明显不合法、不稳定、不值得审核的任务。

**怎么做**

使用：

- [filter_benchmark_v2_hard_constraints.py](/home/lenovo/code/NIPS2026/scripts/filter_benchmark_v2_hard_constraints.py)

当前过滤内容包括：

- `start_url` 是否有效
- URL 是否与站点匹配
- `eval` 是否存在
- `storage_state` 是否存在
- recovery 的 `wrong_start_url` 是否和原起点不同
- recovery 的 `wrong_start_url` 是否仍在同站点
- 去掉重复的 `intent + start_url`

输出：

- `data/benchmark_tasks_v2/task_cards_hard_filtered.jsonl`
- `data/benchmark_tasks_v2/hard_filter_report.json`

**为什么做**

不要浪费 VLM 和人工审核成本在明显坏样本上。  
这一步是 benchmark 构建中最重要的“客观合法性检查”。

**当前状态**

这一步已经完成：

- 总候选：`2539`
- 通过：`2300`

---

### A4. 抓真实页面证据

**做什么**

对每个 task 抓真实页面截图，而不是继续依赖 `gpt4v_som_910` 的旧轨迹截图。

**怎么做**

在本地/WSL 启动 `VWA/WebArena` 网站环境后，使用：

- [capture_benchmark_v2_pages.py](/home/lenovo/code/NIPS2026/scripts/capture_benchmark_v2_pages.py)

需要抓：

- 对 `multi_path / high_distraction`
  - `start_url`
- 对 `recovery`
  - `wrong_start_url`
  - `original_start_url`

建议输出：

- `page_capture_manifest.jsonl`
- `page_evidence/`

**为什么做**

如果 benchmark 最后仍依赖 `GPT-4V+SoM` 的历史截图，审稿人会质疑 benchmark 构建过程对单一 agent 轨迹有偏置。  
真实页面抓图能把 benchmark 从 “trajectory-assisted” 升级成 “environment-grounded”。

**当前状态**

脚本已经写好，但还没有在真实网站环境下全量运行。

---

### A5. 用 VLM 做结构化筛选

**做什么**

让多模态模型基于任务定义 + 真实页面截图输出结构化判断，而不是直接“拍板去留”。

**怎么做**

使用：

- [screen_benchmark_v2_with_vlm.py](/home/lenovo/code/NIPS2026/scripts/screen_benchmark_v2_with_vlm.py)

模型建议：

- `Qwen2.5-VL-7B-Instruct`

VLM 输出的不是最终结论，而是字段。

对 `multi_path`：

- `multi_path_valid`
- `route_plurality`
- `visual_dependence`

对 `high_distraction`：

- `distraction_visible`
- `distractor_density`
- `target_confusability`

对 `recovery`：

- `recovery_wrong_start_valid`
- `recovery_recoverable`
- `recovery_misleadingness`
- `recovery_answer_leakage`
- `recovery_severity`

程序再生成：

- `triage_label = keep / revise / drop`

**为什么做**

这是 benchmark 构建的关键改进点：

- 不让单个模型直接决定 benchmark
- 让模型只输出“维度判断”
- 最终 `keep/drop/revise` 由程序规则与人类共同决定

---

### A6. 行为验证（非常重要）

**做什么**

用 2 到 4 个不同 agent 在候选任务上跑一轮，验证这些任务是否真的诱发对应行为。

**怎么做**

对每类抽样：

- `multi_path`
- `high_distraction`
- `recovery`

每类先抽 `20-30` 题做 pilot。

要收集：

- success / failure
- step count
- detour
- repeated actions
- recovery 成功或失败

**为什么做**

这是从“看起来像 benchmark”到“行为上成立的 benchmark”的关键一步。  
NeurIPS 级 benchmark 通常不能只靠构造逻辑成立，还要证明它在行为上确实诱发了对应挑战。

**这一步对三类的意义**

- `multi_path`
  - 要看到明显不同路径
- `high_distraction`
  - 要看到更多 detour 和误点
- `recovery`
  - 要看到“偏航后恢复 / 恢复失败”的稳定现象

---

### A7. 人工精修并冻结 benchmark_v2

**做什么**

让人只审核边界样本，而不是审核全量。

**怎么做**

先用脚本生成 review sheet：

- [build_benchmark_v2_review_sheet.py](/home/lenovo/code/NIPS2026/scripts/build_benchmark_v2_review_sheet.py)

人工重点看：

- VLM 低置信度样本
- VLM 输出 `revise`
- `recovery`
- 行为验证失败样本
- 每类随机抽样质检样本

最终形成：

- `benchmark_v2_final.json`

**为什么做**

这样人类审核成本最小，但 benchmark 的最终责任仍然在人，而不是在模型。

---

## 阶段 B. 轨迹收集

### B1. 为最终 benchmark 收集多种轨迹

**做什么**

针对冻结后的 benchmark v2，每题收集多条轨迹，而不是只保留一个成功演示。

**怎么做**

每题至少收：

- 高质量成功轨迹
- 绕路成功轨迹
- 明显失败轨迹
- recovery 成功轨迹
- recovery 失败轨迹

来源可以是：

- 现有 `GPT-4V+SoM`
- 新跑的开源 agent
- 人类轨迹

**为什么做**

Web-PQB 的研究对象是“过程质量”，不是只有任务本身。  
没有丰富轨迹分布，后面的 `V_t / y_t` 标注会严重偏正样本。

---

## 阶段 C. 人工标注与金标集

### C1. 建立高质量 step-level 金标

**做什么**

构造最终用于训练 judge 的 `step-level gold set`。

**怎么做**

标注对象：

- `(goal, S_{t-1}, a_t, S_t)`

人工标：

- `V_t ∈ {-1, 0, +1}`
- `y_t ∈ {0, 1}`

程序递推：

- `O_t`
- `R_t`

建议规模：

- `8k - 15k` step

**为什么做**

如果没有足够可靠的人工金标，后面所有 judge 训练和 alignment claim 都站不住。

---

### C2. 建立 trajectory-level preference 金标

**做什么**

让人类直接比较两条轨迹哪条过程更好。

**怎么做**

同任务下抽轨迹对，标：

- 哪条过程更好
- 可选理由

建议规模：

- `500 - 1000` 对

**为什么做**

这一步是为了验证：

- `TPQ` 是否真的比 `Success Rate` 更符合人类过程偏好

---

## 阶段 D. Judge 训练与评估

### D1. 训练多模态 judge

**做什么**

训练或微调一个 step-level judge 去预测：

- `V_t`
- `y_t`

**怎么做**

先做三档：

- zero-shot VLM
- prompt-based Qwen2.5-VL
- 微调版 Web-PRM / VLM

输入：

- `goal`
- `before screenshot`
- `action`
- `after screenshot`

输出：

- `pred_V_t`
- `pred_y_t`

**为什么做**

这是你方法里 `Web-PRM` 的核心实验，不做出来论文就只剩 benchmark，没有 model story。

---

### D2. 比较不同 judge 变体

**做什么**

证明你的结构化定义比简单替代方案更合理。

**怎么做**

至少做这些 ablation：

- `V-only`
- `y-only`
- full `V + y -> O_t`
- text-only
- image-only
- image + task
- image + task + action

**为什么做**

NeurIPS 审稿人会非常关心：

- 你的设计是不是必要
- 有没有简单 baseline 就够了

---

## 阶段 E. 论文主实验

### E1. Benchmark validity

**做什么**

证明三类任务确实对应不同过程行为。

**怎么做**

比较三类任务上 agent 行为统计：

- detour rate
- repeated actions
- recovery event
- path diversity

**为什么做**

支撑 benchmark 的构造合理性。

---

### E2. Human alignment

**做什么**

验证 `TPQ` 是否更符合人类判断。

**怎么做**

比较：

- `Success Rate`
- step count
- simple heuristic
- trajectory-level LLM judge
- `TPQ`

指标：

- 与人类偏好的一致性
- Spearman / Pearson

**为什么做**

这是论文最关键的主 claim。

---

### E3. Same success, different process

**做什么**

证明“同样都成功”的轨迹里，`TPQ` 仍能区分路径质量。

**怎么做**

取同任务的成功轨迹，比较：

- 快速直达
- 绕路成功
- 偏航后恢复成功

**为什么做**

这是 process-quality 指标最有说服力的场景。

---

### E4. Recovery sensitivity

**做什么**

专门验证 `recovery` 类对过程质量的区分能力。

**怎么做**

比较：

- 错误起点后快速恢复
- 错误起点后长期迷失
- 错误起点后彻底失败

**为什么做**

第三类是你的亮点，不单独做强会很可惜。

---

### E5. Utility: reranking / best-of-k

**做什么**

证明这个分数不是“只能看”，而是“有用”。

**怎么做**

对每个任务生成 `k` 条候选轨迹，用：

- `TPQ`
- 或 learned judge score

去选最优轨迹。比较最终选中的：

- success
- human preference
- average TPQ

**为什么做**

这是让工作从“评测指标”升级成“对 agent 有用的信号”的关键。

---

## 阶段 F. 写作与投稿包装

### F1. 清晰界定贡献

论文里要明确三点贡献：

- 新 benchmark：`Web-PQB`
- 新 process-quality formalism：`V_t / y_t / O_t / TPQ`
- 新 judge / reward modeling setting：`Web-PRM`

### F2. 清晰区分现有工作

重点对比：

- WebArena / VisualWebArena
- trajectory-level agent judges
- general PRM / process supervision
- reward models for web agents

### F3. 把 benchmark 写成协议，不只是数据

要强调这不只是一个任务表，而是：

- task-centric construction
- page evidence capture
- hard-constraint filtering
- VLM screening
- human curation
- behavior validation

---

## 5. 推荐时间线

## 第 1 周

- 完成本地真实页面抓图小规模验证
- 跑通 `capture_benchmark_v2_pages.py`
- 修正 capture pipeline

## 第 2 周

- 全量抓取页面证据
- 跑 `screen_benchmark_v2_with_vlm.py`
- 生成人工 review sheet

## 第 3 周

- 完成 benchmark v2 精修
- 冻结 test/dev 集
- 开始收集最终轨迹池

## 第 4-5 周

- 进行 step-level 人工标注
- 建 trajectory preference 集

## 第 6 周

- 训练 judge
- 跑 step-level eval

## 第 7 周

- 跑 benchmark validity
- 跑 human alignment
- 跑 reranking / utility

## 第 8 周

- 整理图表
- 完成论文写作
- 做消融和补实验

---

## 6. 哪些是必须做的，哪些是加分项

### 必须做

- benchmark v2 冻结
- recovery 做扎实
- step-level gold set
- human alignment
- reranking / utility 至少一个

### 强烈建议做

- 多 agent 行为验证
- text-only vs VLM judge 对比
- 三类任务分开报结果

### 加分项

- online RL / DPO using process reward
- 外部 benchmark generalization
- 更大模型或更多 VLM baseline

---

## 7. 当前最应该立刻做什么

如果只看当前阶段，最值得立刻做的不是继续想新模型，而是：

1. 在本地/WSL 启动真实 `VWA/WebArena` 站点  
2. 跑 `capture_benchmark_v2_pages.py` 抓真实页面证据  
3. 把 `page_capture_manifest + screenshots` 传到 AutoDL  
4. 用 `Qwen2.5-VL-7B` 跑 `screen_benchmark_v2_with_vlm.py`  
5. 生成人工 review sheet 并开始精修 benchmark v2

一句话总结：

> 从现在到 NeurIPS 水准，最关键的不是再换模型，而是把 benchmark 做实、把金标做稳、把 `TPQ` 的有效性证明出来。

