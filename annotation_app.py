"""
Web-PQB 标注平台
键盘快捷键: 1/2/3 → V_t (+1/0/-1), Q/W → y_t (最优/非最优)
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
import io
import base64

from canonical_data import available_canonical_sources, load_canonical_steps, load_step_image

# ── 路径配置 ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/lenovo/code/NIPS2026/data")
ANNO_DIR = Path("/home/lenovo/code/NIPS2026/annotations")
ANNO_DIR.mkdir(exist_ok=True)
GAMMA_PENALTY = 0.5

# ── Demo 数据（无真实数据时使用）────────────────────────────────────────────────
DEMO_STEPS = [
    {
        "step_id": "demo_001",
        "task_id": "demo_task_1",
        "task_goal": "在购物网站上搜索「黑色 M 码 T 恤」并加入购物车",
        "action": "click(element='搜索框')",
        "action_desc": "点击页面顶部的搜索输入框",
        "screenshot_before": None,
        "screenshot_after": None,
    },
    {
        "step_id": "demo_002",
        "task_id": "demo_task_1",
        "task_goal": "在购物网站上搜索「黑色 M 码 T 恤」并加入购物车",
        "action": "type(text='黑色 T 恤')",
        "action_desc": "在搜索框中输入「黑色 T 恤」",
        "screenshot_before": None,
        "screenshot_after": None,
    },
    {
        "step_id": "demo_003",
        "task_id": "demo_task_1",
        "task_goal": "在购物网站上搜索「黑色 M 码 T 恤」并加入购物车",
        "action": "click(element='促销广告横幅')",
        "action_desc": "点击了页面上的促销广告，跳转到无关页面",
        "screenshot_before": None,
        "screenshot_after": None,
    },
    {
        "step_id": "demo_004",
        "task_id": "demo_task_2",
        "task_goal": "预订明天入住的单人间酒店，价格低于 500 元",
        "action": "click(element='返回按钮')",
        "action_desc": "误入错误页面后，点击浏览器返回按钮",
        "screenshot_before": None,
        "screenshot_after": None,
    },
    {
        "step_id": "demo_005",
        "task_id": "demo_task_2",
        "task_goal": "预订明天入住的单人间酒店，价格低于 500 元",
        "action": "scroll(direction='down', amount=3)",
        "action_desc": "在酒店列表页向下滚动查看更多选项",
        "screenshot_before": None,
        "screenshot_after": None,
    },
]


def make_placeholder_image(label: str, color: tuple, size=(640, 480)) -> Image.Image:
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    # 画边框
    draw.rectangle([2, 2, size[0]-3, size[1]-3], outline=(180, 180, 180), width=2)
    # 文字居中
    text = label
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size[0]-tw)//2, (size[1]-th)//2), text, fill=(120, 120, 120))
    return img


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_steps_from_jsonl(jsonl_path: Path) -> list:
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def discover_data() -> list:
    """扫描 data 目录，加载所有 JSONL 轨迹文件中的步骤。"""
    steps = []
    for jsonl_file in DATA_DIR.rglob("*.jsonl"):
        try:
            loaded = load_steps_from_jsonl(jsonl_file)
            steps.extend(loaded)
        except Exception:
            pass
    return steps


def load_real_steps(source_keys: tuple[str, ...]) -> list:
    return load_canonical_steps(source_keys)


def load_existing_annotations(anno_file: Path) -> dict:
    """返回 {step_id: annotation_dict}"""
    result = {}
    if anno_file.exists():
        with open(anno_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        result[rec["step_id"]] = rec
                    except Exception:
                        pass
    return result


def save_annotation(anno_file: Path, record: dict):
    with open(anno_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_anno_file(annotator_id: str) -> Path:
    safe = annotator_id.replace(" ", "_").replace("/", "_")
    return ANNO_DIR / f"annotations_{safe}.jsonl"


def get_annotation_v(record: dict | None) -> int | None:
    if not record:
        return None
    value = record.get("V_t")
    return int(value) if value is not None else None


def get_annotation_y(record: dict | None) -> int | None:
    if not record:
        return None
    if record.get("y_t") is not None:
        return int(record["y_t"])
    legacy_o = record.get("O_t")
    if legacy_o is None:
        return None
    try:
        return 1 if float(legacy_o) == 1.0 else 0
    except Exception:
        return None


def build_step_lookup(steps: list[dict]) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for item in steps:
        trajectory_id = item.get("trajectory_id", item.get("task_id"))
        step_idx = item.get("step_idx")
        step_id = item.get("step_id")
        if trajectory_id is None or step_idx is None or step_id is None:
            continue
        lookup[(trajectory_id, int(step_idx))] = step_id
    return lookup


def compute_prev_o(
    step: dict,
    annotations: dict[str, dict],
    step_lookup: dict[tuple[str, int], str],
) -> tuple[float | None, str | None]:
    trajectory_id = step.get("trajectory_id", step.get("task_id"))
    step_idx = step.get("step_idx")
    if trajectory_id is None or step_idx is None:
        return 1.0, None

    step_idx = int(step_idx)
    current_o = 1.0
    for prev_idx in range(step_idx):
        prev_step_id = step_lookup.get((trajectory_id, prev_idx))
        if prev_step_id is None:
            return None, None
        prev_record = annotations.get(prev_step_id)
        prev_y = get_annotation_y(prev_record)
        if prev_y is None:
            return None, prev_step_id
        current_o = 1.0 if prev_y == 1 else GAMMA_PENALTY * current_o
    return current_o, None


def compute_current_o(prev_o: float | None, y_val: int | None) -> float | None:
    if prev_o is None or y_val is None:
        return None
    return 1.0 if y_val == 1 else GAMMA_PENALTY * prev_o


# ── Streamlit 页面配置 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Web-PQB 标注平台",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS：隐藏 Streamlit 默认元素，美化界面 ────────────────────────────────────
st.markdown("""
<style>
/* 整体背景 */
.stApp { background-color: #f8f9fa; }

/* 标注卡片 */
.anno-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

/* 快捷键按钮样式 */
.key-btn {
    display: inline-block;
    background: #e9ecef;
    border: 1px solid #ced4da;
    border-radius: 6px;
    padding: 4px 10px;
    font-family: monospace;
    font-size: 14px;
    font-weight: bold;
    margin: 2px;
    cursor: pointer;
}
.key-btn.active-v { background: #d4edda; border-color: #28a745; color: #155724; }
.key-btn.active-o { background: #cce5ff; border-color: #004085; color: #004085; }

/* 进度条文字 */
.progress-text { font-size: 13px; color: #6c757d; }

/* 截图标题 */
.img-label {
    text-align: center;
    font-size: 12px;
    color: #6c757d;
    margin-bottom: 4px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* 动作描述框 */
.action-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 14px;
}

/* 任务目标框 */
.goal-box {
    background: #e8f4f8;
    border-left: 4px solid #17a2b8;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 14px;
}

/* 已标注标记 */
.badge-done { color: #28a745; font-weight: bold; }
.badge-skip { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# ── 键盘快捷键 JS ──────────────────────────────────────────────────────────────
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    const key = e.key.toLowerCase();
    // 防止在输入框中触发
    if (document.activeElement.tagName === 'INPUT' ||
        document.activeElement.tagName === 'TEXTAREA') return;

    const vMap = {'1': '+1', '2': '0', '3': '-1'};
    const yMap = {'q': '1', 'w': '0', 'e': '0'};

    if (vMap[key]) {
        // 找到对应的 radio button 并点击
        const radios = document.querySelectorAll('input[type="radio"]');
        radios.forEach(r => {
            if (r.value === vMap[key] && r.name && r.name.includes('V_t')) {
                r.click();
            }
        });
    }
    if (yMap[key]) {
        const radios = document.querySelectorAll('input[type="radio"]');
        radios.forEach(r => {
            if (r.value === yMap[key] && r.name && r.name.includes('y_t')) {
                r.click();
            }
        });
    }
    // 回车 = 提交
    if (key === 'enter') {
        const btns = document.querySelectorAll('button');
        btns.forEach(b => {
            if (b.innerText.includes('提交') || b.innerText.includes('Submit')) {
                b.click();
            }
        });
    }
    // 左右箭头 = 上一步/下一步
    if (key === 'arrowleft') {
        const btns = document.querySelectorAll('button');
        btns.forEach(b => { if (b.innerText.includes('上一步')) b.click(); });
    }
    if (key === 'arrowright') {
        const btns = document.querySelectorAll('button');
        btns.forEach(b => { if (b.innerText.includes('下一步')) b.click(); });
    }
});
</script>
""", unsafe_allow_html=True)

# ── Session State 初始化 ───────────────────────────────────────────────────────
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "v_val" not in st.session_state:
    st.session_state.v_val = None
if "y_val" not in st.session_state:
    st.session_state.y_val = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "steps" not in st.session_state:
    st.session_state.steps = None
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 设置")

    annotator_id = st.text_input("标注者 ID", value="annotator_01", key="annotator_id")
    anno_file = get_anno_file(annotator_id)

    st.divider()

    # 数据加载
    st.subheader("数据源")
    canonical_sources = available_canonical_sources()
    canonical_labels = {info["label"]: info["key"] for info in canonical_sources}
    data_options = ["Canonical parquet（真实数据）", "自动扫描 data/", "上传 JSONL 文件", "Demo 演示模式"]
    data_mode = st.radio(
        "选择数据模式",
        data_options,
        index=0 if canonical_sources else 3,
    )

    if data_mode == "Canonical parquet（真实数据）":
        if canonical_sources:
            selected_labels = st.multiselect(
                "选择真实数据源",
                options=list(canonical_labels.keys()),
                default=list(canonical_labels.keys()),
            )

            for info in canonical_sources:
                st.caption(
                    f"{info['label']}: {info['num_trajectories']} 条轨迹 / {info['num_steps']} 步"
                )

            if st.button("📦 加载真实数据"):
                selected_keys = tuple(canonical_labels[label] for label in selected_labels)
                steps = load_real_steps(selected_keys)
                if steps:
                    st.session_state.steps = steps
                    st.session_state.demo_mode = False
                    st.session_state.step_idx = 0
                    st.session_state.annotations = load_existing_annotations(anno_file)
                    st.success(f"加载 {len(steps)} 个真实 step")
                else:
                    st.warning("没有选中可用数据源，或对应 parquet 为空。")
        else:
            st.warning("尚未发现 canonical parquet，请先运行转换脚本。")

    elif data_mode == "自动扫描 data/":
        if st.button("🔄 扫描并加载"):
            steps = discover_data()
            if steps:
                st.session_state.steps = steps
                st.session_state.demo_mode = False
                st.session_state.step_idx = 0
                st.session_state.annotations = load_existing_annotations(anno_file)
                st.success(f"加载 {len(steps)} 步")
            else:
                st.warning("未找到 JSONL 文件，切换到 Demo 模式")
                st.session_state.steps = DEMO_STEPS
                st.session_state.demo_mode = True

    elif data_mode == "上传 JSONL 文件":
        uploaded = st.file_uploader("上传轨迹 JSONL", type=["jsonl", "json"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            steps = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        steps.append(json.loads(line))
                    except Exception:
                        pass
            if steps:
                st.session_state.steps = steps
                st.session_state.demo_mode = False
                st.session_state.step_idx = 0
                st.session_state.annotations = load_existing_annotations(anno_file)
                st.success(f"加载 {len(steps)} 步")

    else:  # Demo
        if st.session_state.steps is None or st.session_state.demo_mode:
            st.session_state.steps = DEMO_STEPS
            st.session_state.demo_mode = True
            st.session_state.annotations = load_existing_annotations(anno_file)
        if st.button("🎮 加载 Demo"):
            st.session_state.steps = DEMO_STEPS
            st.session_state.demo_mode = True
            st.session_state.step_idx = 0
            st.session_state.annotations = load_existing_annotations(anno_file)

    st.divider()

    # 快捷键说明
    st.subheader("⌨️ 快捷键")
    st.markdown("""
| 键 | 功能 |
|---|---|
| `1` | V = +1 推进 |
| `2` | V = 0 中性 |
| `3` | V = -1 倒退 |
| `Q` | y = 1 最优/有效恢复 |
| `W` / `E` | y = 0 非最优/偏航 |
| `Enter` | 提交标注 |
| `←` / `→` | 上/下一步 |
""")
    st.caption("自动递推规则：O₀=1.0；若 yₜ=1，则 Oₜ=1.0；若 yₜ=0，则 Oₜ=0.5×Oₜ₋₁。")

    st.divider()

    # 统计
    if st.session_state.steps:
        total = len(st.session_state.steps)
        done = len(st.session_state.annotations)
        st.metric("总步数", total)
        st.metric("已标注", done)
        st.metric("剩余", total - done)
        st.progress(done / total if total > 0 else 0)

    st.divider()
    st.caption(f"保存路径: `{anno_file}`")

# ── 主界面 ────────────────────────────────────────────────────────────────────
st.title("🖥️ Web-PQB 标注平台")

# 确保有数据
if st.session_state.steps is None:
    canonical_sources = available_canonical_sources()
    st.session_state.annotations = load_existing_annotations(get_anno_file(
        st.session_state.get("annotator_id", "annotator_01")
    ))
    if canonical_sources:
        default_keys = tuple(info["key"] for info in canonical_sources)
        st.session_state.steps = load_real_steps(default_keys)
        st.session_state.demo_mode = False
    else:
        st.session_state.steps = DEMO_STEPS
        st.session_state.demo_mode = True

steps = st.session_state.steps
if not steps:
    st.warning("没有可标注的数据，请在侧边栏加载数据。")
    st.stop()

# Demo 模式提示
if st.session_state.demo_mode:
    st.info("🎮 **Demo 模式** — 使用示例数据演示标注流程。真实数据请在侧边栏切换数据源。")

# 当前步骤
idx = st.session_state.step_idx
step = steps[idx]
step_id = step.get("step_id", f"step_{idx:04d}")
total = len(steps)
step_lookup = build_step_lookup(steps)

# ── 进度条 ────────────────────────────────────────────────────────────────────
col_prog, col_nav = st.columns([3, 1])
with col_prog:
    st.progress((idx + 1) / total)
    done_count = len(st.session_state.annotations)
    st.markdown(
        f'<span class="progress-text">步骤 {idx+1} / {total} &nbsp;|&nbsp; '
        f'已标注 {done_count} 步 &nbsp;|&nbsp; '
        f'任务: <b>{step.get("task_id", "—")}</b></span>',
        unsafe_allow_html=True,
    )

with col_nav:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("◀ 上一步", disabled=(idx == 0)):
            st.session_state.step_idx = max(0, idx - 1)
            st.session_state.v_val = None
            st.session_state.y_val = None
            st.session_state.submitted = False
            st.rerun()
    with c2:
        # 跳转输入
        jump = st.number_input("跳转", min_value=1, max_value=total, value=idx+1,
                                label_visibility="collapsed")
        if jump - 1 != idx:
            st.session_state.step_idx = jump - 1
            st.session_state.v_val = None
            st.session_state.y_val = None
            st.session_state.submitted = False
            st.rerun()
    with c3:
        if st.button("下一步 ▶", disabled=(idx == total - 1)):
            st.session_state.step_idx = min(total - 1, idx + 1)
            st.session_state.v_val = None
            st.session_state.y_val = None
            st.session_state.submitted = False
            st.rerun()

st.divider()

# ── 三栏布局：截图前 | 任务+动作 | 截图后 ─────────────────────────────────────
col_before, col_center, col_after = st.columns([2, 1.5, 2])

# 加载截图
def load_screenshot(step_data: dict, which: str, label: str, color: tuple) -> Image.Image:
    image = load_step_image(step_data, which)
    if image is not None:
        return image

    fallback_key = "screenshot_before" if which == "before" else "screenshot_after"
    fallback_path = step_data.get(fallback_key)
    if fallback_path and Path(fallback_path).exists():
        try:
            return Image.open(fallback_path).convert("RGB")
        except Exception:
            pass
    return make_placeholder_image(label, color)

img_before = load_screenshot(
    step,
    "before",
    f"St-1\n(动作前截图)\n\n步骤 {idx+1}",
    (240, 245, 250),
)
img_after = load_screenshot(
    step,
    "after",
    f"St\n(动作后截图)\n\n步骤 {idx+1}",
    (245, 250, 240),
)

with col_before:
    st.markdown('<div class="img-label">St-1 &nbsp; 动作前</div>', unsafe_allow_html=True)
    st.image(img_before, use_container_width=True)

with col_center:
    st.markdown('<div class="anno-card">', unsafe_allow_html=True)

    source_label = step.get("source_label") or step.get("dataset_source") or "unknown"
    meta_parts = [
        f"来源: {source_label}",
        f"轨迹: {step.get('task_id', '—')}",
    ]
    if step.get("website"):
        meta_parts.append(f"站点: {step['website']}")
    if step.get("domain"):
        meta_parts.append(f"域名: {step['domain']}")
    st.caption(" | ".join(meta_parts))

    # 任务目标
    st.markdown("**🎯 任务目标**")
    st.markdown(
        f'<div class="goal-box">{step.get("task_goal", "（无任务描述）")}</div>',
        unsafe_allow_html=True,
    )
    if step.get("needs_goal_mapping"):
        st.warning("这条轨迹暂时没有任务目标映射。建议优先依据前后截图变化和动作本身进行标注。")

    # 执行动作
    st.markdown("**⚡ 执行动作**")
    action_text = step.get("action", "（无动作）")
    action_desc = step.get("action_desc", "")
    st.code(action_text, language=None)
    if action_desc:
        st.markdown(
            f'<div class="action-box">{action_desc}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 标注区域 ──────────────────────────────────────────────────────────────
    st.divider()

    # 检查是否已标注
    existing = st.session_state.annotations.get(step_id)
    prev_o, missing_prev_step_id = compute_prev_o(step, st.session_state.annotations, step_lookup)
    existing_v = get_annotation_v(existing)
    existing_y = get_annotation_y(existing)
    existing_o = compute_current_o(prev_o, existing_y) if existing_y is not None and prev_o is not None else None
    if existing:
        st.markdown(
            f'<span class="badge-done">✅ 已标注</span> &nbsp; '
            f'V={f"{existing_v:+d}" if existing_v is not None else "—"} &nbsp; '
            f'y={existing_y if existing_y is not None else "—"} &nbsp; '
            f'O={f"{existing_o:.3f}" if existing_o is not None else "待递推"}',
            unsafe_allow_html=True,
        )
        default_v = str(existing_v) if existing_v is not None else None
        default_y = str(existing_y) if existing_y is not None else None
    else:
        default_v = None
        default_y = None

    # V_t 选择
    st.markdown("**第一题：V_t（这步推进了吗？）**")
    v_options = {"+1 推进 [按 1]": "+1", "0 中性 [按 2]": "0", "-1 倒退 [按 3]": "-1"}
    v_labels = list(v_options.keys())
    v_values = list(v_options.values())

    v_default_idx = v_values.index(default_v) if default_v in v_values else None
    v_sel = st.radio(
        "V_t",
        v_labels,
        index=v_default_idx,
        key=f"V_t_{step_id}",
        label_visibility="collapsed",
        horizontal=False,
    )
    v_val = v_options.get(v_sel) if v_sel else None

    st.markdown("**第二题：y_t（这步属于高质量主路径或有效恢复动作吗？）**")
    y_options = {"1 最优 / 有效恢复 [按 Q]": "1", "0 非最优 / 偏航 [按 W 或 E]": "0"}
    y_labels = list(y_options.keys())
    y_values = list(y_options.values())

    y_default_idx = y_values.index(default_y) if default_y in y_values else None
    y_sel = st.radio(
        "y_t",
        y_labels,
        index=y_default_idx,
        key=f"y_t_{step_id}",
        label_visibility="collapsed",
        horizontal=False,
    )
    y_val = y_options.get(y_sel) if y_sel else None

    if missing_prev_step_id:
        st.warning(
            f"当前步的前序轨迹步骤 `{missing_prev_step_id}` 还没有完成有效标注，"
            "因此暂时无法精确递推当前 O_t。建议先把同一条轨迹前面的步骤补齐。"
        )

    if prev_o is not None:
        st.caption(f"当前轨迹递推基线：O_(t-1) = {prev_o:.3f}，固定衰减系数 gamma = {GAMMA_PENALTY}")

    # 备注
    note = st.text_input(
        "备注（可选）",
        value=existing.get("note", "") if existing else "",
        key=f"note_{step_id}",
        placeholder="有疑问？记录在这里...",
    )

    v_num = int(v_val) if v_val is not None else None
    y_num = int(y_val) if y_val is not None else None
    current_o = compute_current_o(prev_o, y_num)
    current_r = (v_num * current_o) if (v_num is not None and current_o is not None) else None

    # 提交按钮
    col_sub, col_skip = st.columns(2)
    with col_sub:
        submit_disabled = (v_num is None or y_num is None or current_o is None)
        if st.button("✅ 提交 [Enter]", disabled=submit_disabled, type="primary", use_container_width=True):
            record = {
                "step_id": step_id,
                "task_id": step.get("task_id", ""),
                "source": step.get("dataset_source", ""),
                "trajectory_id": step.get("trajectory_id", step.get("task_id", "")),
                "step_idx": step.get("step_idx"),
                "annotator_id": annotator_id,
                "gamma_penalty": GAMMA_PENALTY,
                "V_t": v_num,
                "y_t": y_num,
                "O_prev": prev_o,
                "O_t": current_o,
                "R_t": current_r,
                "note": note,
                "timestamp": datetime.now().isoformat(),
            }
            st.session_state.annotations[step_id] = record
            save_annotation(anno_file, record)
            st.session_state.submitted = True

            # 自动跳到下一步
            if idx < total - 1:
                st.session_state.step_idx = idx + 1
                st.session_state.v_val = None
                st.session_state.y_val = None
                st.session_state.submitted = False
            st.rerun()

    with col_skip:
        if st.button("⏭ 跳过", use_container_width=True):
            record = {
                "step_id": step_id,
                "task_id": step.get("task_id", ""),
                "source": step.get("dataset_source", ""),
                "trajectory_id": step.get("trajectory_id", step.get("task_id", "")),
                "step_idx": step.get("step_idx"),
                "annotator_id": annotator_id,
                "gamma_penalty": GAMMA_PENALTY,
                "V_t": None,
                "y_t": None,
                "O_prev": prev_o,
                "O_t": None,
                "R_t": None,
                "note": "SKIPPED",
                "timestamp": datetime.now().isoformat(),
            }
            st.session_state.annotations[step_id] = record
            save_annotation(anno_file, record)
            if idx < total - 1:
                st.session_state.step_idx = idx + 1
                st.rerun()

    # 实时 R_t 预览
    if current_o is not None:
        st.markdown(
            f'<div style="text-align:center; margin-top:8px; font-size:15px; color:#495057;">'
            f'自动递推 O_t = {current_o:.3f}</div>',
            unsafe_allow_html=True,
        )
    if current_r is not None:
        color = "#28a745" if current_r > 0 else ("#dc3545" if current_r < 0 else "#6c757d")
        st.markdown(
            f'<div style="text-align:center; margin-top:8px; font-size:18px; font-weight:bold; color:{color};">'
            f'R_t = {current_r:+.3f}</div>',
            unsafe_allow_html=True,
        )

with col_after:
    st.markdown('<div class="img-label">St &nbsp; 动作后</div>', unsafe_allow_html=True)
    st.image(img_after, use_container_width=True)

with st.expander("🔎 当前样本详情", expanded=False):
    st.json(
        {
            "step_id": step_id,
            "source": step.get("dataset_source"),
            "trajectory_id": step.get("trajectory_id"),
            "step_idx": step.get("step_idx"),
            "website": step.get("website"),
            "domain": step.get("domain"),
            "source_file": step.get("source_file"),
            "obs_prev_image_ref": step.get("obs_prev_image_ref"),
            "obs_next_image_ref": step.get("obs_next_image_ref"),
            "gamma_penalty": GAMMA_PENALTY,
            "needs_goal_mapping": step.get("needs_goal_mapping"),
            "final_success": step.get("final_success"),
            "is_terminal": step.get("is_terminal"),
        }
    )

# ── 底部：标注历史预览 ─────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 标注历史（最近 20 条）", expanded=False):
    if st.session_state.annotations:
        recent = list(st.session_state.annotations.values())[-20:]
        rows = []
        for r in reversed(recent):
            rows.append({
                "步骤 ID": r["step_id"],
                "任务 ID": r.get("task_id", ""),
                "V_t": r["V_t"],
                "y_t": r.get("y_t"),
                "O_t": r["O_t"],
                "R_t": r["R_t"],
                "备注": r.get("note", ""),
                "时间": r["timestamp"][:19],
            })
        st.dataframe(rows, use_container_width=True)

        # 导出按钮
        all_records = list(st.session_state.annotations.values())
        jsonl_str = "\n".join(json.dumps(r, ensure_ascii=False) for r in all_records)
        st.download_button(
            "⬇️ 导出全部标注 (JSONL)",
            data=jsonl_str,
            file_name=f"annotations_{annotator_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="application/jsonl",
        )
    else:
        st.info("暂无标注记录")
