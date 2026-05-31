#!/usr/bin/env python3
"""
项目索引构建脚本 - 多进程并行版本
使用 ProcessPoolExecutor 并行解析所有 .py 文件，构建 SQLite + FTS5 索引。
"""
import ast
import os
import re
import sqlite3
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ==================== 配置 ====================
ROOT = Path("/data/luolie/缝合模块")
DB_PATH = ROOT / "index.db"
DBG_PATH = ROOT / "debug_venue.txt"

WORKERS = min(64, os.cpu_count() or 16)

# ==================== 正则引擎 ====================
# 顶会 + 年份  (宽松匹配, 不含年份的后面再补)
VENUE_RE = re.compile(
    r"(?P<venue>"
    r"CVPR|ICCV|ECCV|NeurIPS|AAAI|ICLR|TPAMI|IJCAI|ACMMM|SIGGRAPH|MM|TIP|TNNLS|TM|TCSVT|"
    r"Arxiv|ArXiv|arXiv|NN|Elsevier|IEEE|SCI|ICCASSP|ICASSP|BIBM|BMVC|ACCV|WACV|CoRL|WWW|"
    r"SIGIR|EMNLP|ACL|NAACL|KDD|ICDM|SDM|ICME|ACML|MLMI|JSTSP|NovEL|LMVT|DSP|CCF|"
    r"Nature|Science|Cell"
    r")[.\s]*(?P<year>20[12]\d)?",
    re.I
)
# 单独年份（顶会前缀后面没有紧跟年份的情况）
YEAR_RE = re.compile(r"\b(20[12]\d|19[7-9]\d)\b")
# 括号里的完整标签 "(CVPR 2024)" / "(AAAI2025)" / "(Arxiv2024)"
PAREN_RE = re.compile(
    r"\((?P<venue>[A-Za-z]+)[.\s]*(?P<year>20[12]\d)\)"
)

# 机制关键词 → 分类标签
MECH_KEYWORDS = {
    "Attention":        ["attention", "attn"],
    "Convolution":      ["conv", "dwconv", "pconv", "dcn", "dconv", "dwc", "cgl", "akconv", "gconv", "scconv"],
    "Pooling":          ["pool", "unpool", "downsample", "upsample", "hwd"],
    "Normalization":    ["norm", "bn", "gn", "ln", "rmsnorm", "layernorm", "batchnorm", "groupnorm"],
    "MLP/Mixer":        ["mlp", "mixer", "ffn", "gated", "glu", "swiglu", "c2f"],
    "Mamba/SSM":        ["mamba", "ssm", "state.space", "selective.scan", "hydra", "vim", "jamba"],
    "Frequency":        ["fft", "dct", "fourier", "wavelet", "freq", "频域", "频谱", "dwt", "cwt"],
    "Gate":             ["gate", "sigmoid", "hardmish", "softsign"],
    "Fusion":           ["fuse", "fusion", "blend", "merge", "cat", "concat", "coord"],
    "Deformable":       ["dcn", "deform", "offset"],
    "Strip":            ["strip", "条带", "stripe", "条形"],
    "Token/Patch":      ["token", "patch", "embed", "spatial"],
    "Multi-Scale":      ["multi.scale", "msda", "mspf", "ppm", "aspp", "fpn", "neck"],
    "Efficient":        ["efficient", "lightweight", "mobile", "fast", "swift"],
    "KAN":              ["kan", "kolmogorov", "bspline"],
}

# 任务关键词 → 任务标签
TASK_KEYWORDS = {
    "Image Restoration": ["restoration", "denoise", "deblur", "derain", "desnow", "dehaze", "irsr", "图像恢复", "restormer"],
    "Object Detection":  ["detect", "yolo", "retinanet", "faster-rcnn", "ssd", "目标检测", "小目标"],
    "Segmentation":      ["segment", "seg", "mask", "semantic", "实例分割", "语义分割", "deeplab"],
    "Classification":     ["classif", "imagenet", "top-1", "top-5", "分类", "cls"],
    "Super-Resolution":   ["sr", "super.resolution", "超分辨率", "edsr", "swinir"],
    "Tracking":          ["track", "sot", "mot", "跟踪"],
    "ReID":              ["reid", "retrieval", "行人检索"],
    "Depth":             ["depth", "stereo", "单目", "深度估计"],
    "Time Series":       ["forecast", "时间序列", "预测", "lstm", "gru", "ts", "series"],
    "Speech/Audio":      ["speech", "asr", "tts", "audio", "语音"],
    "NLP":               ["nlp", "bert", "gpt", "transformer", "text"],
    "Low-Level Vision":   ["low.level", "enhance", "iqa"],
    "Medical":           ["medical", "ct", "mri", "xray", "医疗"],
    "Remote Sensing":    ["remote", "sensing", "遥感", "sar"],
}

# ==================== 核心解析函数 ====================
def extract_metadata(filepath: str) -> dict | None:
    """解析单个 .py 文件，返回元数据字典。"""
    fname = Path(filepath).name
    fname_clean = fname.rsplit(".", 1)[0] if "." in fname else fname

    # ---- 1. 顶会和年份 ----
    venue, year = None, None
    # 先尝试括号内标签
    pm = PAREN_RE.search(fname_clean)
    if pm:
        venue = pm.group("venue").strip()
        year = pm.group("year")
    if not venue:
        vm = VENUE_RE.search(fname_clean)
        if vm:
            v = vm.group("venue").strip()
            y = vm.group("year")
            venue = v
            year = y
    # 补充年份
    if not year:
        ym = YEAR_RE.search(fname_clean)
        if ym:
            year = ym.group(1)
    if year:
        year = int(year)
    # 标准化顶会名
    if venue:
        venue = venue.title()
        venue = venue.replace("Arxiv", "ArXiv").replace("ArXiv", "ArXiv")

    # ---- 2. 模块类型 ----
    text = (fname_clean + " " + Path(filepath).read_text(errors="ignore", encoding="utf-8")[:2000]).lower()
    matched_types = []
    for mtype, kws in MECH_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                matched_types.append(mtype)
                break
    module_types = list(dict.fromkeys(matched_types)) if matched_types else ["Other"]

    # ---- 3. 适用任务 ----
    matched_tasks = []
    for task, kws in TASK_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in fname_clean.lower():
                matched_tasks.append(task)
                break
    tasks = list(dict.fromkeys(matched_tasks))

    # ---- 4. AST 解析 ----
    classes, functions = [], []
    try:
        source = Path(filepath).read_text(encoding="utf-8")
    except Exception:
        try:
            source = Path(filepath).read_text(encoding="gbk")
        except Exception:
            source = ""
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                init_params = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_params = [a.arg for a in item.args.args if a.arg != "self"]
                        break
                classes.append({"name": node.name, "params": init_params})
            elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
    except SyntaxError:
        pass

    # ---- 5. 摘要 ----
    lines = source.splitlines()
    summary_parts = []
    for line in lines[:30]:
        s = line.strip()
        if s.startswith("#") and len(s) > 3:
            summary_parts.append(s.lstrip("#").strip())
    summary = " ".join(summary_parts[:4])[:300]

    # ---- 6. 关联 txt ----
    parent = Path(filepath).parent
    stem = Path(filepath).stem
    related_txt = []
    for txt in parent.glob(stem[:5] + "*"):
        if txt.suffix == ".txt" and txt.stem != stem:
            related_txt.append(str(txt.relative_to(ROOT)))
    # 也查同名 txt（如果有）
    for txt in parent.glob("*.txt"):
        if stem[:6] in txt.stem and txt.stem != stem:
            related_txt.append(str(txt.relative_to(ROOT)))
    related_txt = "; ".join(related_txt[:3])

    # ---- 7. 分类 ----
    path_str = str(filepath)
    if "二次创新" in path_str:
        category = "微创新组件"
        innov_level = "组件级"
        code_complexity = "低"
    elif ("cv/" in path_str or "即插即用/" in path_str) and "/" in path_str:
        category = "即插即用模块"
        innov_level = "独立模块"
        code_complexity = "中"
    elif "blocks" in path_str and path_str.count("/") >= 2:
        category = "Blocks独立模块"
        innov_level = "独立模块"
        code_complexity = "中"
    elif "AI缝合术" in path_str:
        if "2026" in path_str:
            category = "AI缝合术_2026"
        elif "2025" in path_str:
            category = "AI缝合术_2025"
        elif "2024" in path_str:
            category = "AI缝合术_2024"
        elif "2023" in path_str or "及以前" in path_str:
            category = "AI缝合术_2023及以前"
        elif "19.9" in path_str:
            category = "AI缝合术_福利包"
        else:
            category = "AI缝合术_其他"
        innov_level = "完整模块"
        code_complexity = "中"
    else:
        category = "其他"
        innov_level = "其他"
        code_complexity = "中"

    line_count = len(lines)
    file_size = Path(filepath).stat().st_size

    return {
        "filename":     fname,
        "py_path":      str(Path(filepath).relative_to(ROOT)),
        "category":     category,
        "venue":        venue,
        "year":         year,
        "module_type":  "|".join(module_types),
        "task_domain":  "|".join(tasks),
        "summary":      summary,
        "classes":      "|".join(c["name"] for c in classes),
        "class_params": "|".join(c["name"] + "(" + ",".join(c["params"]) + ")" for c in classes),
        "functions":    "|".join(functions[:20]),
        "line_count":   line_count,
        "file_size":    file_size,
        "related_txt":  related_txt,
        "innov_level":  innov_level,
        "task_dir":     "; ".join(tasks) if tasks else "通用",
        "code_complexity": code_complexity,
    }


def process_batch(file_paths: list[str]) -> list[dict]:
    """处理一批文件（在子进程中运行）"""
    results = []
    for fp in file_paths:
        try:
            r = extract_metadata(fp)
            if r:
                results.append(r)
        except Exception as e:
            pass
    return results


# ==================== 主程序 ====================
def build_index():
    # ---- 收集所有 .py 文件 ----
    dirs_to_scan = [
        ROOT / "AI缝合术模块" / "2024年（全）",
        ROOT / "AI缝合术模块" / "2025年",
        ROOT / "AI缝合术模块" / "2026年持续更新",
        ROOT / "AI缝合术模块" / "2023年及以前（全）",
        ROOT / "AI缝合术模块" / "19.9元福利（235个代码）",
        ROOT / "blocks",
        ROOT / "cv",
        ROOT / "二次创新",
    ]

    all_files = []
    for d in dirs_to_scan:
        if not d.exists():
            print(f"[跳过] {d}")
            continue
        for fp in d.rglob("*.py"):
            if ".idea" not in str(fp) and "__pycache__" not in str(fp):
                all_files.append(str(fp))

    print(f"共找到 {len(all_files)} 个 .py 文件，使用 {WORKERS} 个进程...")

    # ---- 分批（每批 ~100 个文件，减少进程间通信开销） ----
    BATCH = 100
    batches = [all_files[i:i+BATCH] for i in range(0, len(all_files), BATCH)]
    print(f"分为 {len(batches)} 批，每批 ~{BATCH} 个文件")

    all_results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        done = 0
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                results = fut.result()
                all_results.extend(results)
                done += 1
                if done % 10 == 0 or done == len(batches):
                    print(f"  进度: {done}/{len(batches)} 批, 累计 {len(all_results)} 条记录")
            except Exception as e:
                print(f"  批次 {batch_idx} 出错: {e}")

    print(f"\n总计解析成功 {len(all_results)} 条记录")

    # ---- 写调试文件：venue 提取情况抽样 ----
    with open(DBG_PATH, "w", encoding="utf-8") as f:
        f.write("=== Venue 提取调试 ===\n")
        venue_found = 0
        for r in all_results:
            if r["venue"]:
                venue_found += 1
            else:
                f.write(f"  NULL venue: {r['py_path']}\n")
        f.write(f"\n有 venue 记录: {venue_found}/{len(all_results)}\n")
    print(f"调试文件: {DBG_PATH}")

    # ---- 建立 SQLite 数据库 ----
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE modules (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL,
            py_path         TEXT UNIQUE NOT NULL,
            category        TEXT,
            venue           TEXT,
            year            INTEGER,
            module_type     TEXT,
            task_domain     TEXT,
            summary         TEXT,
            classes         TEXT,
            class_params    TEXT,
            functions       TEXT,
            line_count      INTEGER,
            file_size       INTEGER,
            related_txt     TEXT,
            innov_level     TEXT,
            task_dir        TEXT,
            code_complexity TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("CREATE INDEX idx_category   ON modules(category)")
    cur.execute("CREATE INDEX idx_venue      ON modules(venue)")
    cur.execute("CREATE INDEX idx_year       ON modules(year)")
    cur.execute("CREATE INDEX idx_module_type ON modules(module_type)")

    # FTS5 全文检索表
    cur.execute("""
        CREATE VIRTUAL TABLE modules_fts USING fts5(
            filename, summary, module_type, task_domain,
            content=modules, content_rowid=id
        )
    """)

    # 插入数据
    cols = ["filename","py_path","category","venue","year","module_type","task_domain",
            "summary","classes","class_params","functions","line_count","file_size",
            "related_txt","innov_level","task_dir","code_complexity"]
    for r in all_results:
        cur.execute(
            f"INSERT INTO modules ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            [r.get(c, "") for c in cols]
        )

    # FTS 重建
    cur.execute("INSERT INTO modules_fts(modules_fts) VALUES('rebuild')")
    conn.commit()

    # ---- 统计报告 ----
    cur.execute("SELECT COUNT(*) FROM modules")
    total = cur.fetchone()[0]
    print(f"\n数据库写入完成，共 {total} 条记录")

    print("\n--- 按分类统计 ---")
    cur.execute("SELECT category, COUNT(*) FROM modules GROUP BY category ORDER BY 2 DESC")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\n--- 按顶会统计 (Top 20) ---")
    cur.execute("SELECT venue, COUNT(*) FROM modules WHERE venue IS NOT NULL AND venue != '' GROUP BY venue ORDER BY 2 DESC LIMIT 20")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\n--- 按年份统计 ---")
    cur.execute("SELECT year, COUNT(*) FROM modules WHERE year IS NOT NULL GROUP BY year ORDER BY 1 DESC LIMIT 20")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print(f"\n数据库路径: {DB_PATH}")


if __name__ == "__main__":
    build_index()
