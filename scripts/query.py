#!/usr/bin/env python3
"""
检索脚本 - 支持多维查询
用法示例:
  python3 query.py                          # 交互模式
  python3 query.py --venue CVPR --year 2024 # 按顶会+年份查询
  python3 query.py --type Attention          # 按机制类型查询
  python3 query.py --task Detection          # 按任务类型查询
  python3 query.py --search "图像恢复"        # 全文搜索
  python3 query.py --class-name SSA          # 按类名查询
  python3 query.py --top 10 --venue CVPR     # Top N 结果
  python3 query.py --all                     # 打印全部（分页）
  python3 query.py --report                  # 生成统计报告
  python3 query.py --stats                   # 快速统计
  python3 query.py --export-md               # 导出 Markdown 目录
"""
import argparse
import sqlite3
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path("/data/luolie/缝合模块")
DB = ROOT / "index.db"

PAGE_SIZE = 20

# ==================== 查询引擎 ====================
def get_conn():
    return sqlite3.connect(str(DB))

def query_by_venue(conn, venue, year=None, limit=50):
    sql = "SELECT * FROM modules WHERE venue LIKE ?"
    params = [f"%{venue}%"]
    if year:
        sql += " AND year = ?"
        params.append(year)
    sql += f" ORDER BY year DESC, venue LIMIT {limit}"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def query_by_year(conn, year, limit=50):
    sql = f"SELECT * FROM modules WHERE year = ? ORDER BY venue, filename LIMIT {limit}"
    cur = conn.execute(sql, (year,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def query_by_type(conn, module_type, limit=50):
    sql = f"SELECT * FROM modules WHERE module_type LIKE ? ORDER BY year DESC LIMIT {limit}"
    cur = conn.execute(sql, (f"%{module_type}%",))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def query_by_task(conn, task, limit=50):
    sql = f"SELECT * FROM modules WHERE task_domain LIKE ? OR task_dir LIKE ? ORDER BY year DESC LIMIT {limit}"
    cur = conn.execute(sql, (f"%{task}%", f"%{task}%"))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def query_by_class(conn, class_name, limit=50):
    sql = f"SELECT * FROM modules WHERE classes LIKE ? OR class_params LIKE ? ORDER BY year DESC LIMIT {limit}"
    cur = conn.execute(sql, (f"%{class_name}%", f"%{class_name}%"))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def fulltext_search(conn, keyword, limit=50):
    sql = f"""
        SELECT modules.* FROM modules
        JOIN modules_fts ON modules.id = modules_fts.rowid
        WHERE modules_fts MATCH ?
        ORDER BY rank, year DESC LIMIT {limit}
    """
    try:
        cur = conn.execute(sql, (keyword,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()], cols
    except Exception:
        # FTS 失败时降级为 LIKE
        sql2 = f"SELECT * FROM modules WHERE filename||summary||module_type||task_domain LIKE ? LIMIT {limit}"
        cur = conn.execute(sql2, (f"%{keyword}%",))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def query_by_category(conn, category, limit=50):
    sql = f"SELECT * FROM modules WHERE category = ? ORDER BY year DESC, filename LIMIT {limit}"
    cur = conn.execute(sql, (category,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

def query_all(conn, limit=None, offset=0):
    lim = limit or 999999
    sql = f"SELECT * FROM modules ORDER BY year DESC, category, filename LIMIT {lim} OFFSET {offset}"
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()], cols

# ==================== 输出格式 ====================
def fmt_row(r):
    venue = r.get("venue") or "-"
    year  = r.get("year") or "-"
    types = r.get("module_type", "").replace("|", ", ") or "-"
    tasks = r.get("task_dir", "") or "-"
    path  = r.get("py_path", "")
    cls   = r.get("classes", "") or "-"
    size  = r.get("file_size", 0) or 0
    return f"  [{venue} {year}] {r['filename']}\n    类型: {types} | 任务: {tasks}\n    类: {cls} | 大小: {size//1024}KB\n    路径: {path}"

def print_results(rows, cols, title=""):
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    print(f"\n共找到 {len(rows)} 条结果:\n")
    for i, r in enumerate(rows, 1):
        print(f"[{i}] {fmt_row(r)}")
        print()

def print_table(rows, cols):
    """简洁表格输出"""
    for r in rows:
        print(f"  {r.get('venue',''):<12} {r.get('year','-'):<6} {r.get('filename','')[:50]}")

def stats_report(conn):
    print("\n" + "="*60)
    print("  项目模块统计报告")
    print("="*60)

    cur = conn.execute("SELECT COUNT(*) FROM modules")
    total = cur.fetchone()[0]
    print(f"\n总计模块数: {total}")

    print("\n--- 按分类 ---")
    cur = conn.execute("SELECT category, COUNT(*) as n FROM modules GROUP BY category ORDER BY n DESC")
    for row in cur.fetchall():
        bar = "█" * (row[1] * 30 // total)
        print(f"  {row[0]:<25} {row[1]:>5}  {bar}")

    print("\n--- 按顶会 (Top 15) ---")
    cur = conn.execute("SELECT venue, COUNT(*) as n FROM modules WHERE venue IS NOT NULL AND venue != '' GROUP BY venue ORDER BY n DESC LIMIT 15")
    for row in cur.fetchall():
        bar = "█" * (row[1] * 30 // total)
        print(f"  {row[0]:<12} {row[1]:>5}  {bar}")

    print("\n--- 按年份 ---")
    cur = conn.execute("SELECT year, COUNT(*) as n FROM modules WHERE year IS NOT NULL GROUP BY year ORDER BY 1 DESC")
    for row in cur.fetchall():
        bar = "█" * (row[1] * 30 // total)
        print(f"  {row[0]:>6}  {row[1]:>5}  {bar}")

    print("\n--- 按机制类型 ---")
    # 拆分 | 分隔的 module_type
    cur = conn.execute("SELECT module_type FROM modules WHERE module_type IS NOT NULL")
    type_counts = {}
    for (mt,) in cur.fetchall():
        for t in mt.split("|"):
            t = t.strip()
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * (n * 30 // total)
        print(f"  {t:<20} {n:>5}  {bar}")

    print("\n--- 按任务领域 ---")
    cur = conn.execute("SELECT task_dir FROM modules WHERE task_dir IS NOT NULL AND task_dir != ''")
    task_counts = {}
    for (td,) in cur.fetchall():
        for t in td.split(";"):
            t = t.strip()
            if t:
                task_counts[t] = task_counts.get(t, 0) + 1
    for t, n in sorted(task_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * (n * 30 // total)
        print(f"  {t:<20} {n:>5}  {bar}")

    print("\n--- 代码行数分布 ---")
    cur = conn.execute("""
        SELECT
            CASE
                WHEN line_count < 50 THEN '< 50 行'
                WHEN line_count < 100 THEN '50-100 行'
                WHEN line_count < 200 THEN '100-200 行'
                WHEN line_count < 500 THEN '200-500 行'
                ELSE '> 500 行'
            END as bucket,
            COUNT(*) as n
        FROM modules
        GROUP BY bucket
        ORDER BY MIN(line_count)
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:<15} {row[1]:>5}")

def export_markdown(conn, out_path):
    """导出完整 Markdown 目录"""
    print(f"正在导出 Markdown 目录到: {out_path}")

    md = []
    md.append("# 缝合模块项目总目录\n")
    md.append("> 自动生成 | 由 `scripts/build_index.py` 索引\n\n")

    # ---- 统计摘要 ----
    cur = conn.execute("SELECT COUNT(*) FROM modules")
    total = cur.fetchone()[0]
    md.append(f"**总计: {total} 个深度学习模块**\n\n")

    # ---- 按分类组织 ----
    categories = [
        ("AI缝合术_2026", "AI缝合术模块 · 2026 年"),
        ("AI缝合术_2025", "AI缝合术模块 · 2025 年"),
        ("AI缝合术_2024", "AI缝合术模块 · 2024 年"),
        ("AI缝合术_2023及以前", "AI缝合术模块 · 2023 年及以前"),
        ("AI缝合术_福利包", "AI缝合术模块 · 福利包"),
        ("Blocks独立模块", "Blocks 独立模块"),
        ("即插即用模块", "即插即用 / CV 模块"),
        ("微创新组件", "二次创新微创新组件"),
    ]

    cur = conn.execute("SELECT category FROM modules GROUP BY category ORDER BY COUNT(*) DESC")
    all_cats = [r[0] for r in cur.fetchall()]
    for cat, cat_name in categories:
        if cat not in all_cats:
            continue
        md.append(f"\n## {cat_name}\n")
        md.append(f"| 模块名 | 类名 | 顶会 | 年 | 类型 | 适用任务 | 代码行数 |\n")
        md.append(f"|--------|------|------|----|------|----------|----------|\n")

        cur = conn.execute(
            "SELECT filename, classes, venue, year, module_type, task_dir, line_count, py_path "
            "FROM modules WHERE category = ? ORDER BY year DESC, filename",
            (cat,)
        )
        for row in cur.fetchall():
            fname, cls, venue, year, mtype, task, lines, path = row
            fname_link = f"[{fname}]({path})"
            cls_short = cls.split("|")[0] if cls else "-"
            mtype_short = mtype.replace("|", ", ")[:30] if mtype else "-"
            md.append(f"| {fname_link} | {cls_short} | {venue or '-'} | {year or '-'} | {mtype_short} | {task or '-'} | {lines} |\n")

    # ---- 顶会索引 ----
    md.append("\n---\n\n## 顶会论文索引\n\n")
    cur = conn.execute(
        "SELECT DISTINCT venue, year, COUNT(*) as n "
        "FROM modules WHERE venue IS NOT NULL AND venue != '' "
        "GROUP BY venue, year ORDER BY year DESC, venue"
    )
    md.append("| 顶会 | 年份 | 模块数 |\n|------|------|--------|\n")
    for row in cur.fetchall():
        md.append(f"| {row[0]} | {row[1]} | {row[2]} |\n")

    # ---- 机制分类索引 ----
    md.append("\n---\n\n## 机制类型索引\n\n")
    md.append("按注意力机制、卷积类型等分类的模块列表：\n\n")
    # Attention
    md.append("### Attention 类\n\n")
    cur = conn.execute(
        "SELECT filename, venue, year, py_path FROM modules "
        "WHERE module_type LIKE '%Attention%' ORDER BY year DESC LIMIT 30"
    )
    for row in cur.fetchall():
        md.append(f"- [{row[0]}]({row[3]}) [{row[1] or ''} {row[2] or ''}]\n")
    md.append("\n### Convolution 类\n\n")
    cur = conn.execute(
        "SELECT filename, venue, year, py_path FROM modules "
        "WHERE module_type LIKE '%Convolution%' ORDER BY year DESC LIMIT 30"
    )
    for row in cur.fetchall():
        md.append(f"- [{row[0]}]({row[3]}) [{row[1] or ''} {row[2] or ''}]\n")
    md.append("\n### Frequency 类\n\n")
    cur = conn.execute(
        "SELECT filename, venue, year, py_path FROM modules "
        "WHERE module_type LIKE '%Frequency%' ORDER BY year DESC LIMIT 20"
    )
    for row in cur.fetchall():
        md.append(f"- [{row[0]}]({row[3]}) [{row[1] or ''} {row[2] or ''}]\n")
    md.append("\n### Mamba/SSM 类\n\n")
    cur = conn.execute(
        "SELECT filename, venue, year, py_path FROM modules "
        "WHERE module_type LIKE '%Mamba%' ORDER BY year DESC LIMIT 20"
    )
    for row in cur.fetchall():
        md.append(f"- [{row[0]}]({row[3]}) [{row[1] or ''} {row[2] or ''}]\n")
    md.append("\n### Multi-Scale 类\n\n")
    cur = conn.execute(
        "SELECT filename, venue, year, py_path FROM modules "
        "WHERE module_type LIKE '%Multi-Scale%' ORDER BY year DESC LIMIT 20"
    )
    for row in cur.fetchall():
        md.append(f"- [{row[0]}]({row[3]}) [{row[1] or ''} {row[2] or ''}]\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"Markdown 目录已导出: {out_path}")


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="缝合模块检索工具", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venue", help="按顶会查询 (CVPR/AAAI/ICCV/ArXiv/TPAMI ...)")
    parser.add_argument("--year", type=int, help="按年份查询 (2024/2025 ...)")
    parser.add_argument("--type", help="按机制类型查询 (Attention/Convolution/Frequency ...)")
    parser.add_argument("--task", help="按任务类型查询 (Detection/Restoration/Segmentation ...)")
    parser.add_argument("--class-name", dest="class_name", help="按类名查询")
    parser.add_argument("--search", help="全文搜索关键词")
    parser.add_argument("--category", help="按分类查询")
    parser.add_argument("--top", type=int, default=50, help="最多返回条数 (默认 50)")
    parser.add_argument("--all", action="store_true", help="打印所有模块（分页）")
    parser.add_argument("--report", action="store_true", help="生成完整统计报告")
    parser.add_argument("--stats", action="store_true", help="快速统计")
    parser.add_argument("--export-md", dest="export_md", action="store_true", help="导出 Markdown 目录")
    parser.add_argument("--limit", type=int, default=50, help="limit (默认 50)")
    args = parser.parse_args()

    conn = get_conn()

    if args.stats:
        stats_report(conn)
        conn.close()
        return

    if args.report:
        stats_report(conn)
        print("\n")
        # 额外输出 Top 模块
        print("\n--- CVPR 2024 Top Attention 模块 ---")
        rows, cols = query_by_venue(conn, "CVPR", 2024, 20)
        for r in rows[:10]:
            print(f"  {r['filename']} | 类: {r['classes']}")
        conn.close()
        return

    if args.export_md:
        out = ROOT / "PROJECT_INDEX.md"
        export_markdown(conn, out)
        conn.close()
        return

    if args.all:
        rows, cols = query_all(conn, limit=args.limit)
        print_results(rows, cols, f"全部模块 (共 {len(rows)} 条)")
        conn.close()
        return

    if args.venue:
        rows, cols = query_by_venue(conn, args.venue, args.year, args.top)
        title = f"顶会查询: {args.venue}" + (f" {args.year}" if args.year else "")
        print_results(rows, cols, title)
        conn.close()
        return

    if args.type:
        rows, cols = query_by_type(conn, args.type, args.top)
        print_results(rows, cols, f"机制类型: {args.type}")
        conn.close()
        return

    if args.task:
        rows, cols = query_by_task(conn, args.task, args.top)
        print_results(rows, cols, f"任务类型: {args.task}")
        conn.close()
        return

    if args.class_name:
        rows, cols = query_by_class(conn, args.class_name, args.top)
        print_results(rows, cols, f"类名查询: {args.class_name}")
        conn.close()
        return

    if args.search:
        rows, cols = fulltext_search(conn, args.search, args.top)
        print_results(rows, cols, f"全文搜索: {args.search}")
        conn.close()
        return

    if args.category:
        rows, cols = query_by_category(conn, args.category, args.top)
        print_results(rows, cols, f"分类: {args.category}")
        conn.close()
        return

    # 无参数 -> 打印使用说明
    parser.print_help()
    print("\n--- 快速统计 ---")
    stats_report(conn)
    conn.close()


if __name__ == "__main__":
    main()
