#!/bin/bash
# 并行批量将 PDF 转换为 TXT

PDF_LIST="/home/luolie/.cursor/projects/data-luolie/agent-tools/pdf_list.txt"
LOG_FILE="/data/luolie/缝合模块/pdf_conversion.log"
PARALLEL_JOBS=32  # 并行任务数，可根据CPU核心数调整

echo "开始批量转换 PDF 到 TXT..."
echo "时间: $(date)" | tee "$LOG_FILE"
echo "并行数: $PARALLEL_JOBS"
echo "总文件数: $(wc -l < "$PDF_LIST")"
echo "---" | tee -a "$LOG_FILE"

# 使用 xargs 并行处理
cat "$PDF_LIST" | xargs -P "$PARALLEL_JOBS" -I {} bash -c '
    pdf_path="{}"
    txt_path="${pdf_path%.pdf}.txt"
    
    if [ -f "$txt_path" ]; then
        echo "SKIP: $txt_path"
    elif pdftotext "$pdf_path" "$txt_path" 2>/dev/null; then
        echo "OK: $txt_path"
    else
        echo "FAIL: $pdf_path"
    fi
' 2>/dev/null | tee -a "$LOG_FILE"

echo "---" | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"

# 统计结果
total=$(wc -l < "$PDF_LIST")
success=$(grep -c "^OK:" "$LOG_FILE")
skip=$(grep -c "^SKIP:" "$LOG_FILE")
fail=$(grep -c "^FAIL:" "$LOG_FILE")

echo ""
echo "=== 统计结果 ==="
echo "总计: $total"
echo "成功: $success"
echo "跳过: $skip"
echo "失败: $fail"
