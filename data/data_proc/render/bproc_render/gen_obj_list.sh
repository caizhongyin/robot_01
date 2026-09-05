#!/bin/bash
# 通过 find 命令收集所有 OBJ 模型，生成 obj_list.txt
# 用法: bash gen_obj_list.sh [搜索目录]

SEARCH_DIR="${1:-test_objs}"
find "$SEARCH_DIR" -name "*.obj" | sort > obj_list.txt
echo "已生成 obj_list.txt，共 $(wc -l < obj_list.txt) 个 OBJ 文件"
