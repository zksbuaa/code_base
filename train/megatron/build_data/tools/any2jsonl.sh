#!/bin/bash
# 将各种其他类型的多个文件转换为一个单独的 jsonl 文件

#######################################
# 使用说明
#######################################
usage() {
    cat << EOF
用法：$(basename "$0") [选项]

选项：
    -s, --source DIR     源文件夹或文件路径，支持多个路径，用逗号分隔
    -d, --depth NUM      获取文件时的最大深度 (默认: 1)
    -f, --suffixes LIST  需要处理的文件后缀，用逗号分隔 (默认: jsonl.zst,gz,parquet,jsonl,json)
    -m, --method TYPE    文件选择策略: random / size_desc / size_asc (默认: random)
    -o, --output FILE    输出文件路径 (默认: ./output.jsonl)
    -g, --size-gb NUM    目标文件最大大小，以GB为单位 (默认: 100)
    -e, --default-ext EXT 处理无后缀名文件时使用的默认后缀 (默认: json)
    -n, --no-ext         包含无后缀文件 (默认: 不包含)
    -h, --help           显示此帮助信息并退出

示例：
    $(basename "$0") -s /path/to/data -d 2 -o /path/to/output.jsonl
    $(basename "$0") --source /path/to/data1,/path/to/data2 --method size_desc
    $(basename "$0") -s /path/to/data -n -e jsonl  # 处理无后缀文件作为jsonl
EOF
    exit 1
}

#######################################
# 解析命令行参数
#######################################
parse_params() {
    # 默认参数值
    SOURCE_DIRS=("/path/to/source")
    MAX_DEPTH=1
    FILE_SUFFIXES=("*.jsonl.zst" "*.gz" "*.parquet" "*.jsonl" "*.json")
    SORT_METHOD="random"
    TARGET_FILE="./output.jsonl"
    TARGET_SIZE_IN_GB=100
    DEFAULT_EXT="json"
    INCLUDE_NO_EXT=false
    
    # 解析命令行参数
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -s|--source)
                IFS=',' read -ra SOURCE_DIRS <<< "$2"
                shift 2
                ;;
            -d|--depth)
                MAX_DEPTH="$2"
                shift 2
                ;;
            -f|--suffixes)
                IFS=',' read -ra temp_suffixes <<< "$2"
                FILE_SUFFIXES=()
                for suffix in "${temp_suffixes[@]}"; do
                    FILE_SUFFIXES+=("*.$suffix")
                done
                shift 2
                ;;
            -m|--method)
                SORT_METHOD="$2"
                if [[ ! "$SORT_METHOD" =~ ^(random|size_desc|size_asc)$ ]]; then
                    echo "错误：无效的排序方法 $SORT_METHOD"
                    usage
                fi
                shift 2
                ;;
            -o|--output)
                TARGET_FILE="$2"
                shift 2
                ;;
            -g|--size-gb)
                TARGET_SIZE_IN_GB="$2"
                if ! [[ "$TARGET_SIZE_IN_GB" =~ ^[0-9]+$ ]]; then
                    echo "错误：大小必须是整数"
                    usage
                fi
                shift 2
                ;;
            -e|--default-ext)
                DEFAULT_EXT="$2"
                shift 2
                ;;
            -n|--no-ext)
                INCLUDE_NO_EXT=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "未知参数: $1"
                usage
                ;;
        esac
    done

    TARGET_SIZE=$((TARGET_SIZE_IN_GB * 1024 * 1024 * 1024))

    # 验证参数
    if [ ${#SOURCE_DIRS[@]} -eq 0 ]; then
        echo "错误：未指定源目录"
        usage
    fi
    
    for dir in "${SOURCE_DIRS[@]}"; do
        if [ ! -e "$dir" ]; then
            echo "错误：源路径不存在: $dir"
            exit 1
        fi
    done

    # 打印当前配置
    echo "当前配置："
    echo "  源路径: ${SOURCE_DIRS[*]}"
    echo "  最大深度: $MAX_DEPTH"
    echo "  文件后缀: ${FILE_SUFFIXES[*]}"
    echo "  排序方法: $SORT_METHOD"
    echo "  目标文件: $TARGET_FILE"
    echo "  目标大小: ${TARGET_SIZE_IN_GB}GB"
    echo "  默认后缀: $DEFAULT_EXT"
    echo "  包含无后缀文件: $([ "$INCLUDE_NO_EXT" = true ] && echo "是" || echo "否")"
}

#######################################
# 处理方法：根据文件后缀来决定如何处理
#######################################
process_file() {
    local file="$1"
    
    # 获取文件扩展名
    local filename=$(basename "$file")
    local suffix=""
    
    if [[ "$filename" == *.* ]]; then
        if [[ "$filename" == *.jsonl.zst ]]; then
            suffix="jsonl.zst"
        else
            suffix="${filename##*.}"
        fi
    else
        # 没有扩展名的文件
        if [ "$INCLUDE_NO_EXT" = true ]; then
            echo "注意：文件 '$file' 没有扩展名，使用默认后缀 '$DEFAULT_EXT'" >&2
            suffix="$DEFAULT_EXT"
        else
            echo "错误：文件 '$file' 没有扩展名，且未启用处理无后缀文件" >&2
            return 1
        fi
    fi
    
    case "$suffix" in
        "jsonl.zst")
            zstd -dc "$file" ;;
        "gz")
            gunzip -c "$file" ;;
        "parquet")
            # 使用我们的Python脚本处理parquet文件
            if ! command -v python &> /dev/null; then
                echo "错误：python 不存在，无法处理 parquet 文件" >&2
                return 1
            fi
            
            # 检查脚本是否存在
            PARQUET_SCRIPT="$(dirname "$0")/parquet2jsonl.py"
            if [ ! -f "$PARQUET_SCRIPT" ]; then
                PARQUET_SCRIPT="./parquet2jsonl.py"
                if [ ! -f "$PARQUET_SCRIPT" ]; then
                    echo "错误：parquet2jsonl.py 脚本不存在，无法处理 parquet 文件" >&2
                    return 1
                fi
            fi
            
            # 确保脚本有执行权限
            chmod +x "$PARQUET_SCRIPT"
            
            # 执行脚本转换parquet文件
            python "$PARQUET_SCRIPT" "$file"
            ;;
        "jsonl"|"json")
            cat "$file" ;;
        *)
            echo "不支持的文件类型：$suffix" >&2
            return 1
            ;;
    esac
}

#######################################
# 获取符合条件的文件列表
#######################################
get_files() {
    local dirs=("${@}")
    local depth=$MAX_DEPTH
    local suffixes=("${FILE_SUFFIXES[@]}")
    
    local files=()
    local temp_file
    
    # 创建临时文件来存储文件列表
    temp_file=$(mktemp)
    
    # 使用 glob 模式查找文件，考虑深度限制
    #for dir in "${dirs[@]}"; do
    #    for suffix in "${suffixes[@]}"; do
    #        # 使用 find 命令来查找文件，支持递归和匹配后缀
    #        find "$dir" -maxdepth "$depth" -type f -name "$suffix" -print 2>/dev/null >> "$temp_file"
    #    done
        
    #    # 如果需要包含没有后缀的文件
    #    if [ "$INCLUDE_NO_EXT" = true ]; then
    #        # 查找没有扩展名的文件
    #        find "$dir" -maxdepth "$depth" -type f -not -name "*.*" -print 2>/dev/null >> "$temp_file"
    #    fi
    #done
    for dir in "${dirs[@]}"; do
        if [ -f "$dir" ]; then
            # 如果是文件，则直接加入列表（不检查后缀）
            echo "$dir" >> "$temp_file"
        elif [ -d "$dir" ]; then
            # 如果是目录，则正常递归查找
            for suffix in "${suffixes[@]}"; do
                find "$dir" -maxdepth "$depth" -type f -name "$suffix" >> "$temp_file"
            done
            
            if [ "$INCLUDE_NO_EXT" = true ]; then
                find "$dir" -maxdepth "$depth" -type f -not -name "*.*" >> "$temp_file"
            fi
        else
            echo "警告：路径不存在: $dir" >&2
        fi
    done
    
    # 检查是否找到文件
    if [ ! -s "$temp_file" ]; then
        echo "警告：没有找到符合条件的文件" >&2
        rm -f "$temp_file"
        return 1
    fi
    
    # 根据选择策略排序文件
    case "$SORT_METHOD" in
        "size_desc")
            # 使用临时文件存储排序结果
            sort_temp=$(mktemp)
            while read -r file; do
                if [ -f "$file" ]; then
                    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
                    echo -e "$size\t$file"
                fi
            done < "$temp_file" | sort -nr > "$sort_temp"
            
            # 提取排序后的文件名
            awk '{print $2}' "$sort_temp" > "$temp_file"
            rm -f "$sort_temp"
            ;;
        "size_asc")
            # 使用临时文件存储排序结果
            sort_temp=$(mktemp)
            while read -r file; do
                if [ -f "$file" ]; then
                    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
                    echo -e "$size\t$file"
                fi
            done < "$temp_file" | sort -n > "$sort_temp"
            
            # 提取排序后的文件名
            awk '{print $2}' "$sort_temp" > "$temp_file"
            rm -f "$sort_temp"
            ;;
        "random")
            # 使用 shuf 命令洗牌文件列表，但从文件读取而不是命令行参数
            shuf "$temp_file" -o "$temp_file"
            ;;
    esac

    # 返回找到的文件列表
    cat "$temp_file"
    rm -f "$temp_file"
}

#######################################
# 计算文件大小
#######################################
get_current_size() {
    if [ -f "$TARGET_FILE" ]; then
        stat -c%s "$TARGET_FILE" 2>/dev/null || stat -f%z "$TARGET_FILE" 2>/dev/null
    else
        echo "0"
    fi
}

#######################################
# 主逻辑：从源文件夹中获取文件并处理
#######################################
process_files() {
    local file_list="$1"
    local total_size=0
    local processed_count=0
    local total_files=$(wc -l < "$file_list")

    while read -r file; do
        # 处理文件并追加到目标文件
        echo -n "处理文件 (${processed_count}/${total_files}): $file ... "
        if process_file "$file" >> "$TARGET_FILE"; then
            echo "成功"
        else
            echo "失败"
            continue
        fi
        
        ((processed_count++))
        
        # 计算目标文件当前大小
        total_size=$(get_current_size)
        echo "当前目标文件大小：$(numfmt --to=iec-i --suffix=B $total_size)"
        
        # 如果目标文件大小达到了最大限制，退出
        if [ "$total_size" -ge "$TARGET_SIZE" ]; then
            echo "目标文件大小已达到限制，退出处理。"
            break
        fi
    done < "$file_list"
}

#######################################
# 主程序入口
#######################################
main() {
    # 安装依赖
    apt update
    apt install -y zstd
    pip install pyarrow

    # 解析命令行参数
    parse_params "$@"
    
    # 创建目标文件的目录（如果不存在）
    target_dir=$(dirname "$TARGET_FILE")
    mkdir -p "$target_dir"

    # 创建临时文件存储文件列表
    temp_file_list=$(mktemp)
    
    # 获取符合条件的文件列表
    echo "获取文件列表..."
    get_files "${SOURCE_DIRS[@]}" > "$temp_file_list"

    if [ ! -s "$temp_file_list" ]; then
        echo "没有找到符合条件的文件，退出。"
        rm -f "$temp_file_list"
        exit 1
    else
        file_count=$(wc -l < "$temp_file_list")
        echo "找到 ${file_count} 个符合条件的文件"

        # 打印前 100 个文件
        if [ "$file_count" -gt 100 ]; then
            echo "前 100 个文件："
            head -n 100 "$temp_file_list"
        else
            echo "文件列表："
            cat "$temp_file_list"
        fi
    fi

    # 如果目标文件存在则删除
    if [ -f "$TARGET_FILE" ]; then
        echo "目标文件已存在，将被覆盖"
        rm -f "$TARGET_FILE"
    fi

    # 开始处理文件
    echo "开始处理文件..."
    process_files "$temp_file_list"
    
    # 清理临时文件
    rm -f "$temp_file_list"

    echo "处理完毕！"
    echo "输出文件: $TARGET_FILE"
    echo "文件大小: $(numfmt --to=iec-i --suffix=B $(get_current_size))"
}

# 执行主程序
main "$@"