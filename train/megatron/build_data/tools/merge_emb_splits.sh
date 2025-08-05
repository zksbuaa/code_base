#!/bin/bash
set -exuo pipefail

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --emb_dim)
            emb_dim="$2"
            shift 2
            ;;
        --save_emb_path)
            save_emb_path="$2"
            shift 2
            ;;
        --output_file)
            output_file="$2"
            shift 2
            ;;
        --total_docs)
            total_docs="$2"
            shift 2
            ;;
        --total_nodes)
            total_nodes="$2"
            shift 2
            ;;
        --gpus_per_node)
            gpus_per_node="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
emb_dim=${emb_dim:-1024}
save_emb_path=${save_emb_path:-"/mnt/public/code/dynamic_train/dt-data/emb-cluster/data/.emb_splits"}
output_file=${output_file:-"/mnt/public/code/dynamic_train/dt-data/emb-cluster/data/merged_embs.npy"}
total_docs=${total_docs:-163615821}
total_nodes=${total_nodes:-15}
gpus_per_node=${gpus_per_node:-8}

# Generate list of split files
split_files=()
for node_id in $(seq 0 $((total_nodes - 1))); do
    for gpu_id in $(seq 0 $((gpus_per_node - 1))); do
        split_file="${save_emb_path}/emb_split_${node_id}_${gpu_id}.npy"
        if [[ ! -f "$split_file" ]]; then
            echo "Error: Split file $split_file does not exist"
            exit 1
        fi
        split_files+=("$split_file")
    done
done

echo "Found ${#split_files[@]} split files:"
printf '%s\n' "${split_files[@]}"

# Merge files using cat
echo "Merging files with cat..."
cat "${split_files[@]}" > "$output_file"

echo "Merging completed. Final file saved at $output_file"
