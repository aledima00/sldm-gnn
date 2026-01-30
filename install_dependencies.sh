has_cuda_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1 || return 1

    # almeno una GPU
    [ "$(nvidia-smi -L | wc -l)" -ge 1 ] || return 1

    return 0
}

if has_cuda_gpu; then
    echo "CUDA-capable GPU detected. Installing core dependencies for cuda..."
    uv sync --extra cuda
    echo "Done."
    echo "Installing PyG extras for cuda..."
    uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
    echo "Done."
else
    echo "No CUDA-capable GPU detected. Installing core dependencies for CPU..."
    uv sync --extra cpu
    echo "Done."
    echo "Installing PyG extras for CPU..."
    uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
    echo "Done."
fi