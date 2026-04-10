// kernels.cu - CUDA kernels for TorchWM

#include <torch/extension.h>

__global__ void batched_normalize_kernel(
    float* data,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        // Clamp to [0, 1]
        data[idx] = fmaxf(0.0f, fminf(1.0f, data[idx]));
    }
}

__global__ void batched_add_noise_kernel(
    float* data,
    const float* noise,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        data[idx] += noise[idx];
    }
}

torch::Tensor batched_normalize_cuda(torch::Tensor data) {
    auto sizes = data.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int height = sizes[2];
    int width = sizes[3];
    
    int total_elements = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    batched_normalize_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        batch_size, channels, height, width
    );
    
    return data;
}

torch::Tensor batched_add_noise_cuda(torch::Tensor data, torch::Tensor noise) {
    int total_elements = data.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    batched_add_noise_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        noise.data_ptr<float>(),
        total_elements
    );
    
    return data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_normalize_cuda", &batched_normalize_cuda, "Batched normalize kernel");
    m.def("batched_add_noise_cuda", &batched_add_noise_cuda, "Batched add noise kernel");
}