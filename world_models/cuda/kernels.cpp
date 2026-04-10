// kernels.cpp - C++ wrappers for CUDA kernels

#include <torch/extension.h>

void batched_normalize_kernel(float* data, int batch_size, int channels, int height, int width);
void batched_add_noise_kernel(float* data, const float* noise, int total_elements);

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