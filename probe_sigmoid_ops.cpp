#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <string>

static void check(ggml_backend_dev_t dev, const char* name, const char* op, struct ggml_tensor* t) {
    printf("  %-10s %-14s: %s\n", name, op, ggml_backend_dev_supports_op(dev, t) ? "SUPPORTED" : "NOT supported");
}

int main() {
    ggml_backend_load_all();
    size_t n = ggml_backend_dev_count();
    ggml_backend_dev_t vk=nullptr, cpu=nullptr;
    for (size_t i=0;i<n;i++){
        auto d = ggml_backend_dev_get(i);
        std::string nm = ggml_backend_dev_name(d);
        if (nm.find("Vulkan")!=std::string::npos && !vk) vk=d;
        if (ggml_backend_dev_type(d)==GGML_BACKEND_DEVICE_TYPE_CPU && !cpu) cpu=d;
    }
    struct ggml_init_params p = {1<<16, nullptr, true};
    struct ggml_context* ctx = ggml_init(p);
    auto a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    auto exp_t = ggml_exp(ctx, a);
    auto div_t = ggml_div(ctx, a, b);
    auto add_t = ggml_add(ctx, a, b);
    auto neg_t = ggml_neg(ctx, a);
    check(vk, "Vulkan0", "EXP", exp_t);
    check(cpu, "CPU", "EXP", exp_t);
    check(vk, "Vulkan0", "DIV", div_t);
    check(cpu, "CPU", "DIV", div_t);
    check(vk, "Vulkan0", "ADD", add_t);
    check(cpu, "CPU", "ADD", add_t);
    check(vk, "Vulkan0", "NEG", neg_t);
    check(cpu, "CPU", "NEG", neg_t);
    return 0;
}
