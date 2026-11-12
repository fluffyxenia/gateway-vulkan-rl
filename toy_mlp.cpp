// Minimal custom training loop against ggml's C API, explicitly targeting the Vulkan
// backend (with CPU as the ggml_backend_sched fallback for any op Vulkan doesn't
// implement). Step 1 validation for the gateway-vulkan-rl pathfinder project.
//
// Task: regress y = sin(x0) + 0.5*cos(x1) from 2 inputs, with a 3-layer MLP using
// SiLU activations (ggml has no GELU backward op on any backend, so SiLU/ReLU only).
//
// Usage: ./toy_mlp [nepoch]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static const int64_t NE_IN    = 2;
static const int64_t NE_OUT   = 1;
static const int64_t NHIDDEN  = 32;
static const int64_t NDATA    = 2048;
static const int64_t NBATCH   = 32;

int main(int argc, char ** argv) {
    int64_t nepoch = argc > 1 ? std::atoll(argv[1]) : 40;

    ggml_backend_load_all();

    size_t dev_count = ggml_backend_dev_count();
    printf("Found %zu ggml backend devices:\n", dev_count);
    ggml_backend_dev_t dev_vulkan = nullptr;
    ggml_backend_dev_t dev_cpu    = nullptr;
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        printf("  [%zu] %s - %s\n", i, name.c_str(), ggml_backend_dev_description(dev));
        if (name.find("Vulkan") != std::string::npos && !dev_vulkan) {
            dev_vulkan = dev;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && !dev_cpu) {
            dev_cpu = dev;
        }
    }
    if (!dev_vulkan) {
        fprintf(stderr, "No Vulkan device found - aborting.\n");
        return 1;
    }
    if (!dev_cpu) {
        fprintf(stderr, "No CPU device found - aborting.\n");
        return 1;
    }

    ggml_backend_t backend_vulkan = ggml_backend_dev_init(dev_vulkan, nullptr);
    ggml_backend_t backend_cpu    = ggml_backend_dev_init(dev_cpu, nullptr);
    GGML_ASSERT(backend_vulkan && backend_cpu);

    // Vulkan first so the scheduler prefers it; CPU is the fallback for any
    // unsupported op.
    ggml_backend_t backends[2] = { backend_vulkan, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    // ---- build the dataset ----
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, NE_IN, NE_OUT, NDATA, /*ndata_shard=*/1);

    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> unif(-3.14159f, 3.14159f);
        float * data   = ggml_get_data_f32(ggml_opt_dataset_data(dataset));
        float * labels = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));
        for (int64_t i = 0; i < NDATA; ++i) {
            float x0 = unif(rng);
            float x1 = unif(rng);
            data[i*NE_IN + 0] = x0;
            data[i*NE_IN + 1] = x1;
            labels[i*NE_OUT + 0] = std::sin(x0) + 0.5f*std::cos(x1);
        }
    }

    // ---- build the model graph ----
    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params p = { /*mem_size=*/64*ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        ctx_static = ggml_init(p);
    }
    {
        struct ggml_init_params p = {
            /*mem_size=*/ GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(),
            nullptr, /*no_alloc=*/true,
        };
        ctx_compute = ggml_init(p);
    }

    struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NE_IN, NBATCH);
    ggml_set_name(inputs, "inputs");

    struct ggml_tensor * w1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NE_IN, NHIDDEN);
    struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NHIDDEN);
    struct ggml_tensor * w2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NHIDDEN, NHIDDEN);
    struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NHIDDEN);
    struct ggml_tensor * w3 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NHIDDEN, NE_OUT);
    struct ggml_tensor * b3 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NE_OUT);
    for (ggml_tensor * t : {w1, b1, w2, b2, w3, b3}) {
        ggml_set_param(t);
    }

    struct ggml_tensor * h1 = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w1, inputs), b1));
    struct ggml_tensor * h2 = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w2, h1), b2));
    struct ggml_tensor * outputs = ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w3, h2), b3);
    ggml_set_name(outputs, "outputs");

    // Allocate weights + inputs on the Vulkan backend's buffer type.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend_vulkan);
    GGML_ASSERT(buf);
    printf("Weights/inputs buffer type: %s\n", ggml_backend_buffer_name(buf));

    // Random init (host side, then upload).
    {
        std::mt19937 rng(1234);
        auto init_tensor = [&](ggml_tensor * t, float scale) {
            int64_t n = ggml_nelements(t);
            std::vector<float> host(n);
            std::normal_distribution<float> nd(0.0f, scale);
            for (int64_t i = 0; i < n; ++i) host[i] = nd(rng);
            ggml_backend_tensor_set(t, host.data(), 0, n*sizeof(float));
        };
        init_tensor(w1, 0.5f); init_tensor(b1, 0.0f);
        init_tensor(w2, 0.3f); init_tensor(b2, 0.0f);
        init_tensor(w3, 0.3f); init_tensor(b3, 0.0f);
    }

    struct ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    opt_params.ctx_compute = ctx_compute;
    opt_params.inputs      = inputs;
    opt_params.outputs     = outputs;
    opt_params.optimizer   = GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    printf("\n=== Training MLP (2 -> %lld -> %lld -> 1, SiLU) on Vulkan0 (CPU fallback available) ===\n",
           (long long)NHIDDEN, (long long)NHIDDEN);
    printf("ndata=%lld batch=%lld nepoch=%lld\n\n", (long long)NDATA, (long long)NBATCH, (long long)nepoch);

    ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
                 GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                 ggml_opt_get_default_optimizer_params, nepoch, NBATCH, /*val_split=*/0.1f, /*silent=*/false);

    printf("\nDone.\n");

    ggml_opt_dataset_free(dataset);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_vulkan);
    ggml_backend_free(backend_cpu);

    return 0;
}
