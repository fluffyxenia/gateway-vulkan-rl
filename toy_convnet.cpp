// Step 2 functional test: a small conv+pool+dense net trained via ggml_opt_fit with
// backend_sched = [Vulkan, CPU]. IM2COL_BACK and POOL_2D_BACK are NOT implemented on
// Vulkan (confirmed via probe_ops) - this test checks whether ggml_backend_sched
// silently/automatically routes those backward ops to the CPU backend and the whole
// forward+backward+optimizer-step pipeline still runs to completion with a sane loss
// curve, as opposed to erroring out or hanging.

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

static const int64_t IW = 8, IH = 8, IC = 1;
static const int64_t OC1 = 4;      // conv output channels
static const int64_t NE_IN  = IW*IH*IC;
static const int64_t NE_OUT = 1;
static const int64_t NDATA  = 2048;
static const int64_t NBATCH = 16;

int main(int argc, char ** argv) {
    int64_t nepoch = argc > 1 ? std::atoll(argv[1]) : 30;

    ggml_backend_load_all();
    size_t dev_count = ggml_backend_dev_count();
    ggml_backend_dev_t dev_vulkan = nullptr, dev_cpu = nullptr;
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        if (name.find("Vulkan") != std::string::npos && !dev_vulkan) dev_vulkan = dev;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && !dev_cpu) dev_cpu = dev;
    }
    if (!dev_vulkan || !dev_cpu) { fprintf(stderr, "missing device\n"); return 1; }

    ggml_backend_t backend_vulkan = ggml_backend_dev_init(dev_vulkan, nullptr);
    ggml_backend_t backend_cpu    = ggml_backend_dev_init(dev_cpu, nullptr);
    ggml_backend_t backends[2] = { backend_vulkan, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    // ---- dataset: image -> mean of top-left 4x4 quadrant ----
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, NE_IN, NE_OUT, NDATA, /*ndata_shard=*/1);
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> unif(0.0f, 1.0f);
        float * data   = ggml_get_data_f32(ggml_opt_dataset_data(dataset));
        float * labels = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));
        for (int64_t i = 0; i < NDATA; ++i) {
            float quad_sum = 0.0f;
            for (int64_t y = 0; y < IH; ++y) {
                for (int64_t x = 0; x < IW; ++x) {
                    float v = unif(rng);
                    data[i*NE_IN + y*IW + x] = v;
                    if (x < 4 && y < 4) quad_sum += v;
                }
            }
            labels[i*NE_OUT + 0] = quad_sum / 16.0f;
        }
    }

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params p = { 64*ggml_tensor_overhead(), nullptr, true };
        ctx_static = ggml_init(p);
    }
    {
        struct ggml_init_params p = {
            GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true,
        };
        ctx_compute = ggml_init(p);
    }

    struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NE_IN, NBATCH);
    ggml_set_name(inputs, "inputs");

    struct ggml_tensor * kernel = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, IC, OC1);
    struct ggml_tensor * w_d1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 4*4*OC1, 16);
    struct ggml_tensor * b_d1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 16);
    struct ggml_tensor * w_d2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 16, NE_OUT);
    struct ggml_tensor * b_d2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NE_OUT);
    for (ggml_tensor * t : {kernel, w_d1, b_d1, w_d2, b_d2}) ggml_set_param(t);

    struct ggml_tensor * img = ggml_reshape_4d(ctx_compute, inputs, IW, IH, IC, NBATCH);
    struct ggml_tensor * conv = ggml_conv_2d(ctx_compute, kernel, img, 1, 1, 1, 1, 1, 1); // -> [8,8,OC1,N]
    struct ggml_tensor * act  = ggml_silu(ctx_compute, conv);
    struct ggml_tensor * pooled = ggml_pool_2d(ctx_compute, act, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f); // -> [4,4,OC1,N]
    struct ggml_tensor * flat = ggml_reshape_2d(ctx_compute, ggml_cont(ctx_compute, pooled), 4*4*OC1, NBATCH);
    struct ggml_tensor * h1 = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d1, flat), b_d1));
    struct ggml_tensor * outputs = ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d2, h1), b_d2);
    ggml_set_name(outputs, "outputs");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend_vulkan);
    GGML_ASSERT(buf);

    {
        std::mt19937 rng(99);
        auto init_tensor = [&](ggml_tensor * t, float scale) {
            int64_t n = ggml_nelements(t);
            std::vector<float> host(n);
            std::normal_distribution<float> nd(0.0f, scale);
            for (int64_t i = 0; i < n; ++i) host[i] = nd(rng);
            ggml_backend_tensor_set(t, host.data(), 0, n*sizeof(float));
        };
        init_tensor(kernel, 0.3f);
        init_tensor(w_d1, 0.3f); init_tensor(b_d1, 0.0f);
        init_tensor(w_d2, 0.3f); init_tensor(b_d2, 0.0f);
    }

    printf("Found %zu ggml backend devices (Vulkan0=%p CPU=%p)\n", dev_count, (void*)backend_vulkan, (void*)backend_cpu);
    printf("\n=== Training conv+pool+dense net (conv2d 1->%lld ch, avgpool2x2, dense %lld->16->1) ===\n",
           (long long)OC1, (long long)(4*4*OC1));
    printf("IM2COL_BACK/POOL_2D_BACK are Vulkan-unsupported per probe_ops - expecting silent CPU fallback via ggml_backend_sched\n");
    printf("ndata=%lld batch=%lld nepoch=%lld\n\n", (long long)NDATA, (long long)NBATCH, (long long)nepoch);

    ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
                 GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                 ggml_opt_get_default_optimizer_params, nepoch, NBATCH, 0.1f, false);

    printf("\nDone - no crash, pipeline completed end to end.\n");
    printf("n_splits in scheduler (number of backend hand-offs in the graph): %d\n", ggml_backend_sched_get_n_splits(sched));

    ggml_opt_dataset_free(dataset);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_vulkan);
    ggml_backend_free(backend_cpu);
    return 0;
}
