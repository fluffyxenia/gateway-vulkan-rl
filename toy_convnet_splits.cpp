// Surgical follow-up to toy_convnet: directly compares ggml_backend_sched split counts
// for a forward-only graph vs a forward+backward graph on the same conv+pool+dense net,
// using the low-level ggml_opt_alloc/ggml_opt_eval API so we control exactly which graph
// is being measured (ggml_opt_fit's aggregate run mixes forward-only validation passes
// with forward+backward training passes, which confounded a naive single n_splits read).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const int64_t IW = 8, IH = 8, IC = 1, OC1 = 4;
static const int64_t NE_IN = IW*IH*IC, NE_OUT = 1, NBATCH = 16;

int main() {
    ggml_backend_load_all();
    size_t dev_count = ggml_backend_dev_count();
    ggml_backend_dev_t dev_vulkan = nullptr, dev_cpu = nullptr;
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        if (name.find("Vulkan") != std::string::npos && !dev_vulkan) dev_vulkan = dev;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && !dev_cpu) dev_cpu = dev;
    }
    ggml_backend_t backend_vulkan = ggml_backend_dev_init(dev_vulkan, nullptr);
    ggml_backend_t backend_cpu    = ggml_backend_dev_init(dev_cpu, nullptr);
    ggml_backend_t backends[2] = { backend_vulkan, backend_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    { struct ggml_init_params p = { 64*ggml_tensor_overhead(), nullptr, true }; ctx_static = ggml_init(p); }
    { struct ggml_init_params p = { GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true }; ctx_compute = ggml_init(p); }

    struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NE_IN, NBATCH);
    ggml_set_name(inputs, "inputs");
    struct ggml_tensor * kernel = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, IC, OC1);
    struct ggml_tensor * w_d1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 4*4*OC1, 16);
    struct ggml_tensor * b_d1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 16);
    struct ggml_tensor * w_d2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 16, NE_OUT);
    struct ggml_tensor * b_d2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NE_OUT);
    for (ggml_tensor * t : {kernel, w_d1, b_d1, w_d2, b_d2}) ggml_set_param(t);

    struct ggml_tensor * img = ggml_reshape_4d(ctx_compute, inputs, IW, IH, IC, NBATCH);
    struct ggml_tensor * conv = ggml_conv_2d(ctx_compute, kernel, img, 1, 1, 1, 1, 1, 1);
    struct ggml_tensor * act  = ggml_silu(ctx_compute, conv);
    struct ggml_tensor * pooled = ggml_pool_2d(ctx_compute, act, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
    struct ggml_tensor * flat = ggml_reshape_2d(ctx_compute, ggml_cont(ctx_compute, pooled), 4*4*OC1, NBATCH);
    struct ggml_tensor * h1 = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d1, flat), b_d1));
    struct ggml_tensor * outputs = ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d2, h1), b_d2);
    ggml_set_name(outputs, "outputs");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend_vulkan);
    {
        std::mt19937 rng(1);
        auto init_tensor = [&](ggml_tensor * t, float scale) {
            int64_t n = ggml_nelements(t);
            std::vector<float> host(n);
            std::normal_distribution<float> nd(0.0f, scale);
            for (int64_t i = 0; i < n; ++i) host[i] = nd(rng);
            ggml_backend_tensor_set(t, host.data(), 0, n*sizeof(float));
        };
        init_tensor(kernel, 0.3f); init_tensor(w_d1, 0.3f); init_tensor(b_d1, 0.0f);
        init_tensor(w_d2, 0.3f); init_tensor(b_d2, 0.0f);
    }

    struct ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    opt_params.ctx_compute = ctx_compute;
    opt_params.inputs = inputs;
    opt_params.outputs = outputs;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    ggml_opt_context_t opt_ctx = ggml_opt_init(opt_params);

    // dummy batch data
    {
        std::vector<float> host(NE_IN*NBATCH, 0.5f);
        ggml_backend_tensor_set(ggml_opt_inputs(opt_ctx), host.data(), 0, host.size()*sizeof(float));
    }

    // Forward-only pass
    ggml_opt_alloc(opt_ctx, /*backward=*/false);
    ggml_opt_eval(opt_ctx, nullptr);
    int n_splits_fwd = ggml_backend_sched_get_n_splits(sched);
    printf("Forward-only graph: n_splits = %d\n", n_splits_fwd);

    // Forward+backward pass
    ggml_opt_alloc(opt_ctx, /*backward=*/true);
    ggml_opt_eval(opt_ctx, nullptr);
    int n_splits_bwd = ggml_backend_sched_get_n_splits(sched);
    printf("Forward+backward graph: n_splits = %d\n", n_splits_bwd);

    printf("\n%s\n", n_splits_bwd > n_splits_fwd
        ? "CONFIRMED: backward graph required more backend splits than forward-only -> IM2COL_BACK/POOL_2D_BACK triggered a real Vulkan<->CPU handoff, not single-backend execution."
        : "Split counts equal - backward graph did NOT require extra backend hand-offs (inconclusive/contradicts hybrid-fallback theory as tested).");

    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_vulkan);
    ggml_backend_free(backend_cpu);
    return 0;
}
