// Directly query ggml_backend_dev_supports_op() for the specific backward-pass /
// optimizer ops this project's background research claimed are Vulkan-supported or not.
// This is a ground-truth check independent of any full training run: it asks the
// backend itself "can you execute this op" rather than inferring it from timing.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>

static void check(ggml_backend_dev_t dev, const char * devname, const char * opname, struct ggml_tensor * t) {
    bool ok = ggml_backend_dev_supports_op(dev, t);
    printf("  %-10s %-20s : %s\n", devname, opname, ok ? "SUPPORTED" : "NOT supported");
}

int main() {
    ggml_backend_load_all();
    size_t dev_count = ggml_backend_dev_count();
    ggml_backend_dev_t dev_vulkan = nullptr;
    ggml_backend_dev_t dev_cpu = nullptr;
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        if (name.find("Vulkan") != std::string::npos && !dev_vulkan) dev_vulkan = dev;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && !dev_cpu) dev_cpu = dev;
    }
    if (!dev_vulkan || !dev_cpu) { fprintf(stderr, "missing device\n"); return 1; }

    struct ggml_init_params p = { /*mem_size=*/ 1 << 20, nullptr, /*no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(p);

    auto t1 = [&](int64_t n0) { return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0); };
    auto t2 = [&](int64_t n0, int64_t n1) { return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1); };
    auto t4 = [&](int64_t n0, int64_t n1, int64_t n2, int64_t n3) { return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, n2, n3); };

    // OUT_PROD: a[m,n], b[p,n] -> out[m,p]  (this is what matmul's gradient uses)
    struct ggml_tensor * out_prod = ggml_out_prod(ctx, t2(8, 4), t2(8, 4));

    // OUT_PROD with F16 operand 'a' - probing the "F32-only" claim
    struct ggml_tensor * a_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
    struct ggml_tensor * out_prod_f16 = ggml_out_prod(ctx, a_f16, t2(8, 4));

    // SILU_BACK: a=dy, b=x, same shape
    struct ggml_tensor * silu_back = ggml_silu_back(ctx, t1(16), t1(16));

    // RMS_NORM_BACK: a=x, b=dy, same shape
    struct ggml_tensor * rms_back = ggml_rms_norm_back(ctx, t1(16), t1(16), 1e-5f);

    // GET_ROWS_BACK: a=grad of get_rows result [n_embd, n_rows], b=row indices [n_rows] (I32), c=data shape [n_embd, n_rows_total]
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    struct ggml_tensor * get_rows_back = ggml_get_rows_back(ctx, t2(8, 4), idx, t2(8, 10));

    // SOFT_MAX_BACK (via ggml_soft_max_ext_back): a, b same shape
    struct ggml_tensor * softmax_back = ggml_soft_max_ext_back(ctx, t2(8, 4), t2(8, 4), 1.0f, 0.0f);

    // IM2COL_BACK: a=kernel [KW,KH,IC,OC], b=grad of im2col output, ne=shape of im2col input (the data tensor b in forward)
    struct ggml_tensor * kernel = t4(3, 3, 1, 4);
    // forward im2col output for stride1/pad1/dil1 2D on an 8x8x1 input with this kernel: [KW*KH*IC, OW, OH, N] = [9, 8, 8, 1]
    struct ggml_tensor * im2col_grad = t4(9, 8, 8, 1);
    int64_t ne_data[4] = {8, 8, 1, 1}; // shape of the original data tensor fed to im2col
    struct ggml_tensor * im2col_back = ggml_im2col_back(ctx, kernel, im2col_grad, ne_data, 1, 1, 1, 1, 1, 1, true);

    // POOL_2D_BACK: a=grad of pool output, af=original input to pool (forward), same pool params
    struct ggml_tensor * pool_in  = t4(8, 8, 4, 1);
    struct ggml_tensor * pool_out_grad = t4(4, 4, 4, 1);
    struct ggml_tensor * pool_back = ggml_pool_2d_back(ctx, pool_out_grad, pool_in, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);

    // OPT_STEP_ADAMW / OPT_STEP_SGD
    struct ggml_tensor * w  = t1(16);
    ggml_set_param(w);
    struct ggml_tensor * gr = t1(16);
    struct ggml_tensor * m  = t1(16);
    struct ggml_tensor * v  = t1(16);
    struct ggml_tensor * adamw_params = t1(7);
    struct ggml_tensor * opt_adamw = ggml_opt_step_adamw(ctx, w, gr, m, v, adamw_params);
    struct ggml_tensor * sgd_params = t1(2);
    struct ggml_tensor * opt_sgd = ggml_opt_step_sgd(ctx, w, gr, sgd_params);

    struct { const char * name; struct ggml_tensor * t; } ops[] = {
        {"OUT_PROD",      out_prod},
        {"OUT_PROD_F16",  out_prod_f16},
        {"SILU_BACK",     silu_back},
        {"RMS_NORM_BACK", rms_back},
        {"GET_ROWS_BACK", get_rows_back},
        {"SOFT_MAX_BACK", softmax_back},
        {"IM2COL_BACK",   im2col_back},
        {"POOL_2D_BACK",  pool_back},
        {"OPT_STEP_ADAMW",opt_adamw},
        {"OPT_STEP_SGD",  opt_sgd},
    };

    printf("Vulkan device: %s\nCPU device:    %s\n\n", ggml_backend_dev_description(dev_vulkan), ggml_backend_dev_description(dev_cpu));
    for (auto & o : ops) {
        check(dev_vulkan, "Vulkan0", o.name, o.t);
        check(dev_cpu,    "CPU",    o.name, o.t);
    }

    ggml_free(ctx);
    return 0;
}
