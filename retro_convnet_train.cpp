// Step 4 pipeline-proof: small conv+pool+dense net training on a batch of real
// stable-retro Sonic 3 & Knuckles frames + a placeholder reward-ish target, using the
// same Vulkan+CPU-fallback ggml_backend_sched setup validated in steps 1-2.
//
// Reads a raw binary file written by the Python driver:
//   [NBATCH * IH * IW float32 pixels, row-major, grayscale 0..1] [NBATCH float32 targets]
// Trains a few epochs, prints the loss curve, exits.
//
// Usage: ./retro_convnet_train <batch_file> <IW> <IH> <nbatch> <nepoch>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <batch_file> <IW> <IH> <nbatch> <nepoch>\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int64_t IW = std::atoll(argv[2]);
    int64_t IH = std::atoll(argv[3]);
    int64_t NBATCH = std::atoll(argv[4]);
    int64_t nepoch = std::atoll(argv[5]);
    const int64_t IC = 1, OC1 = 4, NE_OUT = 1;
    const int64_t PW = IW/2, PH = IH/2; // after one 2x2 avgpool

    int64_t NE_IN = IW*IH*IC;
    std::vector<float> all_data((NE_IN + NE_OUT) * NBATCH);
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
        std::vector<float> pixels(NE_IN*NBATCH);
        std::vector<float> targets(NBATCH);
        f.read(reinterpret_cast<char*>(pixels.data()), pixels.size()*sizeof(float));
        f.read(reinterpret_cast<char*>(targets.data()), targets.size()*sizeof(float));
        if (!f) { fprintf(stderr, "short read from %s\n", path.c_str()); return 1; }

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

        ggml_opt_dataset_t dataset = ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, NE_IN, NE_OUT, NBATCH, 1);
        std::memcpy(ggml_get_data_f32(ggml_opt_dataset_data(dataset)), pixels.data(), pixels.size()*sizeof(float));
        std::memcpy(ggml_get_data_f32(ggml_opt_dataset_labels(dataset)), targets.data(), targets.size()*sizeof(float));

        struct ggml_context * ctx_static;
        struct ggml_context * ctx_compute;
        { struct ggml_init_params p = { 64*ggml_tensor_overhead(), nullptr, true }; ctx_static = ggml_init(p); }
        { struct ggml_init_params p = { GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true }; ctx_compute = ggml_init(p); }

        struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NE_IN, NBATCH);
        ggml_set_name(inputs, "inputs");
        struct ggml_tensor * kernel = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, IC, OC1);
        struct ggml_tensor * w_d1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, PW*PH*OC1, 32);
        struct ggml_tensor * b_d1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 32);
        struct ggml_tensor * w_d2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 32, NE_OUT);
        struct ggml_tensor * b_d2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NE_OUT);
        for (ggml_tensor * t : {kernel, w_d1, b_d1, w_d2, b_d2}) ggml_set_param(t);

        int64_t nparams = ggml_nelements(kernel)+ggml_nelements(w_d1)+ggml_nelements(b_d1)+ggml_nelements(w_d2)+ggml_nelements(b_d2);

        struct ggml_tensor * img = ggml_reshape_4d(ctx_compute, inputs, IW, IH, IC, NBATCH);
        struct ggml_tensor * conv = ggml_conv_2d(ctx_compute, kernel, img, 1, 1, 1, 1, 1, 1);
        struct ggml_tensor * act  = ggml_silu(ctx_compute, conv);
        struct ggml_tensor * pooled = ggml_pool_2d(ctx_compute, act, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
        struct ggml_tensor * flat = ggml_reshape_2d(ctx_compute, ggml_cont(ctx_compute, pooled), PW*PH*OC1, NBATCH);
        struct ggml_tensor * h1 = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d1, flat), b_d1));
        struct ggml_tensor * outputs = ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d2, h1), b_d2);
        ggml_set_name(outputs, "outputs");

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend_vulkan);
        if (!buf) { fprintf(stderr, "weight buffer alloc failed\n"); return 2; }

        {
            std::mt19937 rng(17);
            auto init_tensor = [&](ggml_tensor * t, float scale) {
                int64_t n = ggml_nelements(t);
                std::vector<float> host(n);
                std::normal_distribution<float> nd(0.0f, scale);
                for (int64_t i = 0; i < n; ++i) host[i] = nd(rng);
                ggml_backend_tensor_set(t, host.data(), 0, n*sizeof(float));
            };
            init_tensor(kernel, 0.3f); init_tensor(w_d1, 0.1f); init_tensor(b_d1, 0.0f);
            init_tensor(w_d2, 0.1f); init_tensor(b_d2, 0.0f);
        }

        printf("Net: conv2d(1->%lld,3x3) -> SiLU -> avgpool2x2 -> dense(%lld->32->1), nparams=%lld\n",
               (long long)OC1, (long long)(PW*PH*OC1), (long long)nparams);
        printf("Input frame: %lldx%lld (downsampled real Sonic 3 & Knuckles grayscale frames via stable-retro)\n",
               (long long)IW, (long long)IH);
        printf("Target: placeholder reward-ish scalar (NOT the final reward design - proving the pipeline only)\n");
        printf("nbatch=%lld nepoch=%lld\n\n", (long long)NBATCH, (long long)nepoch);

        ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
                     GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                     ggml_opt_get_default_optimizer_params, nepoch, NBATCH, 0.0f, false);

        printf("\nDone - retro-frame training step completed on Vulkan+CPU-fallback backend_sched.\n");

        ggml_opt_dataset_free(dataset);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_static);
        ggml_free(ctx_compute);
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend_vulkan);
        ggml_backend_free(backend_cpu);
    }
    return 0;
}
