// Cross-validation tool (v3, two-conv): loads a weights file (same format as
// sonic_policy_train.cpp v3) and a single frame from a raw binary file, runs ONLY the
// forward pass (no training), and prints the resulting per-button probabilities - to be
// diffed against policy_net.py's numpy forward pass on the exact same weights+frame, to
// confirm the two implementations agree (tensor layout assumptions correct) before
// trusting either for real training/rollout.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    std::string weights_path = argv[1];
    std::string frame_path = argv[2];
    int64_t IW = std::atoll(argv[3]), IH = std::atoll(argv[4]), N_BUTTONS = std::atoll(argv[5]);
    const int64_t IC=1, OC1=8, OC2=16;
    const int64_t PW1=IW/2, PH1=IH/2, PW2=PW1/2, PH2=PH1/2, FLAT=PW2*PH2*OC2, NHIDDEN=64;

    ggml_backend_load_all();
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    struct ggml_init_params p = { 64*ggml_tensor_overhead(), nullptr, true };
    struct ggml_context * ctx_static = ggml_init(p);
    struct ggml_init_params p2 = { GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    struct ggml_context * ctx_compute = ggml_init(p2);

    struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, IW*IH*IC, 1);
    struct ggml_tensor * kernel1 = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, IC, OC1);
    struct ggml_tensor * kernel2 = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, OC1, OC2);
    struct ggml_tensor * w_d1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, FLAT, NHIDDEN);
    struct ggml_tensor * b_d1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NHIDDEN);
    struct ggml_tensor * w_d2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NHIDDEN, N_BUTTONS);
    struct ggml_tensor * b_d2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, N_BUTTONS);

    struct ggml_tensor * img   = ggml_reshape_4d(ctx_compute, inputs, IW, IH, IC, 1);
    struct ggml_tensor * conv1 = ggml_conv_2d(ctx_compute, kernel1, img, 1, 1, 1, 1, 1, 1);
    struct ggml_tensor * act1  = ggml_silu(ctx_compute, conv1);
    struct ggml_tensor * pool1 = ggml_pool_2d(ctx_compute, act1, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
    struct ggml_tensor * conv2 = ggml_conv_2d(ctx_compute, kernel2, pool1, 1, 1, 1, 1, 1, 1);
    struct ggml_tensor * act2  = ggml_silu(ctx_compute, conv2);
    struct ggml_tensor * pool2 = ggml_pool_2d(ctx_compute, act2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
    struct ggml_tensor * flat  = ggml_reshape_2d(ctx_compute, ggml_cont(ctx_compute, pool2), FLAT, 1);
    struct ggml_tensor * h1    = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d1, flat), b_d1));
    struct ggml_tensor * logits= ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d2, h1), b_d2);
    // Forward-only tool, no backward graph needed, so plain ggml_sigmoid() is fine here
    // (the "no backward for sigmoid" limitation only applies when the graph must be
    // differentiated - see sonic_policy_train.cpp's manual construction for that case).
    struct ggml_tensor * probs = ggml_sigmoid(ctx_compute, logits);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend_cpu);

    int64_t total = ggml_nelements(kernel1)+ggml_nelements(kernel2)+ggml_nelements(w_d1)+ggml_nelements(b_d1)+ggml_nelements(w_d2)+ggml_nelements(b_d2);
    std::vector<float> host(total);
    { std::ifstream wf(weights_path, std::ios::binary); wf.read((char*)host.data(), total*sizeof(float)); }
    size_t off = 0;
    auto load_t = [&](ggml_tensor * t) { int64_t n = ggml_nelements(t); ggml_backend_tensor_set(t, host.data()+off, 0, n*sizeof(float)); off += n; };
    load_t(kernel1); load_t(kernel2); load_t(w_d1); load_t(b_d1); load_t(w_d2); load_t(b_d2);

    std::vector<float> frame(IW*IH);
    { std::ifstream ff(frame_path, std::ios::binary); ff.read((char*)frame.data(), frame.size()*sizeof(float)); }
    ggml_backend_tensor_set(inputs, frame.data(), 0, frame.size()*sizeof(float));

    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend_cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(gf, probs);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(inputs, frame.data(), 0, frame.size()*sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> out(N_BUTTONS);
    ggml_backend_tensor_get(probs, out.data(), 0, N_BUTTONS*sizeof(float));
    printf("ggml forward probs:");
    for (auto v : out) printf(" %.6f", v);
    printf("\n");
    return 0;
}
