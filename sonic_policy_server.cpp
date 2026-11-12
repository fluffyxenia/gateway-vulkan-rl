// Step 14: unified persistent policy server, replacing BOTH the per-episode
// sonic_policy_train subprocess spawn AND policy_net.py's numpy CPU rollout-inference
// path with one long-lived process holding a single Vulkan context + the live weight
// tensors, serving two request types over stdin(requests)/stdout(binary
// responses)/stderr(human log) for the lifetime of one sonic_rl_train.py run.
//
// Why: observed on the live running process - sonic_policy_train was being spawned
// fresh every single episode (subprocess.run per episode in sonic_rl_train.py),
// paying full process-start + Vulkan device re-init + shader (re-)compile every ~17-18s
// cycle (RSS jumping ~500MB->~3GB with disk-wait each time), while rollout inference ran
// on CPU via numpy despite the model (<1.2MB) being trivially cheap to keep resident on
// the already-open Vulkan device. This file fixes both by keeping ONE process alive:
//   - weight tensors (kernel1,kernel2,w_d1,b_d1,w_d2,b_d2) live in ctx_weights, built
//     ONCE at startup, loaded once from disk (or randomly initialized if missing),
//     allocated once on the Vulkan backend - identical loading code to
//     sonic_policy_train.cpp, copied verbatim for loading-behavior equivalence.
//   - INFER graph (forward-only, batch=1) is ALSO built and allocated exactly ONCE at
//     startup (same forward-pass construction as forward_check.cpp, but on the Vulkan
//     backend instead of CPU, so rollout inference actually uses the GPU). Each INFER
//     request just overwrites the fixed input tensor's data and re-runs
//     ggml_backend_sched_graph_compute on the SAME already-allocated graph - no
//     rebuild, no realloc, no shader recompile after the first call.
//   - TRAIN requests build a FRESH compute context + ggml_opt_context_t sized to that
//     call's minibatch_size (since episode length/minibatch_size can vary call to
//     call), reusing the weight tensors from ctx_weights, and are freed at the end of
//     the call. This is a deliberate choice, not an oversight: AdamW's optimizer state
//     (m/v moment buffers) therefore still resets every TRAIN call, exactly matching
//     the old per-process-per-episode design's behavior - needed for a direct
//     same-loss numeric-equivalence check against the old subprocess-per-episode
//     design. Momentum persisting across episodes would be a real, separate behavior
//     change this file deliberately does NOT make.
//   - The Vulkan device, backend_sched, and ctx_weights buffer are never torn down
//     between calls - only the per-TRAIN-call compute context/graph/optimizer is.
//
// Protocol (all integers/floats little-endian, native x86_64 byte order - no explicit
// byteswap needed since the only client is also x86_64 Python via struct '<'):
//   stdin  request:  1 byte opcode, then opcode-specific payload (see below)
//   stdout response: opcode-specific payload only, flushed immediately after writing
//   stderr: human-readable log lines, same content/style as the old trainer's printf
//           output (epoch/minibatch loss, nan-diagnostics, weights-saved notices) -
//           redirect this to the same kind of log file used for the old per-episode
//           trainer stdout, so existing `grep -ci nan` / `tail` workflows keep working.
//
//   'I' INFER  request:  NE_IN (=IW*IH) float32 = one grayscale frame
//               response: N_BUTTONS float32 = per-button sigmoid probabilities
//   'T' TRAIN  request:  int64 T, int64 minibatch_size, int64 nepoch,
//                        float32 lr, float32 weight_decay,
//                        then T*NE_IN float32 frames, T*N_BUTTONS float32 button masks,
//                        T float32 per-sample coefficients (Step 16: no more
//                        entropy_coef - this field now does double duty as the real
//                        REINFORCE advantage for genuine episode rows, or a small
//                        fixed BC-anchor weight for concatenated real-demo rows; the
//                        server doesn't distinguish the two, that's a caller concern)
//               response: int32 status (0=all finite, 1=non-finite loss seen at least
//                         once this call - matches the old trainer's nan-diag trigger
//                         condition), then nepoch float32 = average pg_loss per epoch
//   'Q' QUIT   request:  no payload, no response - flush weights to disk and exit(0)
//   EOF on stdin is treated the same as an explicit QUIT (flush + exit 0), so a client
//   that simply closes the pipe still leaves weights in a consistent, saved state.
//
// Usage: ./sonic_policy_server <weights_file> <IW> <IH> <n_buttons>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static bool read_exact(void * dst, size_t n) {
    size_t got = fread(dst, 1, n, stdin);
    return got == n;
}
static void write_exact(const void * src, size_t n) {
    fwrite(src, 1, n, stdout);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <weights_file> <IW> <IH> <n_buttons>\n", argv[0]);
        return 1;
    }
    std::string weights_path = argv[1];
    int64_t IW = std::atoll(argv[2]);
    int64_t IH = std::atoll(argv[3]);
    int64_t N_BUTTONS = std::atoll(argv[4]);

    const int64_t IC = 1, OC1 = 8, OC2 = 16;
    const int64_t PW1 = IW/2, PH1 = IH/2;
    const int64_t PW2 = PW1/2, PH2 = PH1/2;
    const int64_t NE_IN = IW*IH*IC;
    const int64_t FLAT = PW2*PH2*OC2;
    const int64_t NHIDDEN = 64;

    // ---- one-time backend/device setup, kept alive for the whole process lifetime ----
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
    // TWO separate scheds, not one shared between INFER and TRAIN - found by a real
    // crash during integration testing: ggml-backend.h's own documentation for
    // ggml_backend_sched_reset() says allocating a *different* graph on a sched
    // "deallocates all tensors that were previously allocated and leaves them with
    // dangling pointers" for whatever graph was allocated on it before. The original
    // one-sched design allocated the INFER graph once at startup, then every TRAIN
    // call's ggml_opt_init()/ggml_opt_alloc() implicitly allocated ITS OWN graph on the
    // SAME sched - silently invalidating the INFER graph's tensors. First INFER call
    // after the first TRAIN call crashed with
    // "ggml-vulkan.cpp:5757: GGML_ASSERT(buffer != nullptr) failed" inside
    // ggml_backend_sched_graph_compute_async - exactly this mechanism. Fix: give INFER
    // and TRAIN their own independent sched, each managing its own graph/allocation
    // bookkeeping; both still point at the same underlying backend_vulkan/backend_cpu
    // physical devices and the same ctx_weights tensors, so there's no duplicated
    // model memory, only duplicated (tiny) scheduler bookkeeping.
    ggml_backend_sched_t sched_infer = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_t sched_train = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    // ---- persistent weight tensors, loaded/initialized ONCE ----
    struct ggml_context * ctx_weights;
    { struct ggml_init_params p = { 16*ggml_tensor_overhead(), nullptr, true }; ctx_weights = ggml_init(p); }
    struct ggml_tensor * kernel1 = ggml_new_tensor_4d(ctx_weights, GGML_TYPE_F32, 3, 3, IC, OC1);
    struct ggml_tensor * kernel2 = ggml_new_tensor_4d(ctx_weights, GGML_TYPE_F32, 3, 3, OC1, OC2);
    struct ggml_tensor * w_d1 = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, FLAT, NHIDDEN);
    struct ggml_tensor * b_d1 = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, NHIDDEN);
    struct ggml_tensor * w_d2 = ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, NHIDDEN, N_BUTTONS);
    struct ggml_tensor * b_d2 = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, N_BUTTONS);
    for (ggml_tensor * t : {kernel1, kernel2, w_d1, b_d1, w_d2, b_d2}) ggml_set_param(t);

    ggml_backend_buffer_t weights_buf = ggml_backend_alloc_ctx_tensors(ctx_weights, backend_vulkan);
    if (!weights_buf) { fprintf(stderr, "weight buffer alloc failed\n"); return 2; }

    const int64_t TOTAL_PARAMS = ggml_nelements(kernel1)+ggml_nelements(kernel2)+ggml_nelements(w_d1)
                                +ggml_nelements(b_d1)+ggml_nelements(w_d2)+ggml_nelements(b_d2);

    auto save_weights = [&]() {
        std::vector<float> host(TOTAL_PARAMS);
        size_t off = 0;
        auto save_t = [&](ggml_tensor * t) {
            int64_t n = ggml_nelements(t);
            ggml_backend_tensor_get(t, host.data()+off, 0, n*sizeof(float));
            off += n;
        };
        save_t(kernel1); save_t(kernel2); save_t(w_d1); save_t(b_d1); save_t(w_d2); save_t(b_d2);
        std::ofstream wf(weights_path, std::ios::binary);
        wf.write(reinterpret_cast<char*>(host.data()), TOTAL_PARAMS*sizeof(float));
        fprintf(stderr, "Weights saved to %s\n", weights_path.c_str());
        fflush(stderr);
    };

    {
        bool loaded = false;
        std::ifstream wf(weights_path, std::ios::binary);
        if (wf) {
            std::vector<float> host(TOTAL_PARAMS);
            wf.read(reinterpret_cast<char*>(host.data()), TOTAL_PARAMS*sizeof(float));
            if (wf) {
                size_t off = 0;
                auto load_t = [&](ggml_tensor * t) {
                    int64_t n = ggml_nelements(t);
                    ggml_backend_tensor_set(t, host.data()+off, 0, n*sizeof(float));
                    off += n;
                };
                load_t(kernel1); load_t(kernel2); load_t(w_d1); load_t(b_d1); load_t(w_d2); load_t(b_d2);
                fprintf(stderr, "Loaded existing weights from %s (%lld params)\n", weights_path.c_str(), (long long)TOTAL_PARAMS);
                loaded = true;
            } else {
                fprintf(stderr, "weights file %s too short, re-initializing randomly\n", weights_path.c_str());
            }
        }
        if (!loaded) {
            std::mt19937 rng(2024);
            auto init_tensor = [&](ggml_tensor * t, float scale) {
                int64_t n = ggml_nelements(t);
                std::vector<float> host(n);
                std::normal_distribution<float> nd(0.0f, scale);
                for (int64_t i = 0; i < n; ++i) host[i] = nd(rng);
                ggml_backend_tensor_set(t, host.data(), 0, n*sizeof(float));
            };
            init_tensor(kernel1, 0.3f); init_tensor(kernel2, 0.15f);
            init_tensor(w_d1, 0.05f); init_tensor(b_d1, 0.0f);
            init_tensor(w_d2, 0.1f); init_tensor(b_d2, 0.0f);
            fprintf(stderr, "No existing weights at %s - randomly initialized %lld params\n", weights_path.c_str(), (long long)TOTAL_PARAMS);
        }
        fflush(stderr);
    }

    // ---- INFER path: built + allocated ONCE, batch=1, reused for every INFER call ----
    struct ggml_context * ctx_infer_static;
    struct ggml_context * ctx_infer_compute;
    { struct ggml_init_params p = { 4*ggml_tensor_overhead(), nullptr, true }; ctx_infer_static = ggml_init(p); }
    { struct ggml_init_params p = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true }; ctx_infer_compute = ggml_init(p); }
    struct ggml_tensor * infer_input = ggml_new_tensor_2d(ctx_infer_static, GGML_TYPE_F32, NE_IN, 1);
    ggml_set_name(infer_input, "infer_input");
    ggml_backend_buffer_t infer_input_buf = ggml_backend_alloc_ctx_tensors(ctx_infer_static, backend_vulkan);
    if (!infer_input_buf) { fprintf(stderr, "infer input buffer alloc failed\n"); return 2; }

    struct ggml_tensor * iimg   = ggml_reshape_4d(ctx_infer_compute, infer_input, IW, IH, IC, 1);
    struct ggml_tensor * iconv1 = ggml_conv_2d(ctx_infer_compute, kernel1, iimg, 1, 1, 1, 1, 1, 1);
    struct ggml_tensor * iact1  = ggml_silu(ctx_infer_compute, iconv1);
    struct ggml_tensor * ipool1 = ggml_pool_2d(ctx_infer_compute, iact1, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
    struct ggml_tensor * iconv2 = ggml_conv_2d(ctx_infer_compute, kernel2, ipool1, 1, 1, 1, 1, 1, 1);
    struct ggml_tensor * iact2  = ggml_silu(ctx_infer_compute, iconv2);
    struct ggml_tensor * ipool2 = ggml_pool_2d(ctx_infer_compute, iact2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
    struct ggml_tensor * iflat  = ggml_reshape_2d(ctx_infer_compute, ggml_cont(ctx_infer_compute, ipool2), FLAT, 1);
    struct ggml_tensor * ih1    = ggml_silu(ctx_infer_compute, ggml_add(ctx_infer_compute, ggml_mul_mat(ctx_infer_compute, w_d1, iflat), b_d1));
    struct ggml_tensor * ilogits= ggml_add(ctx_infer_compute, ggml_mul_mat(ctx_infer_compute, w_d2, ih1), b_d2);
    struct ggml_tensor * iprobs = ggml_sigmoid(ctx_infer_compute, ilogits); // forward-only, plain sigmoid is fine here

    struct ggml_cgraph * gf_infer = ggml_new_graph(ctx_infer_compute);
    ggml_build_forward_expand(gf_infer, iprobs);
    ggml_backend_sched_alloc_graph(sched_infer, gf_infer);  // allocated ONCE - reused for every INFER call below

    fprintf(stderr, "sonic_policy_server ready: IW=%lld IH=%lld N_BUTTONS=%lld FLAT=%lld (persistent INFER graph allocated once, TRAIN graph rebuilt per call)\n",
            (long long)IW, (long long)IH, (long long)N_BUTTONS, (long long)FLAT);
    fflush(stderr);

    std::vector<float> infer_frame(NE_IN);
    std::vector<float> infer_out(N_BUTTONS);

    for (;;) {
        unsigned char opcode;
        if (fread(&opcode, 1, 1, stdin) != 1) {
            fprintf(stderr, "stdin closed (EOF) - treating as QUIT, flushing weights\n");
            save_weights();
            break;
        }

        if (opcode == 'Q') {
            fprintf(stderr, "QUIT received - flushing weights\n");
            save_weights();
            break;
        }

        if (opcode == 'I') {
            if (!read_exact(infer_frame.data(), NE_IN*sizeof(float))) { fprintf(stderr, "short INFER read\n"); break; }
            ggml_backend_tensor_set(infer_input, infer_frame.data(), 0, NE_IN*sizeof(float));
            ggml_backend_sched_graph_compute(sched_infer, gf_infer);  // same allocated graph, every call
            ggml_backend_tensor_get(iprobs, infer_out.data(), 0, N_BUTTONS*sizeof(float));
            write_exact(infer_out.data(), N_BUTTONS*sizeof(float));
            fflush(stdout);
            continue;
        }

        if (opcode == 'T') {
            int64_t T, NBATCH, nepoch;
            float lr, weight_decay;
            if (!read_exact(&T, 8) || !read_exact(&NBATCH, 8) || !read_exact(&nepoch, 8) ||
                !read_exact(&lr, 4) || !read_exact(&weight_decay, 4)) {
                fprintf(stderr, "short TRAIN header read\n"); break;
            }
            std::vector<float> all_frames(NE_IN*T), all_button_mask(N_BUTTONS*T), all_advantage(T);
            if (!read_exact(all_frames.data(), all_frames.size()*sizeof(float)) ||
                !read_exact(all_button_mask.data(), all_button_mask.size()*sizeof(float)) ||
                !read_exact(all_advantage.data(), all_advantage.size()*sizeof(float))) {
                fprintf(stderr, "short TRAIN batch read\n"); break;
            }

            int64_t n_minibatches = NBATCH > 0 ? T / NBATCH : 0;
            if (n_minibatches < 1) {
                fprintf(stderr, "TRAIN: T (%lld) < minibatch_size (%lld), skipping\n", (long long)T, (long long)NBATCH);
                int32_t status = 1;
                write_exact(&status, 4);
                for (int64_t e = 0; e < nepoch; ++e) { float z = 0.0f; write_exact(&z, 4); }
                fflush(stdout);
                continue;
            }

            // ---- fresh per-call compute context + graph, sized to this call's NBATCH ----
            struct ggml_context * ctx_call_static;
            struct ggml_context * ctx_call_compute;
            { struct ggml_init_params p = { 8*ggml_tensor_overhead(), nullptr, true }; ctx_call_static = ggml_init(p); }
            { struct ggml_init_params p = { GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true }; ctx_call_compute = ggml_init(p); }

            struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_call_static, GGML_TYPE_F32, NE_IN, NBATCH);
            ggml_set_name(inputs, "inputs");
            struct ggml_tensor * bmask = ggml_new_tensor_2d(ctx_call_static, GGML_TYPE_F32, N_BUTTONS, NBATCH);
            ggml_set_name(bmask, "button_mask");
            struct ggml_tensor * adv = ggml_new_tensor_2d(ctx_call_static, GGML_TYPE_F32, 1, NBATCH);
            ggml_set_name(adv, "advantage");
            struct ggml_tensor * ones_kb = ggml_new_tensor_2d(ctx_call_static, GGML_TYPE_F32, N_BUTTONS, NBATCH);
            ggml_set_name(ones_kb, "ones_kb");

            struct ggml_tensor * img    = ggml_reshape_4d(ctx_call_compute, inputs, IW, IH, IC, NBATCH);
            struct ggml_tensor * conv1  = ggml_conv_2d(ctx_call_compute, kernel1, img, 1, 1, 1, 1, 1, 1);
            struct ggml_tensor * act1   = ggml_silu(ctx_call_compute, conv1);
            struct ggml_tensor * pool1  = ggml_pool_2d(ctx_call_compute, act1, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
            struct ggml_tensor * conv2  = ggml_conv_2d(ctx_call_compute, kernel2, pool1, 1, 1, 1, 1, 1, 1);
            struct ggml_tensor * act2   = ggml_silu(ctx_call_compute, conv2);
            struct ggml_tensor * pool2  = ggml_pool_2d(ctx_call_compute, act2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
            struct ggml_tensor * flat   = ggml_reshape_2d(ctx_call_compute, ggml_cont(ctx_call_compute, pool2), FLAT, NBATCH);
            struct ggml_tensor * h1     = ggml_silu(ctx_call_compute, ggml_add(ctx_call_compute, ggml_mul_mat(ctx_call_compute, w_d1, flat), b_d1));
            struct ggml_tensor * logits = ggml_add(ctx_call_compute, ggml_mul_mat(ctx_call_compute, w_d2, h1), b_d2);

            // Step 16: entropy bonus REMOVED ENTIRELY as a mechanism (not just its
            // coefficient weakened to near-zero) - two real, independently-verified
            // RL training failures (deterministic-wrong-combo-lock at entropy_coef=
            // 0.003, "hold everything" degradation at entropy_coef=0.01) both turned
            // out consistent with entropy being the wrong kind of regularizer for
            // this task: it rewards being UNPREDICTABLE, which has nothing to do with
            // "explore useful behavior" - mashing and holding buttons isn't a skill,
            // playing the right sequence is. Replaced by a BC-anchor term instead (see
            // sonic_rl_train.py's train_on_episode()): `adv` here now does double duty
            // as a per-sample coefficient - the real standardized REINFORCE advantage
            // for genuine episode rows, and a small fixed anchor weight for a
            // concatenated subsample of real human-demo rows mixed into the SAME
            // batch/gradient step. The C++ loss graph doesn't need to know which rows
            // are which - `bce_action * adv` is correct either way; the anchor-vs-
            // REINFORCE distinction is entirely a Python-side batch-construction
            // concern, not a server protocol concern (entropy_coef has been dropped
            // from the TRAIN wire format entirely, not just set to 0, per the
            // explicit "remove it as a mechanism" instruction).
            struct ggml_tensor * neg_logits    = ggml_neg(ctx_call_compute, logits);
            struct ggml_tensor * abs_logits    = ggml_abs(ctx_call_compute, logits);
            struct ggml_tensor * exp_neg_abs   = ggml_exp(ctx_call_compute, ggml_neg(ctx_call_compute, abs_logits));
            struct ggml_tensor * common_log    = ggml_log(ctx_call_compute, ggml_add(ctx_call_compute, ones_kb, exp_neg_abs));
            struct ggml_tensor * relu_x        = ggml_relu(ctx_call_compute, logits);
            struct ggml_tensor * x_times_a     = ggml_mul(ctx_call_compute, logits, bmask);
            struct ggml_tensor * bce_per_button = ggml_sub(ctx_call_compute,
                ggml_add(ctx_call_compute, relu_x, common_log), x_times_a);
            struct ggml_tensor * bce_action    = ggml_sum_rows(ctx_call_compute, bce_per_button);
            struct ggml_tensor * per_sample_loss = ggml_mul(ctx_call_compute, bce_action, adv);
            ggml_set_name(per_sample_loss, "per_sample_loss_reinforce_plus_bc_anchor");

            ggml_backend_buffer_t call_buf = ggml_backend_alloc_ctx_tensors(ctx_call_static, backend_vulkan);
            if (!call_buf) {
                fprintf(stderr, "TRAIN: per-call buffer alloc failed\n");
                ggml_free(ctx_call_static); ggml_free(ctx_call_compute);
                int32_t status = 1;
                write_exact(&status, 4);
                for (int64_t e = 0; e < nepoch; ++e) { float z = 0.0f; write_exact(&z, 4); }
                fflush(stdout);
                continue;
            }
            { std::vector<float> ones(N_BUTTONS*NBATCH, 1.0f); ggml_backend_tensor_set(ones_kb, ones.data(), 0, ones.size()*sizeof(float)); }

            struct ggml_opt_params opt_params = ggml_opt_default_params(sched_train, GGML_OPT_LOSS_TYPE_MEAN);
            opt_params.ctx_compute = ctx_call_compute;
            opt_params.inputs  = inputs;
            opt_params.outputs = per_sample_loss;
            opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
            struct ggml_opt_optimizer_params lr_params = ggml_opt_get_default_optimizer_params(nullptr);
            lr_params.adamw.alpha = lr;
            lr_params.adamw.wd    = weight_decay;
            opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
            opt_params.get_opt_pars_ud = &lr_params;
            ggml_opt_context_t opt_ctx = ggml_opt_init(opt_params);

            fprintf(stderr, "TRAIN: T=%lld minibatch_size=%lld n_minibatches/epoch=%lld nepoch=%lld lr=%g weight_decay=%g\n",
                    (long long)T, (long long)NBATCH, (long long)n_minibatches, (long long)nepoch, lr, weight_decay);

            std::mt19937 shuffle_rng(42);  // same fixed seed as the old per-process trainer, for equivalence
            std::vector<int64_t> order(T);
            std::iota(order.begin(), order.end(), 0);
            std::vector<float> mb_frames(NE_IN*NBATCH), mb_bmask(N_BUTTONS*NBATCH), mb_adv(NBATCH);

            bool any_nonfinite = false;
            std::vector<float> epoch_avg_losses(nepoch, 0.0f);
            for (int64_t epoch = 0; epoch < nepoch; ++epoch) {
                std::shuffle(order.begin(), order.end(), shuffle_rng);
                double epoch_loss_sum = 0.0;
                for (int64_t mb = 0; mb < n_minibatches; ++mb) {
                    for (int64_t i = 0; i < NBATCH; ++i) {
                        int64_t src = order[mb*NBATCH + i];
                        std::copy_n(all_frames.begin() + src*NE_IN, NE_IN, mb_frames.begin() + i*NE_IN);
                        std::copy_n(all_button_mask.begin() + src*N_BUTTONS, N_BUTTONS, mb_bmask.begin() + i*N_BUTTONS);
                        mb_adv[i] = all_advantage[src];
                    }
                    ggml_opt_alloc(opt_ctx, /*backward=*/true);
                    ggml_backend_tensor_set(ggml_opt_inputs(opt_ctx), mb_frames.data(), 0, mb_frames.size()*sizeof(float));
                    ggml_backend_tensor_set(bmask, mb_bmask.data(), 0, mb_bmask.size()*sizeof(float));
                    ggml_backend_tensor_set(adv,   mb_adv.data(), 0, mb_adv.size()*sizeof(float));
                    ggml_opt_result_t result = ggml_opt_result_init();
                    ggml_opt_eval(opt_ctx, result);
                    double loss, unc;
                    ggml_opt_result_loss(result, &loss, &unc);
                    ggml_opt_result_free(result);

                    if (!std::isfinite(loss)) {
                        any_nonfinite = true;
                        auto max_abs = [&](ggml_tensor * t, const char * name) {
                            int64_t n = ggml_nelements(t);
                            std::vector<float> host(n);
                            ggml_backend_tensor_get(t, host.data(), 0, n*sizeof(float));
                            float m = 0.0f; bool nf = false;
                            for (float v : host) { if (!std::isfinite(v)) nf = true; else m = std::max(m, std::fabs(v)); }
                            fprintf(stderr, "    [nan-diag] %-10s max|finite_val|=%.6g  contains_non_finite=%s\n", name, m, nf ? "YES" : "no");
                        };
                        fprintf(stderr, "  [nan-diag] non-finite loss detected at epoch %lld minibatch %lld - dumping parameter stats:\n",
                                (long long)(epoch+1), (long long)(mb+1));
                        max_abs(kernel1, "kernel1"); max_abs(kernel2, "kernel2");
                        max_abs(w_d1, "w_d1"); max_abs(b_d1, "b_d1");
                        max_abs(w_d2, "w_d2"); max_abs(b_d2, "b_d2");
                    }
                    epoch_loss_sum += loss;
                }
                epoch_avg_losses[epoch] = (float)(epoch_loss_sum / n_minibatches);
                fprintf(stderr, " -> epoch %lld/%lld average pg_loss = %.6f\n", (long long)(epoch+1), (long long)nepoch, epoch_avg_losses[epoch]);
            }
            fflush(stderr);

            ggml_opt_free(opt_ctx);
            ggml_backend_buffer_free(call_buf);
            ggml_free(ctx_call_static);
            ggml_free(ctx_call_compute);

            save_weights();  // keep sonic_policy_weights.bin current after every TRAIN call, same as before

            int32_t status = any_nonfinite ? 1 : 0;
            write_exact(&status, 4);
            write_exact(epoch_avg_losses.data(), epoch_avg_losses.size()*sizeof(float));
            fflush(stdout);
            continue;
        }

        fprintf(stderr, "unknown opcode 0x%02x - exiting\n", opcode);
        break;
    }

    ggml_backend_buffer_free(infer_input_buf);
    ggml_free(ctx_infer_static);
    ggml_free(ctx_infer_compute);
    ggml_backend_buffer_free(weights_buf);
    ggml_free(ctx_weights);
    ggml_backend_sched_free(sched_infer);
    ggml_backend_sched_free(sched_train);
    ggml_backend_free(backend_vulkan);
    ggml_backend_free(backend_cpu);
    return 0;
}
