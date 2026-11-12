// Step 4/5 real-RL trainer, v3: deeper conv backbone (conv1 1->8ch, conv2 8->16ch,
// 288,658 params total - see header comment history for the full architecture change
// log). THIS REVISION adds internal mini-batching, added after discovering a real
// Vulkan memory ceiling during the first real BC-pretraining run against this
// architecture: a single ggml graph built with v3's two-conv backbone + a batch of
// ~4000+ samples triggered "radv/amdgpu: Not enough memory for command submission" /
// "device lost on Vulkan0" - each conv layer's im2col expansion buffer scales with
// batch size, and with TWO conv layers (plus their backward-pass gradient buffers) the
// per-sample memory cost is high enough that v2's old full-batch-at-once approach
// (which worked fine for v2's single conv layer, and for v3 at small synthetic-test
// batch sizes) stops fitting once real data sizes (single demo recordings: ~2000-2500
// frames; the 7200-step RL episode cap used throughout this project) get combined or
// even individually approach a few thousand samples. Empirically bisected: nbatch=3000
// completes correctly, nbatch=4049 crashes - see AGENTS.md "Step 12" for the full
// investigation. This is NOT a bug in the architecture or the autodiff - it is a real,
// expected resource ceiling on this iGPU's ~1GiB VRAM BAR + shared-RAM GTT setup that
// v2's shallower, cheaper-per-sample backbone never approached in practice.
//
// Fix: build ONE fixed-size graph sized by <minibatch_size> (not the full dataset), and
// loop internally over as many minibatches as fit in <total_frames>, across <nepoch>
// full passes over the data, with a fresh random shuffle of sample order each epoch
// (proper mini-batch SGD, not just a workaround) - all within a single process/Vulkan
// context, so the one-time Vulkan/shader warm-up cost is only paid once per process
// invocation, not once per minibatch (which would happen if this were instead solved by
// spawning a fresh subprocess per minibatch).
//
// ggml has NO backward pass for GGML_UNARY_OP_SIGMOID as a standalone op (confirmed by
// reading ggml.c's build_backward switch directly). So sigmoid is built manually from
// ops that DO have backward support: sigmoid(x) = 1 / (1 + exp(-x)) via ggml_neg,
// ggml_exp, ggml_add, ggml_div (all four confirmed forward+backward-supported on both
// Vulkan and CPU via a direct ggml_backend_dev_supports_op probe, see probe_sigmoid_ops.cpp).
//
// Loss: logprob(a) = sum_k[a_k*log(p_k)+(1-a_k)*log(1-p_k)] (multi-label analogue of
// categorical REINFORCE), loss = -mean(logprob(a)*advantage) - entropy_coef*mean(entropy).
//
// Usage: ./sonic_policy_train <weights_file> <batch_file> <IW> <IH> <total_frames> <minibatch_size> <n_buttons> <nepoch> <lr> [entropy_coef]
//   batch_file = [total_frames*IH*IW float32 frames][total_frames*n_buttons float32 multi-hot button masks][total_frames float32 advantages]
//   total_frames: total samples in batch_file. minibatch_size: samples per gradient step/graph (must evenly divide cleanly enough - any remainder samples past the last full minibatch are dropped, see printed note).
//   nepoch: full passes over the (reshuffled each time) dataset - each pass does (total_frames/minibatch_size) gradient steps.
//   weights_file: loaded if it exists (continues training), else randomly initialized; always (over)written at the end.
//   entropy_coef: entropy-bonus weight added to the loss to discourage policy collapse (default 0.02 if omitted).

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

int main(int argc, char ** argv) {
    if (argc < 10) {
        fprintf(stderr, "usage: %s <weights_file> <batch_file> <IW> <IH> <total_frames> <minibatch_size> <n_buttons> <nepoch> <lr> [entropy_coef] [weight_decay]\n", argv[0]);
        return 1;
    }
    std::string weights_path = argv[1];
    std::string batch_path   = argv[2];
    int64_t IW = std::atoll(argv[3]);
    int64_t IH = std::atoll(argv[4]);
    int64_t TOTAL = std::atoll(argv[5]);
    int64_t NBATCH = std::atoll(argv[6]);  // minibatch size - this is what sizes the ggml graph
    int64_t N_BUTTONS = std::atoll(argv[7]);
    int64_t nepoch = std::atoll(argv[8]);
    float lr = std::atof(argv[9]);
    float entropy_coef = argc > 10 ? std::atof(argv[10]) : 0.02f;
    // Weight decay - was implicitly 0.0 (ggml's own AdamW default) through every run
    // before this, which turned out to matter: two separate real BC-pretraining runs
    // against v3 both diverged to NaN around epoch ~20 (at lr=0.01, then again at
    // lr=0.003, ruling out LR alone as the cause) via what looks like unbounded weight/
    // logit growth eventually overflowing a float32 logit to literal inf, at which point
    // `inf * 0` (the button-mask multiply, for any button whose target is 0 that step)
    // is NaN by IEEE754 definition regardless of how numerically careful the rest of the
    // loss formula is - see AGENTS.md "Step 12/13" for the full investigation. A nonzero
    // weight decay directly counteracts unbounded weight growth, the suspected
    // mechanism, which plain LR reduction did not address on its own.
    float weight_decay = argc > 11 ? std::atof(argv[11]) : 0.01f;

    const int64_t IC = 1, OC1 = 8, OC2 = 16;
    const int64_t PW1 = IW/2, PH1 = IH/2;   // after pool1
    const int64_t PW2 = PW1/2, PH2 = PH1/2; // after pool2
    const int64_t NE_IN = IW*IH*IC;
    const int64_t FLAT = PW2*PH2*OC2;       // 20*14*16 = 4480
    const int64_t NHIDDEN = 64;

    int64_t n_minibatches = TOTAL / NBATCH;
    if (n_minibatches < 1) { fprintf(stderr, "total_frames (%lld) < minibatch_size (%lld)\n", (long long)TOTAL, (long long)NBATCH); return 1; }
    int64_t used = n_minibatches * NBATCH;
    if (used != TOTAL) {
        printf("Note: %lld of %lld total frames used (%lld minibatches of %lld); %lld leftover frames dropped (not evenly divisible)\n",
               (long long)used, (long long)TOTAL, (long long)n_minibatches, (long long)NBATCH, (long long)(TOTAL - used));
    }

    // ---- read FULL batch file into host memory (host RAM is not the constraint here -
    // only the Vulkan-side per-minibatch compute buffers are) ----
    std::vector<float> all_frames(NE_IN*TOTAL);
    std::vector<float> all_button_mask(N_BUTTONS*TOTAL);
    std::vector<float> all_advantage(TOTAL);
    {
        std::ifstream f(batch_path, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open batch file %s\n", batch_path.c_str()); return 1; }
        f.read(reinterpret_cast<char*>(all_frames.data()), all_frames.size()*sizeof(float));
        f.read(reinterpret_cast<char*>(all_button_mask.data()), all_button_mask.size()*sizeof(float));
        f.read(reinterpret_cast<char*>(all_advantage.data()), all_advantage.size()*sizeof(float));
        if (!f) { fprintf(stderr, "short read from %s\n", batch_path.c_str()); return 1; }
    }

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

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    { struct ggml_init_params p = { 64*ggml_tensor_overhead(), nullptr, true }; ctx_static = ggml_init(p); }
    { struct ggml_init_params p = { GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true }; ctx_compute = ggml_init(p); }

    struct ggml_tensor * inputs = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NE_IN, NBATCH);
    ggml_set_name(inputs, "inputs");
    struct ggml_tensor * bmask = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N_BUTTONS, NBATCH);
    ggml_set_name(bmask, "button_mask");
    struct ggml_tensor * adv = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 1, NBATCH);
    ggml_set_name(adv, "advantage");
    struct ggml_tensor * ones_kb = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N_BUTTONS, NBATCH);
    ggml_set_name(ones_kb, "ones_kb");  // constant 1.0 tensor, for manual "1 - x" via ggml_sub

    struct ggml_tensor * kernel1 = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, IC, OC1);
    struct ggml_tensor * kernel2 = ggml_new_tensor_4d(ctx_static, GGML_TYPE_F32, 3, 3, OC1, OC2);
    struct ggml_tensor * w_d1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, FLAT, NHIDDEN);
    struct ggml_tensor * b_d1 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, NHIDDEN);
    struct ggml_tensor * w_d2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, NHIDDEN, N_BUTTONS);
    struct ggml_tensor * b_d2 = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, N_BUTTONS);
    for (ggml_tensor * t : {kernel1, kernel2, w_d1, b_d1, w_d2, b_d2}) ggml_set_param(t);

    struct ggml_tensor * img    = ggml_reshape_4d(ctx_compute, inputs, IW, IH, IC, NBATCH);
    struct ggml_tensor * conv1  = ggml_conv_2d(ctx_compute, kernel1, img, 1, 1, 1, 1, 1, 1);     // [IW,IH,OC1,N]
    struct ggml_tensor * act1   = ggml_silu(ctx_compute, conv1);
    struct ggml_tensor * pool1  = ggml_pool_2d(ctx_compute, act1, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f); // [PW1,PH1,OC1,N]
    struct ggml_tensor * conv2  = ggml_conv_2d(ctx_compute, kernel2, pool1, 1, 1, 1, 1, 1, 1);    // [PW1,PH1,OC2,N]
    struct ggml_tensor * act2   = ggml_silu(ctx_compute, conv2);
    struct ggml_tensor * pool2  = ggml_pool_2d(ctx_compute, act2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f); // [PW2,PH2,OC2,N]
    struct ggml_tensor * flat   = ggml_reshape_2d(ctx_compute, ggml_cont(ctx_compute, pool2), FLAT, NBATCH);
    struct ggml_tensor * h1     = ggml_silu(ctx_compute, ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d1, flat), b_d1));
    struct ggml_tensor * logits = ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, w_d2, h1), b_d2); // [N_BUTTONS, NBATCH]

    // ---- Numerically stable BCE-from-logits + entropy, NOT manual-sigmoid-then-log ----
    // The first real BC-pretraining run against this architecture hit a real bug here:
    // computing probs = sigmoid(logits) and then taking log(probs)/log(1-probs) directly
    // works until a probability saturates to EXACTLY 0.0 or 1.0 in float32 (increasingly
    // likely the longer/more confidently training proceeds, which v3's extra capacity
    // plus the mini-batching fix's extra gradient-steps-per-"epoch" made actually happen
    // around epoch 20 of a real run) - log(0) = -inf, which poisons the gradient and then
    // (via AdamW's moment buffers) PERMANENTLY NaNs every parameter for the rest of
    // training. ggml_clamp() can't fix this - it has no backward pass either (same
    // category as sigmoid, confirmed by the same ggml.c backward-switch check). The
    // actual fix: never take log() of a probability at all. Use the standard
    // numerically-stable "BCE with logits" identity instead (same one PyTorch's
    // binary_cross_entropy_with_logits uses internally):
    //   softplus(z) = log(1+exp(z)) = relu(z) + log(1+exp(-|z|))   <- stable for any z,
    //     the exp() argument is always <= 0, so it can never overflow, and log(1+positive)
    //     can never collapse to exactly log(0).
    //   BCE(x,a) = softplus(x) - x*a                                 (x=logit, a=0/1 target)
    //   H(x)     = p*softplus(-x) + (1-p)*softplus(x)                (binary entropy, still
    //     needs p=sigmoid(x) itself, which IS safe to compute directly - the saturation
    //     problem was only ever in taking log() of an already-saturated p, not in
    //     computing p itself)
    // BCE(x,a) is exactly -logprob(a), so pg_loss = BCE_action * advantage directly - no
    // extra ggml_neg needed, unlike the old formulation (its sign was handled by hand).
    struct ggml_tensor * neg_logits    = ggml_neg(ctx_compute, logits);
    struct ggml_tensor * abs_logits    = ggml_abs(ctx_compute, logits);
    struct ggml_tensor * exp_neg_abs   = ggml_exp(ctx_compute, ggml_neg(ctx_compute, abs_logits));   // exp(-|x|), always in (0,1]
    struct ggml_tensor * common_log    = ggml_log(ctx_compute, ggml_add(ctx_compute, ones_kb, exp_neg_abs)); // log(1+exp(-|x|)), always finite
    struct ggml_tensor * relu_x        = ggml_relu(ctx_compute, logits);       // relu(x)
    struct ggml_tensor * relu_negx     = ggml_relu(ctx_compute, neg_logits);   // relu(-x)

    struct ggml_tensor * x_times_a     = ggml_mul(ctx_compute, logits, bmask);
    struct ggml_tensor * bce_per_button = ggml_sub(ctx_compute,
        ggml_add(ctx_compute, relu_x, common_log), x_times_a);                 // softplus(x) - x*a, [N_BUTTONS, NBATCH]
    struct ggml_tensor * bce_action    = ggml_sum_rows(ctx_compute, bce_per_button); // [1, NBATCH] == -logprob(action)
    struct ggml_tensor * pg_loss       = ggml_mul(ctx_compute, bce_action, adv);     // [1, NBATCH], sign already correct

    // probs = sigmoid(logits) = 1/(1+exp(-x)), safe to compute directly (only its LOG was the hazard)
    struct ggml_tensor * exp_neg_logits = ggml_exp(ctx_compute, neg_logits);
    struct ggml_tensor * denom         = ggml_add(ctx_compute, ones_kb, exp_neg_logits);
    struct ggml_tensor * probs         = ggml_div(ctx_compute, ones_kb, denom);
    struct ggml_tensor * one_minus_probs = ggml_sub(ctx_compute, ones_kb, probs);
    struct ggml_tensor * ent_per_button = ggml_add(ctx_compute,
        ggml_add(ctx_compute, ggml_mul(ctx_compute, probs, relu_negx), ggml_mul(ctx_compute, one_minus_probs, relu_x)),
        common_log);                                                            // p*softplus(-x) + (1-p)*softplus(x), [N_BUTTONS, NBATCH]
    struct ggml_tensor * entropy       = ggml_sum_rows(ctx_compute, ent_per_button); // [1, NBATCH]
    struct ggml_tensor * per_sample_loss = ggml_sub(ctx_compute, pg_loss, ggml_scale(ctx_compute, entropy, entropy_coef));
    ggml_set_name(per_sample_loss, "per_sample_pg_loss_multilabel_stable");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend_vulkan);
    if (!buf) { fprintf(stderr, "weight buffer alloc failed\n"); return 2; }

    // load or init weights
    {
        int64_t total = ggml_nelements(kernel1)+ggml_nelements(kernel2)+ggml_nelements(w_d1)+ggml_nelements(b_d1)+ggml_nelements(w_d2)+ggml_nelements(b_d2);
        bool loaded = false;
        {
            std::ifstream wf(weights_path, std::ios::binary);
            if (wf) {
                std::vector<float> host(total);
                wf.read(reinterpret_cast<char*>(host.data()), total*sizeof(float));
                if (wf) {
                    size_t off = 0;
                    auto load_t = [&](ggml_tensor * t) {
                        int64_t n = ggml_nelements(t);
                        ggml_backend_tensor_set(t, host.data()+off, 0, n*sizeof(float));
                        off += n;
                    };
                    load_t(kernel1); load_t(kernel2); load_t(w_d1); load_t(b_d1); load_t(w_d2); load_t(b_d2);
                    printf("Loaded existing weights from %s (%lld params)\n", weights_path.c_str(), (long long)total);
                    loaded = true;
                } else {
                    fprintf(stderr, "weights file %s too short, re-initializing randomly\n", weights_path.c_str());
                }
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
            printf("No existing weights at %s - randomly initialized %lld params\n", weights_path.c_str(), (long long)total);
        }
    }

    { std::vector<float> ones(N_BUTTONS*NBATCH, 1.0f); ggml_backend_tensor_set(ones_kb, ones.data(), 0, ones.size()*sizeof(float)); }

    struct ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN);
    opt_params.ctx_compute = ctx_compute;
    opt_params.inputs  = inputs;
    opt_params.outputs = per_sample_loss;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    struct ggml_opt_optimizer_params lr_params = ggml_opt_get_default_optimizer_params(nullptr);
    lr_params.adamw.alpha = lr;
    lr_params.adamw.wd    = weight_decay;
    opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    opt_params.get_opt_pars_ud = &lr_params;

    ggml_opt_context_t opt_ctx = ggml_opt_init(opt_params);

    printf("Policy net (v3, two-conv): conv2d(1->%lld,3x3)->SiLU->avgpool2x2->conv2d(%lld->%lld,3x3)->SiLU->avgpool2x2->dense(%lld->%lld->%lld sigmoid), REINFORCE loss\n",
           (long long)OC1, (long long)OC1, (long long)OC2, (long long)FLAT, (long long)NHIDDEN, (long long)N_BUTTONS);
    printf("total_frames=%lld minibatch_size=%lld n_minibatches/epoch=%lld nepoch=%lld lr=%g entropy_coef=%g weight_decay=%g\n",
           (long long)TOTAL, (long long)NBATCH, (long long)n_minibatches, (long long)nepoch, lr, entropy_coef, weight_decay);

    std::mt19937 shuffle_rng(42);
    std::vector<int64_t> order(TOTAL);
    std::iota(order.begin(), order.end(), 0);
    std::vector<float> mb_frames(NE_IN*NBATCH), mb_bmask(N_BUTTONS*NBATCH), mb_adv(NBATCH);

    double first_step_s = 0.0, steady_total_s = 0.0;
    int64_t total_steps = 0, steady_steps = 0;
    for (int64_t epoch = 0; epoch < nepoch; ++epoch) {
        std::shuffle(order.begin(), order.end(), shuffle_rng);
        double epoch_loss_sum = 0.0;
        for (int64_t mb = 0; mb < n_minibatches; ++mb) {
            // gather this minibatch's samples (in shuffled order) into contiguous host buffers
            for (int64_t i = 0; i < NBATCH; ++i) {
                int64_t src = order[mb*NBATCH + i];
                std::copy_n(all_frames.begin() + src*NE_IN, NE_IN, mb_frames.begin() + i*NE_IN);
                std::copy_n(all_button_mask.begin() + src*N_BUTTONS, N_BUTTONS, mb_bmask.begin() + i*N_BUTTONS);
                mb_adv[i] = all_advantage[src];
            }

            auto t0 = std::chrono::steady_clock::now();
            ggml_opt_alloc(opt_ctx, /*backward=*/true);
            ggml_backend_tensor_set(ggml_opt_inputs(opt_ctx), mb_frames.data(), 0, mb_frames.size()*sizeof(float));
            ggml_backend_tensor_set(bmask, mb_bmask.data(), 0, mb_bmask.size()*sizeof(float));
            ggml_backend_tensor_set(adv,   mb_adv.data(), 0, mb_adv.size()*sizeof(float));
            ggml_opt_result_t result = ggml_opt_result_init();
            ggml_opt_eval(opt_ctx, result);
            double loss, unc;
            ggml_opt_result_loss(result, &loss, &unc);
            auto t1 = std::chrono::steady_clock::now();
            double step_s = std::chrono::duration<double>(t1 - t0).count();
            ggml_opt_result_free(result);

            // Diagnostic instrumentation: if loss ever goes non-finite, immediately dump
            // the max |value| in every parameter tensor (and in the pre-loss `logits`
            // tensor for this minibatch) before anything gets further corrupted by
            // continued AdamW updates - this is here because two prior real runs went
            // NaN around epoch ~20 for reasons not yet fully pinned down (see AGENTS.md
            // "Step 12/13"); if it recurs, this gives hard numbers instead of another
            // guess at the mechanism.
            if (!std::isfinite(loss)) {
                auto max_abs = [&](ggml_tensor * t, const char * name) {
                    int64_t n = ggml_nelements(t);
                    std::vector<float> host(n);
                    ggml_backend_tensor_get(t, host.data(), 0, n*sizeof(float));
                    float m = 0.0f; bool any_nonfinite = false;
                    for (float v : host) { if (!std::isfinite(v)) any_nonfinite = true; else m = std::max(m, std::fabs(v)); }
                    printf("    [nan-diag] %-10s max|finite_val|=%.6g  contains_non_finite=%s\n", name, m, any_nonfinite ? "YES" : "no");
                };
                printf("  [nan-diag] non-finite loss detected at epoch %lld minibatch %lld - dumping parameter stats:\n",
                       (long long)(epoch+1), (long long)(mb+1));
                max_abs(kernel1, "kernel1"); max_abs(kernel2, "kernel2");
                max_abs(w_d1, "w_d1"); max_abs(b_d1, "b_d1");
                max_abs(w_d2, "w_d2"); max_abs(b_d2, "b_d2");
                // NOTE: deliberately NOT reading `logits` here (removed after it crashed
                // a real run with GGML_ASSERT(buf != NULL && "tensor buffer not set")) -
                // `logits` lives in ctx_compute, a dynamically-allocated graph tensor
                // whose backing buffer is only valid for the single compute call that
                // produced it ("the graph and its tensors are single-use in terms of
                // allocation" - ggml-backend.h's own documented contract), unlike the
                // ctx_static weight tensors above which persist safely. Only
                // ctx_static-resident tensors are safe to read back after the fact.
            }

            total_steps++;
            if (total_steps == 1) {
                first_step_s = step_s;
            } else {
                steady_total_s += step_s;
                steady_steps++;
            }
            epoch_loss_sum += loss;
            printf("  epoch %lld/%lld minibatch %lld/%lld: pg_loss = %.6f  (%.3fs, %.2f it/s)\n",
                   (long long)(epoch+1), (long long)nepoch, (long long)(mb+1), (long long)n_minibatches,
                   loss, step_s, 1.0/step_s);
        }
        printf(" -> epoch %lld/%lld average pg_loss = %.6f\n", (long long)(epoch+1), (long long)nepoch, epoch_loss_sum / n_minibatches);
    }
    if (steady_steps > 0) {
        printf("Timing summary: first minibatch step (incl. one-time Vulkan warm-up) = %.3fs (%.2f it/s); "
               "steady-state over remaining %lld minibatch steps = %.3fs/step avg (%.2f it/s)\n",
               first_step_s, 1.0/first_step_s, (long long)steady_steps,
               steady_total_s/steady_steps, steady_steps/steady_total_s);
    }

    // save weights back
    {
        int64_t total = ggml_nelements(kernel1)+ggml_nelements(kernel2)+ggml_nelements(w_d1)+ggml_nelements(b_d1)+ggml_nelements(w_d2)+ggml_nelements(b_d2);
        std::vector<float> host(total);
        size_t off = 0;
        auto save_t = [&](ggml_tensor * t) {
            int64_t n = ggml_nelements(t);
            ggml_backend_tensor_get(t, host.data()+off, 0, n*sizeof(float));
            off += n;
        };
        save_t(kernel1); save_t(kernel2); save_t(w_d1); save_t(b_d1); save_t(w_d2); save_t(b_d2);
        std::ofstream wf(weights_path, std::ios::binary);
        wf.write(reinterpret_cast<char*>(host.data()), total*sizeof(float));
    }
    printf("Weights saved to %s\n", weights_path.c_str());

    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend_vulkan);
    ggml_backend_free(backend_cpu);
    return 0;
}
