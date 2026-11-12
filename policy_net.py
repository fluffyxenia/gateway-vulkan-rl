"""
Pure-numpy forward pass + independent per-button Bernoulli action sampling, mirroring
EXACTLY the ggml graph built in sonic_policy_train.cpp (v3, two-conv):

    conv2d(1->OC1,3x3) -> SiLU -> avgpool2x2 -> conv2d(OC1->OC2,3x3) -> SiLU -> avgpool2x2
    -> flatten -> dense(FLAT->NHIDDEN) -> SiLU -> dense(NHIDDEN->N_BUTTONS) -> sigmoid

v3 replaces v2's single conv layer (143,758 params) with a deeper two-conv backbone
(288,658 params: conv1 1->8ch, conv2 8->16ch, dense 4480->64->N_BUTTONS) - already
validated forward+backward on Vulkan locally before being adopted here. The multi-label
independent-per-button action representation from v2 is unchanged - only the backbone
got deeper. v1 (categorical-over-bundles) and v2 (single-conv multi-label) policy_net.py
versions are both superseded; this is NOT backward compatible with either old weights
file shape.

Weight file layout (raw float32, concatenated, no header) must match the save order in
sonic_policy_train.cpp: kernel1, kernel2, w_d1, b_d1, w_d2, b_d2.

Tensor layout note (why the numpy shapes below are what they are): ggml stores tensors
with ne[0] as the fastest-varying axis. A ggml weight tensor created as
new_tensor_2d(ne0=IN, ne1=OUT) and new_tensor_4d(ne0=KW, ne1=KH, ne2=IC, ne3=OC) has
exactly the same flat memory layout as a numpy array of shape (OUT, IN) / (OC, IC, KH, KW)
respectively (both row-major, ne0/last-axis fastest) - i.e. the ggml linear-layer weight
convention and PyTorch's nn.Linear / nn.Conv2d weight conventions coincide here. This
holds regardless of how many conv layers are chained, since each layer's own kernel
tensor follows the same (OC, IC, KH, KW) <-> (ne0=KW,ne1=KH,ne2=IC,ne3=OC) correspondence
independently - verified again for v3 via the same ggml-vs-numpy forward-pass
cross-validation method used for every architecture change in this project so far.

Inference only runs here on CPU via numpy - this is intentionally NOT routed through
ggml/Vulkan. A 288K-param forward pass per action is still cheap on any CPU, and keeping
action-selection off the Vulkan path avoids ~100+ serial subprocess-with-Vulkan-init
calls per episode (slow for a fresh-subprocess-per-step design - a persistent Vulkan
inference server would remove that specific constraint, see AGENTS.md "Step 11" for why
this project is deferring that for now rather than adopting it immediately). Training
(the backward pass / weight updates, the actual point of this whole pathfinder project)
still goes through ggml's Vulkan+CPU-fallback backend_sched in sonic_policy_train.cpp.
"""
import numpy as np

IW, IH = 80, 56
IC, OC1, OC2 = 1, 8, 16
PW1, PH1 = IW // 2, IH // 2
PW2, PH2 = PW1 // 2, PH1 // 2
FLAT = PW2 * PH2 * OC2  # 20*14*16 = 4480
NHIDDEN = 64


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Step 16/17: real incident, found independently both by this agent and by the
# coordinator checking locally on sc4 with concrete numbers - conv2d_same's
# sliding_window_view is cheap (a strided view), but np.einsum(..., optimize=True)
# forces a dense materialized copy of that view for BLAS, and that copy's size scales
# with batch_size * channels * H * W * 9. Fine for the per-step rollout case (n=1
# always). Catastrophic for a large batch: the coordinator measured N=2000 through
# conv2 (8->16ch) alone hitting 1.5GB RSS; scaled linearly to this project's 10984-
# frame real-demo health-check batch, that's ~8.3GB - which is almost exactly what
# took down a 6.7GB-RAM machine (confirmed: ~7GB combined RSS+swap on the real
# incident, process in "D"/disk-sleep state, si+so swap I/O both active). This was
# NOT a gradual leak - a single oversized call, which is why it looked like a sudden
# stall (episode 20's periodic health check) rather than a slow climb.
# Fix: PolicyNet.forward() below now ALWAYS processes in bounded chunks internally,
# regardless of how large the caller's input is - no caller needs to know or manage
# this; a single frame (n=1, the normal rollout case) and an 11000-frame batch (the
# health check, or check_bc_fit_real.py) go through the exact same code path, just
# with the chunk loop running once vs many times.
FORWARD_CHUNK_SIZE = 200  # frames per chunk - within the coordinator's suggested
# 100-200 range; verified empirically (not just reasoned) to keep RSS flat on the
# real 10984-frame batch before trusting it, same as every other fix in this project.


def conv2d_same(x, kernel):
    """x: (n, in_ch, H, W). kernel: (out_ch, in_ch, 3, 3). Returns (n, out_ch, H, W),
    stride 1, 'same' zero padding - matches ggml_conv_2d(..., s0=1,s1=1,p0=1,p1=1,d0=1,d1=1).
    Caller (PolicyNet.forward) is responsible for keeping n bounded via chunking -
    this function itself does not chunk, to keep it a simple, direct mirror of the
    ggml op it matches."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # (n,in_ch,H,W,3,3)
    return np.einsum("nihwkl,oikl->nohw", windows, kernel, optimize=True)


def avgpool2x2(x):
    """x: (n, ch, H, W) with H,W even. Returns (n, ch, H/2, W/2)."""
    n, ch, H, W = x.shape
    return x.reshape(n, ch, H // 2, 2, W // 2, 2).mean(axis=(3, 5))


class PolicyNet:
    def __init__(self, n_buttons: int):
        self.n_buttons = n_buttons
        self.kernel1 = np.zeros((OC1, IC, 3, 3), dtype=np.float32)
        self.kernel2 = np.zeros((OC2, OC1, 3, 3), dtype=np.float32)
        self.w_d1 = np.zeros((NHIDDEN, FLAT), dtype=np.float32)
        self.b_d1 = np.zeros((NHIDDEN,), dtype=np.float32)
        self.w_d2 = np.zeros((n_buttons, NHIDDEN), dtype=np.float32)
        self.b_d2 = np.zeros((n_buttons,), dtype=np.float32)

    def load(self, path: str) -> bool:
        """Returns True if a weights file existed and was loaded, False if not found
        (caller should then use whatever default/random init it wants - but note
        sonic_policy_train.cpp will create+randomly-init the file itself on its first
        call, so in the normal driver flow this is only False before the very first
        training call of a fresh run)."""
        import os
        if not os.path.exists(path):
            return False
        raw = np.fromfile(path, dtype=np.float32)
        off = 0
        def take(n):
            nonlocal off
            chunk = raw[off:off + n]
            off += n
            return chunk
        self.kernel1 = take(self.kernel1.size).reshape(self.kernel1.shape).copy()
        self.kernel2 = take(self.kernel2.size).reshape(self.kernel2.shape).copy()
        self.w_d1 = take(self.w_d1.size).reshape(self.w_d1.shape).copy()
        self.b_d1 = take(self.b_d1.size).reshape(self.b_d1.shape).copy()
        self.w_d2 = take(self.w_d2.size).reshape(self.w_d2.shape).copy()
        self.b_d2 = take(self.b_d2.size).reshape(self.b_d2.shape).copy()
        assert off == raw.size, f"weight file size mismatch: consumed {off}, file has {raw.size}"
        return True

    def _forward_chunk(self, frames: np.ndarray) -> np.ndarray:
        """Same forward pass as before the Step 16/17 chunking fix - only ever called
        internally with n <= FORWARD_CHUNK_SIZE, never directly by external callers."""
        n = frames.shape[0]
        x = frames.reshape(n, IC, IH, IW)
        c1 = silu(conv2d_same(x, self.kernel1))   # (n, OC1, IH, IW)
        p1 = avgpool2x2(c1)                        # (n, OC1, PH1, PW1)
        c2 = silu(conv2d_same(p1, self.kernel2))  # (n, OC2, PH1, PW1)
        p2 = avgpool2x2(c2)                        # (n, OC2, PH2, PW2)
        flat = p2.reshape(n, FLAT)
        h1 = silu(flat @ self.w_d1.T + self.b_d1)  # (n, NHIDDEN)
        logits = h1 @ self.w_d2.T + self.b_d2      # (n, n_buttons)
        return sigmoid(logits)

    def forward(self, frames: np.ndarray) -> np.ndarray:
        """frames: (N, IH, IW) float32 in [0,1]. Returns independent per-button
        probabilities (N, n_buttons) - NOT a distribution summing to 1; each entry is
        its own Bernoulli parameter.

        Step 16/17 fix: processes in FORWARD_CHUNK_SIZE-row chunks internally,
        regardless of N - see the comment above FORWARD_CHUNK_SIZE for the real
        incident this fixes (conv2d_same's einsum materializing a dense copy whose
        size scales with batch size, causing a catastrophic ~7GB spike on a 6.7GB-RAM
        machine the one time this was called with all 10984 real demo frames at
        once). Mathematically identical output to the old single-shot version for any
        N - chunking the batch dimension doesn't change any per-sample computation,
        since every layer here (conv/pool/dense) is applied independently per row."""
        n = frames.shape[0]
        if n <= FORWARD_CHUNK_SIZE:
            return self._forward_chunk(frames)
        outputs = []
        for start in range(0, n, FORWARD_CHUNK_SIZE):
            outputs.append(self._forward_chunk(frames[start:start + FORWARD_CHUNK_SIZE]))
        return np.concatenate(outputs, axis=0)

    def sample_buttons(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """frame: (IH, IW). Returns a (n_buttons,) 0/1 int array - each button sampled
        independently as Bernoulli(p_k). The empty set (all zeros = no-op) and every
        combo (including ones never explicitly enumerated, e.g. DOWN+C) are all directly
        reachable - there is no fixed menu restricting which subsets can come out."""
        probs = self.forward(frame[None, :, :])[0]
        return (rng.random(self.n_buttons) < probs).astype(np.int8)
