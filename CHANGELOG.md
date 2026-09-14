# History

Two sessions built this project. The first (Codex, 13 to 14 September 2026, about 14 hours) is summarised from its log; the second (Claude Code, 14 September 2026) continued from its last checkpoint. Everything below is recorded so nobody repeats a dead end.

## Session 1: Codex, 13 to 14 September 2026

1. **Teacher choice.** Replaced an earlier Qwen/DistilGPT2/vision/finance pipeline with Gemma 3 270M IT as the only teacher, via response distillation (Gemma writes an explanation; the dataset answer stays the anchor). Logit distillation was ruled out: Gemma's 262k vocabulary is incompatible with a 10M student.
2. **Size.** Target cut from 80M to 10M parameters. First layout: GPT-2 tokenizer, 160-wide model, four decoder layers.
3. **Environment.** The project had installed CPU-only torch on a machine with an RTX 5070 Ti; repinned to CUDA 13.0 wheels. Gemma is a gated repo and needed `hf auth login`.
4. **First SFT collapsed** ("100 100 100...", then "Final answer: Neutral" to everything). Causes: tied embeddings initialised at scale 1.0, and 1,350 short task labels as the only training data.
5. **Stage split** into `pretrain`, `anneal`, `sft`/`posttrain`, `evolve`, with a foundation quality gate (held-out loss at or below 4.0 and at least 75% of greedy rollouts free of collapse). Evolution re-added as CEM over a small latent-thought offset.
6. **WikiText-103 foundation, 9 rounds.** Loss 10.86 to 4.06; greedy probes stayed at 0 to 2 healthy of 16.
7. **FineWeb-Edu with replay, 7 slices; Wiki40B and textbook mixtures, 9 slices.** Loss kept falling, generation stayed degenerate. An "immutable anchor" of fixed examples was stored in checkpoints to detect forgetting.
8. **Architecture work.** 8M of the 10M parameters were the GPT-2 embedding table. Factorised embeddings were tried and rejected. Switched to a fresh 8,192-token byte-level BPE trained on the corpus (`compact-10m`: 256-wide, six decoder layers), which discarded all earlier checkpoints.
9. **Compact-model foundation, six runs** on educational web plus Wikipedia plus textbooks; best 59% healthy probes. Scheduled sampling (rejected), Simple Wikipedia mix (rejected), prompt-tail decoder start, sequence-unlikelihood penalty (repetition 57% to 30%, still unreadable), Gemma-rewritten prose for pretraining (15 of 32 rewrites changed facts; rejected).
10. **TinyStories.** Final pivot: fresh tokenizer, 24-token decoder prompt prefill, 128-token input, 160-token targets. Three passes (55M + 86M + 86M tokens). Best checkpoint `long-02`: 20/32 healthy at 64 tokens, 6/32 at 128, greedy. The session ended with the user stopping a third pass.

Cost of the session: about 250 file-change batches, 82 test-suite runs (5 to 133 tests), 45 training runs, no usable model.

## Session 2: Claude Code, 14 September 2026

### Foundation

- Reran the interrupted third TinyStories pass. Held-out loss 1.634 to 1.598, story panel unchanged, so the trainer's rollout guard rolled the checkpoint back to the long-02 weights. Conclusion: the recipe had plateaued.
- Added sampling to generation (temperature, top-p, repetition penalty, repeated-n-gram block) in a 21-line helper. `play.py` samples by default; `story_eval` takes the same flags and writes sampled results separately. Sampling took long-02 from 20/32 to 29/32 healthy at 64 tokens and 6/32 to 18/32 at 128.

### Dead code

- Removed the rejected experiments: Gemma prose module, sequence unlikelihood, self-conditioning, replay flags, factorised and legacy architectures, prompt-tail start, the single-dataset CLI, FinQA, the `post_train` alias. Source went from about 5,300 to 3,960 lines, tests from about 2,700 to 1,600. `load_checkpoint` filters unknown config keys so the live checkpoint still loads.

### SFT with Gemma as teacher

- New sources: `tinystories_instruct` (Gemma writes a story for a TinyStoriesInstruct prompt; a rule verifier accepts it only if every required word is used, dialogue appears when asked, and it has at least 40 words; otherwise the dataset story is the target), `tinystories_continue` (teacher-free continuation replay), `tinystories_instruct_valid` (held-out prompts for evolution). Gemma passes the verifier about 15 to 17% of the time.
- Four attempts before one was accepted by the foundation-retention guard. Causes, in order of size: the SFT trainer cycled encoder effort low/medium/high per step while the base was trained at six loops (+0.39 anchor drift after one epoch); a flat 1e-4 learning rate on an annealed base (+0.32 even with replay); replay targets tokenised without the leading space of pretraining continuations. Fixes: `--thinking-effort high`, `--lr 3e-5` then `2e-5` with warmup and cosine decay, replay with the leading space.
- `story-sft-04` (2.4k prompts): verifier reward 0.21 to 0.42 over the base on held-out prompts.
- `story-sft-05` (7.2k prompts): SFT validation 1.798 to 1.764, reward unchanged. Word compliance stays near one required word in three: capacity-bound.
- Parser fix: when the Story field preceded other fields in the source file, "Summary:" lines leaked into targets (about 4% of rows).

### Evolution

- Batched the evaluator (padded batches of 32; outputs identical to single-prompt decoding) so it can afford more prompts.
- `story-evolved-01` (from sft-04, 32 tuning prompts): tuning reward up, held-out down (0.440 to 0.351). Overfit.
- `story-evolved-03` (from sft-05, 64 tuning prompts, population 10, six generations): held-out 0.387 to 0.410; on the 128-prompt panel greedy reward 0.410 to 0.419 and all-three-words completions 2 to 6. The first evolution run with a real, if small, gain.

### Variety

- Greedy decoding picked "Lily" for about 60% of stories on every SFT checkpoint. Capping (`story-sft-05`) or renaming (`story-sft-06`) the protagonist to 5% of training rows did not change that: the pretrained base carries the prior and greedy picks the mode. Renaming cost a little reward.
- Sampling at T=1.0 with top-p 0.9 and a repeated-4-gram block cut the Lily share to about 30%, doubled distinct openings, and cost no reward on sft-04/05. `play.py` defaults to T=1.0.

### Operations

- Three concurrent CUDA jobs exhausted the 16 GB card and the driver killed all of them, losing a 128-prompt evolution run. GPU work is now sequential.
- Cleanup for the first push: 15 MB session log, two legacy root checkpoints, four unused mixture configs, and 80 dead checkpoints (9.1 GB to 540 MB) deleted; `*.pt` ignored.

### KV cache and TurboQuant

- Decoder KV cache (per-layer self-attention keys/values, encoder cross-attention projections computed once). Exact: 32/32 greedy sequences identical to the uncached path; batch-32 generation 30x faster. On by default everywhere generation happens.
- TurboQuant-style compression of the self-attention cache (`src/quant.py`): random rotation, Lloyd-Max codebooks at 1 to 4 bits, bit-packed codes, fp16 norms, 1-bit QJL residual for unbiased attention scores. 4-bit: 312 bytes per token per layer against 1024 for fp16, 94% next-token agreement, KL 0.011. `src/kv_bench.py` reproduces the table.

### Adaptive depth

- Measured reward vs loop count: the fixed-depth SFT model peaks at 3 loops (0.495), not the 6 it trained at (0.410), and collapses at 8+. Its encoder state never converges, so convergence halting runs to the cap and hurts.
- Added `--loop-range LO HI` (random loop count per training step) and per-row convergence halting (`halt_threshold`, `/effort auto`). A foundation pass with loops in 1..8 gives identical held-out loss at every depth (1.500, vs 1.538 best for the fixed model) and a fixed point after one loop. SFT on it (`story-sft-depth-01`) is flat from 1 to 8 loops (0.44 to 0.46) and stops itself at 2 loops at no cost. Conclusion: robustness and self-stopping, not more intelligence, on this task.
- `play.py` defaults to `medium` (3 loops).

### Rejection-sampling fine-tuning (expert iteration)

- `src/rft.py` samples N stories per training prompt and keeps the verifier-passing best (every required word, dialogue when asked, EOS, <= 8% repeated 4-grams); `load_examples` accepts `jsonl:<path>` sources; `sft` accepts post-training checkpoints so rounds can iterate. `play.py --best-of N` picks the verifier's favourite at inference.
- Rounds from `story-sft-depth-01` (reward 0.459 at 4 loops, 4/128 all-three-words): round 1 kept 861 samples -> 0.511 / 10; round 2 kept 1,263 -> 0.580 / 13; round 3 kept 1,759 -> 0.605 / 21 (0.600 with self-stopping at 2 loops). Self-stopping (2 loops) tracks the same gains. The largest improvement of the project.

### Documentation

- README: quick start, repository layout, every stage's command, the foundation panel, the 128-prompt reward and variety table, the KV cache and TurboQuant tables, and the tried-and-rejected list. This file records the history. `src/instruct_eval.py` and `src/kv_bench.py` reproduce every table.

### Final checkpoints kept on disk (not in git)

| File | Role |
| --- | --- |
| `loop-transformer-10m-tinystories-prefill-long-02.pt` | foundation base |
| `loop-transformer-10m-story-sft-05.pt` | SFT, the checkpoint to build on |
| `loop-transformer-10m-story-evolved-03.pt` | SFT plus evolved latent offset, the one to play with (use `/effort medium`) |
| `loop-transformer-10m-tinystories-depth-01.pt`, `loop-transformer-10m-story-sft-depth-01.pt` | variable-depth base and its SFT; the ones that can stop themselves (`/effort auto`) |
| `loop-transformer-10m-story-sft-depth-rft1/2/3.pt` | rejection-sampling rounds on the variable-depth line; **rft3 is the final model** (reward 0.605, self-stopping at 2 loops) |
| `prefill-01`, `long-01`, `story-sft-04`, `story-sft-06`, `story-evolved-01` | comparison points in the README tables |
