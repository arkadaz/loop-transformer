# LoopTransformer

A **10M-parameter** text-only recurrent encoder-decoder trained from random weights. `google/gemma-3-270m-it` is the project's only teacher; it is frozen and used only to write SFT targets after the student has a healthy base checkpoint.

The stages follow the usual sequence at toy scale: foundation -> quality annealing -> SFT -> verifier-grounded optimization. Every stage has been run on the TinyStories base. The foundation gate itself never passed, so SFT used `--allow-ungated`; read the results below before trusting any checkpoint. [CHANGELOG.md](CHANGELOG.md) records the full history, including what was tried and rejected.

## Quick start

```powershell
uv sync
uv run python play.py --checkpoint checkpoints/loop-transformer-10m-story-qa-02.pt --max-new-tokens 160 --best-of 4
```

Then type a prompt in the trained format, for example `Write a short story for young children. Use the words: dog, ball, happy. Include dialogue.` Optional additions: `The story is about: <one line>` and `Include this sentence: <sentence>`. `/temp 0.7` gives more coherent, less varied stories; `/temp 0` is greedy. Add `--kv-bits 3` to run with a compressed KV cache, or `--no-kv-cache` to recompute each step. `/effort auto` is meant for checkpoints trained with `--loop-range` (the `story-sft-depth-*` ones). `--best-of N` samples N stories and shows the one the verifier scores highest. Checkpoints are not in git; the commands below rebuild them in about two hours on an RTX 5070 Ti.

## Repository layout

| Path | Role |
| --- | --- |
| `main.py` | Dispatches `pretrain`, `anneal`, `sft`/`posttrain`, `evolve`. |
| `play.py` | Interactive runner: sampling controls, effort level, KV-cache options. |
| `src/model.py` | The loop transformer: recurrent encoder, latent thoughts, decoder with prompt prefill, KV cache, sampling helper. |
| `src/pretrain.py` | Foundation trainer: streamed mixture loader, windowing, quality gate, anchor, rollout-guarded selection. |
| `src/pretrain_distill.py` | Shared training loop, checkpoint save/load, the SFT stage with its foundation-retention guard. |
| `src/data.py` | SFT data sources (TinyStoriesInstruct story prompts, continuation replay, GSM8K/ARC/CommonsenseQA/finance/Dolly) and the story verifier. |
| `src/gemma_teacher.py` | Gemma 3 270M IT access, batched generation, append-only target cache, story mode. |
| `src/evolution.py` | CEM over the latent-thought offset with the answer reward and the rule-verified story reward. |
| `src/rft.py` | Rejection-sampling fine-tuning data: N samples per prompt, keep the verifier-passing best; feeds `sft` via `--datasets jsonl:<path>`. |
| `src/qa_build.py` | Story-grounded Q&A data: Gemma writes a question and answer per passage, a verifier keeps the grounded ones. |
| `src/quant.py` | TurboQuant KV-cache compression: random rotation, Lloyd-Max codebooks, 1-bit QJL residual, bit-packed storage. |
| `src/kv_bench.py` | KV-cache report: exactness, speed-up, fidelity per bit width, bytes per token. |
| `src/story_eval.py` | Frozen 32-story continuation report for foundation checkpoints (`--kv-bits` supported). |
| `src/instruct_eval.py` | Verifier reward and variety report for SFT/evolution checkpoints (`--task story|qa`, `--kv-bits`; produces the tables below). |
| `src/student_tokenizer.py` | 8k byte-level BPE trainer, embedded in checkpoints. |
| `configs/tinystories_foundation.json` | The pinned foundation corpus. |
| `tests/` | 101 tests; `uv run python -m pytest -q`. |

## What the model is

| Part | Current implementation (`--architecture compact-10m`) |
| --- | --- |
| Tokenizer | 8,192-token byte-level BPE trained on the foundation corpus and embedded in the checkpoint (`--tokenizer compact-bpe`). Gemma's 262k vocabulary would alone exceed the parameter budget; GPT-2's 50k vocabulary spent 80% of the weights on the embedding table. |
| Encoder | Input tokens plus four learned latent-thought tokens; one 256-wide block reused at every loop. |
| Decoder | Six causal blocks that cross-attend to the final encoder state. The last 24 prompt tokens are also copied into the decoder (`--decoder-prompt-prefill-tokens 24`) so the first generated word does not depend on cross-attention alone. |
| Parameters | 10,015,296, tied embedding and LM head. |
| Context | Architecture limit 512 tokens. Trained on 128 input + 160 target tokens; `play.py` truncates longer prompts to the trained limit. |
| Generation | Sampling (temperature, top-p, repeated-n-gram block) and an exact per-layer KV cache; `--kv-bits 1..4` compresses the cache with TurboQuant. See [KV cache and TurboQuant](#kv-cache-and-turboquant). |

`/effort low`, `medium`, and `high` run 1, 3, and 6 encoder loops; `/effort auto` lets each prompt stop looping once its encoder state stops changing (see [Adaptive depth](#adaptive-depth)). Generation stops at EOS or the token limit.

## Stage 1: foundation on TinyStories

The corpus is `roneneldan/TinyStories` (pinned revision in `configs/tinystories_foundation.json`). Every window is raw human text: no replay, no Gemma prose. Three passes have been run, about 230M supervised tokens in total. Each command streams a disjoint slice of the training split and saves a new checkpoint; the previous checkpoint is never overwritten.

```powershell
uv sync
# 1. fresh tokenizer + short probe (a few minutes)
uv run python main.py pretrain --mixture-config configs/tinystories_foundation.json --streaming --max-documents 15000 --max-validation-documents 2000 --max-examples 25000 --max-validation-examples 1000 --architecture compact-10m --tokenizer compact-bpe --tokenizer-vocab-size 8192 --tokenizer-max-documents 8000 --decoder-prompt-prefill-tokens 24 --max-input-tokens 128 --max-target-tokens 64 --batch-size 64 --epochs 1 --lr 3e-4 --warmup-steps 100 --min-lr-ratio 0.1 --prefix-loss-weight 1 --prefix-loss-tokens 0 --foundation-eval-examples 64 --rollout-tokens 64 --selection-strategy validation --save-every-steps 100 --seed 101 --device cuda --output checkpoints/loop-transformer-10m-tinystories-prefill-probe.pt

# 2. 55M tokens, 64-token targets (~12 min on an RTX 5070 Ti)
uv run python main.py pretrain --init-checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-probe.pt --output checkpoints/loop-transformer-10m-tinystories-prefill-01.pt --mixture-config configs/tinystories_foundation.json --streaming --max-documents 300000 --max-validation-documents 10000 --max-examples 1000000 --max-validation-examples 5000 --decoder-prompt-prefill-tokens 24 --max-input-tokens 128 --max-target-tokens 64 --batch-size 64 --epochs 1 --lr 3e-4 --warmup-steps 500 --min-lr-ratio 0.1 --prefix-loss-weight 1 --prefix-loss-tokens 0 --foundation-eval-examples 64 --rollout-tokens 64 --selection-strategy validation --save-every-steps 1000 --seed 103 --device cuda

# 3. 86M tokens per pass, 160-token targets so 128-token generation is trained (~25 min each).
#    Repeat with --streaming-train-skip-documents 800000, 1600000, ... to move to unseen stories.
uv run python main.py pretrain --init-checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-01.pt --output checkpoints/loop-transformer-10m-tinystories-prefill-long-01.pt --mixture-config configs/tinystories_foundation.json --streaming --max-documents 800000 --max-validation-documents 10000 --max-examples 750000 --max-validation-examples 5000 --decoder-prompt-prefill-tokens 24 --max-input-tokens 128 --max-target-tokens 160 --train-initial-prefix-span 32 --batch-size 32 --epochs 1 --lr 2e-4 --warmup-steps 500 --min-lr-ratio 0.1 --prefix-loss-weight 1 --prefix-loss-tokens 0 --foundation-eval-examples 64 --rollout-tokens 128 --selection-strategy validation --save-every-steps 2000 --seed 109 --device cuda
```

The runner keeps the weights with the best held-out loss, stores a fixed 64-example anchor inside the checkpoint, and applies a gate: held-out loss at or below 4.0 **and** at least 75% of greedy 128-token rollouts free of collapse (short output, runs of one token, repeated trigrams). The gate is a collapse screen, not a proof of quality. No checkpoint has passed the 75% rollout gate yet.

### Evaluate and play

```powershell
# frozen 32-story panel plus 12 contrast prompts, greedy (the gate's decoding)
uv run python -m src.story_eval --checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-long-02.pt --device cuda
# same panel with sampling; written to a separate *.story-eval.sampled.json
uv run python -m src.story_eval --checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-long-02.pt --device cuda --temperature 0.7 --top-p 0.9 --no-repeat-ngram 4
# interactive; sampling at T=1.0 is the default, /temp 0 switches to greedy
uv run python play.py --checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-long-02.pt --max-new-tokens 128
```

Healthy completions out of 32 on the frozen validation panel:

| Checkpoint | Greedy 64t | Greedy 128t | Sampled 64t | Sampled 128t |
| --- | --- | --- | --- | --- |
| `tinystories-prefill-01` (55M tokens) | 13 | 1 | - | - |
| `tinystories-prefill-long-01` (+86M) | 24 | 4 | - | - |
| `tinystories-prefill-long-02` (+86M) | 20 | 6 | 29 | 18 |
| `tinystories-prefill-long-03.partial` (+86M, see note) | 20 | 5 | 28 | 18 |

Note: the third pass improved held-out loss (1.634 to 1.598) but the trainer's greedy rollout guard rejected the trained weights and rolled back to long-02; the last row measured the discarded weights (since deleted). Another 86M tokens of the same recipe changed nothing on the panel: this recipe has plateaued.

Sampled here = temperature 0.7, top-p 0.9, repeated 4-gram block, seed 0. Greedy decoding of a 10M model loops; sampling removes most of that. What sampling does not fix is situation drift: a prompt about finding a toy can turn into a story about a dog stealing it. That is a capacity and data limit, not a decoding bug.

### Tried and rejected (do not repeat)

- GPT-2 vocabulary on WikiText-103, FineWeb-Edu, Wiki40B and textbook mixtures with replay: loss fell, generation stayed degenerate. 80% of the weights were the embedding table.
- Factorized (low-rank) GPT-2 embeddings: beat the plain layout in a matched control but still unusable.
- Scheduled sampling / decoder self-conditioning: lower loss, worse free generation.
- Sequence-unlikelihood penalty on the model's own repeated 4-grams: cut repetition from 57% to 30% but did not produce readable text.
- Gemma-rewritten "simple prose" as pretraining data: 15 of 32 inspected rewrites changed facts. Rejected.

Their code has been deleted; the list stays so nobody repeats them.

## Stages 2 to 4

| Stage | Command | What ran |
| --- | --- | --- |
| Annealing | `anneal` | Implemented (lower learning rate, refuses ungated checkpoints). Skipped: TinyStories is already the clean corpus and the gate never passed. |
| SFT / response distillation | `sft` (alias `posttrain`) | Ran. For each TinyStoriesInstruct prompt (three required words, optional dialogue, optional one-line plot, optional required sentence) Gemma writes a children's story. A rule verifier accepts it only if every required word is used, dialogue appears when asked, and it has at least 40 words; otherwise the dataset's own story is the target. About 15% of Gemma's stories pass. Roughly 60% of each batch is teacher-free story-continuation replay (`tinystories_continue`) so the foundation anchor is retained. Training rows rename the protagonist when one name would exceed 5% of rows (TinyStories is 21% Lily). |
| Evolution | `evolve` | Ran. CEM search over the 1,024-value latent-thought offset (4 thoughts x 256) with `reward = share of required words used + 0.1 dialogue-ok + 0.05 EOS - repetition penalty`, greedy decoding, on prompts from the Instruct validation file. |

```powershell
uv run hf auth login   # once; accept Gemma's terms first
uv run python main.py sft --init-checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-long-02.pt --output checkpoints/loop-transformer-10m-story-sft-05.pt --datasets tinystories_instruct,tinystories_continue --per-source 12000 --validation-fraction 0.05 --epochs 3 --lr 2e-5 --warmup-steps 100 --min-lr-ratio 0.1 --batch-size 16 --max-input-tokens 128 --max-target-tokens 256 --teacher-tokens 220 --teacher-temperature 0.7 --teacher-batch-size 16 --allow-ungated --max-foundation-regression 0.25 --thinking-effort high --seed 42 --device cuda
uv run python main.py evolve --init-checkpoint checkpoints/loop-transformer-10m-story-sft-05.pt --output checkpoints/loop-transformer-10m-story-evolved-03.pt --datasets tinystories_instruct_valid --per-source 800 --validation-fraction 0.5 --eval-examples 64 --population 10 --elites 3 --generations 6 --sigma 0.05 --max-input-tokens 128 --max-new-tokens 160 --thinking-effort high --seed 7 --device cuda
uv run python play.py --checkpoint checkpoints/loop-transformer-10m-story-sft-05.pt --max-new-tokens 160
```

Prompt format the SFT model expects: `Write a short story for young children. Use the words: dog, ball, happy. Include dialogue.` (optionally `The story is about: ...` and `Include this sentence: ...`). Gemma targets are cached in `.cache/gemma_targets.jsonl`, so reruns only train.

```powershell
# the table below
uv run python -m src.instruct_eval checkpoints/loop-transformer-10m-tinystories-prefill-long-02.pt checkpoints/loop-transformer-10m-story-sft-05.pt checkpoints/loop-transformer-10m-story-evolved-03.pt --prompts 128 --temperatures 0.7,1.0
```

Verifier reward on 128 held-out Instruct prompts (max about 1.15). `all 3` = completions using every required word; `openings` = distinct first four words across the 128 completions; `Lily` = completions that mention Lily. Sampled rows use top-p 0.9 and a repeated-4-gram block.

| Checkpoint | Decoding | Reward | Words used | All 3 | Openings | Lily |
| --- | --- | --- | --- | --- | --- | --- |
| base `long-02` | greedy | 0.214 | 0.17 | 0 | 75 | 21 |
| base `long-02` | T=1.0 | 0.239 | 0.16 | 0 | 115 | 18 |
| `story-sft-04` (2.4k prompts) | greedy | 0.422 | 0.33 | 3 | 77 | 75 |
| `story-sft-04` | T=0.7 | 0.435 | 0.32 | 2 | 86 | 63 |
| `story-sft-04` | T=1.0 | 0.433 | 0.32 | 5 | 106 | 42 |
| `story-sft-05` (7.2k prompts, name cap) | greedy | 0.410 | 0.31 | 2 | 51 | 77 |
| `story-sft-05` | T=0.7 | 0.443 | 0.33 | 2 | 70 | 52 |
| `story-sft-05` | T=1.0 | 0.435 | 0.32 | 3 | 101 | 35 |
| `story-sft-06` (7.2k prompts, renaming) | greedy | 0.409 | 0.31 | 2 | 62 | 78 |
| `story-sft-06` | T=1.0 | 0.388 | 0.27 | 0 | 110 | 40 |
| `story-evolved-01` (from sft-04, 32 prompts) | greedy | 0.404 | 0.33 | 6 | 67 | 84 |
| `story-evolved-03` (from sft-05, 64 prompts) | greedy | 0.419 | 0.33 | 6 | 51 | 79 |
| `story-evolved-03` | T=1.0 | 0.440 | 0.32 | 1 | 108 | 46 |

What the table says:

- **SFT is the step that works.** Reward roughly doubles over the base and the model follows the prompt format. It still uses all three required words only a few times in 128 and drifts off plot: a 10M model copies one or two constraint words, not three.
- **More data did not raise the reward.** Three times more prompts improved SFT validation loss (1.798 -> 1.764) but left the verifier reward flat. Word compliance is capacity-bound, not data-bound, at this size.
- **Variety comes from decoding, not from data.** Greedy decoding picks the modal name every time: about 60% Lily for every SFT checkpoint even after the training data was capped or renamed to 5% Lily, because the pretrained base carries the prior. Sampling at T=1.0 cuts Lily to roughly 30%, doubles distinct openings, and costs no reward on sft-04/05. `play.py` therefore defaults to T=1.0. Renaming (sft-06) added nothing beyond that and cost a little reward.
- **Evolution needs enough prompts.** With 32 tuning prompts it raised tuning reward and lowered held-out reward (0.440 -> 0.351). With 64 tuning prompts and the batched evaluator it transferred: held-out 0.387 -> 0.410 during the run, and on the 128-prompt panel greedy reward 0.410 -> 0.419 with all-three-words completions 2 -> 6. A small, real gain; more prompts would help further.

What made SFT work, after three rejected attempts:

- Train at the foundation's loop count (`--thinking-effort high`). The old default cycled low/medium/high per step and dragged the shared block away from its 6-loop behaviour (+0.39 anchor regression after one epoch).
- A fine-tuning learning rate with warmup and cosine decay (3e-5 for 2.4k prompts, 2e-5 for 7k). At a flat 1e-4 the anchor regressed +0.32 even with replay.
- Replay: about 60% of the examples are plain prefix -> continuation pairs, with the leading space that pretraining continuations carry.
- Three epochs. Validation loss bottomed at epoch 2 to 3, then the model memorized. With 7k prompts only epoch 1 stayed inside the anchor limit; the saved sft-05/06 weights are epoch 1.
- Run one GPU job at a time. Three concurrent CUDA jobs exhausted the 16 GB card and the driver killed all of them.

## KV cache and TurboQuant

Generation keeps a decoder KV cache: each layer stores its self-attention keys and values and computes the encoder cross-attention projections once, so a step processes only the new token. The cache reproduces the uncached outputs exactly (32/32 greedy sequences identical on real prompts) and is on by default in `generate`, `play.py`, the evaluators and the evolution stage.

`--kv-bits {1,2,3,4}` compresses the self-attention cache with a TurboQuant-style scheme (`src/quant.py`), data-oblivious and calibration-free:

1. Normalise each key or value vector, rotate it by a fixed random orthogonal matrix so its coordinates are near-Gaussian, and round every coordinate to the MSE-optimal Lloyd-Max codebook for N(0, 1) at the chosen bit width. Codes are bit-packed; the vector norm is kept in fp16.
2. For keys, sketch the rounding residual with a random Gaussian projection and keep one sign bit per coordinate (Quantized Johnson-Lindenstrauss). The attention score is then `q . k_hat + sqrt(pi/2)/d * |r| * <S q, sign(S r)>`, an unbiased estimate of `q . k`. The test suite checks that plain dequantisation understates correlated scores by more than 5% while the QJL-corrected score is unbiased within 2%.
3. Values use step 1 only.

Measured on `story-sft-05`, 32 held-out prompts, 160 new tokens, RTX 5070 Ti (`uv run python -m src.kv_bench`):

| Cache | Bits K / V per coordinate | Bytes per token per layer | Mean abs. logit change | Top-1 agreement | KL(exact, quantised) |
| --- | --- | --- | --- | --- | --- |
| exact fp32 (fp16 equivalent) | 32 (16) | 2048 (1024) | 0 | 100% | 0 |
| TurboQuant 4-bit | 5.5 / 4.25 | 312 | 0.16 | 94.0% | 0.011 |
| TurboQuant 3-bit | 4.5 / 3.25 | 248 | 0.31 | 86.6% | 0.046 |
| TurboQuant 2-bit | 3.5 / 2.25 | 184 | 0.62 | 75.9% | 0.214 |
| TurboQuant 1-bit | 2.5 / 1.25 | 120 | 1.17 | 57.5% | 0.789 |

Speed: batch 32 x 160 tokens went from 19.2 s to 0.6 s (30x); batch 1 from 0.7 s to 0.6 s, because at batch 1 the per-step launch overhead dominates a 10M model. Bit widths are effective values including the fp16 norms and the QJL sign bit (head dimension 64). Mean |logit| is 3.6 for scale.

Verifier reward of `story-evolved-03` on the 128-prompt panel (greedy) with the compressed cache:

| Cache | Reward | All 3 words | Openings | Lily |
| --- | --- | --- | --- | --- |
| exact | 0.419 | 6 | 51 | 79 |
| TurboQuant 4-bit | 0.408 | 3 | 54 | 77 |
| TurboQuant 3-bit | 0.408 | 3 | 52 | 80 |
| TurboQuant 2-bit | 0.391 | 2 | 55 | 79 |

Honest framing: this model's whole cache is a few hundred kilobytes, so compression buys nothing in practice here. The implementation is the point: it is the mechanism a long-context version would need, measured end to end.

What this does not do: the model still uses learned absolute positions up to 512 tokens and materialised attention. Long context needs RoPE plus a context curriculum and retraining; a KV cache does not extend the trained length.

## Adaptive depth

A loop transformer should decide how many loops a prompt needs. Two things were measured first on `story-sft-05`, which was trained at a fixed 6 loops: the encoder state keeps changing at every loop (relative change 0.50, 0.20, 0.19, 0.17 ... 0.05 at loop 12), and quality is a sharp function of the loop count with its peak *below* the trained depth. So a convergence rule could not stop it, and the fixed depth was not even its best operating point.

`--loop-range LO HI` (on `pretrain` and `sft`) draws a random loop count per training step, which forces depth-consistent representations. `halt_threshold` in `encode`/`generate` (`/effort auto` in `play.py`, `--halt-thresholds` in `instruct_eval`) then stops each prompt row once its mean relative state change drops below the threshold, up to a cap; other rows keep looping. No new parameters.

```powershell
# foundation pass with random depth 1..8, from long-02 (the rollout guard at 6 loops rejected the trained weights; they are in the *.partial.pt, promoted here to depth-01.pt)
uv run python main.py pretrain --init-checkpoint checkpoints/loop-transformer-10m-tinystories-prefill-long-02.pt --output checkpoints/loop-transformer-10m-tinystories-depth-01.pt --mixture-config configs/tinystories_foundation.json --streaming --streaming-train-skip-documents 400000 --max-documents 800000 --max-validation-documents 10000 --max-examples 750000 --max-validation-examples 5000 --decoder-prompt-prefill-tokens 24 --max-input-tokens 128 --max-target-tokens 160 --train-initial-prefix-span 32 --batch-size 32 --epochs 1 --lr 1.5e-4 --warmup-steps 500 --min-lr-ratio 0.1 --prefix-loss-weight 1 --prefix-loss-tokens 0 --foundation-eval-examples 64 --rollout-tokens 128 --selection-strategy validation --save-every-steps 2000 --loop-range 1 8 --seed 131 --device cuda
# SFT with the same range
uv run python main.py sft --init-checkpoint checkpoints/loop-transformer-10m-tinystories-depth-01.pt --output checkpoints/loop-transformer-10m-story-sft-depth-01.pt --datasets tinystories_instruct,tinystories_continue --per-source 12000 --validation-fraction 0.05 --epochs 3 --lr 2e-5 --warmup-steps 100 --min-lr-ratio 0.1 --batch-size 16 --max-input-tokens 128 --max-target-tokens 256 --teacher-tokens 220 --teacher-temperature 0.7 --teacher-batch-size 16 --allow-ungated --max-foundation-regression 0.25 --thinking-effort high --loop-range 1 8 --seed 42 --device cuda
# the table below
uv run python -m src.instruct_eval checkpoints/loop-transformer-10m-story-sft-05.pt checkpoints/loop-transformer-10m-story-sft-depth-01.pt --prompts 128 --loops 1,2,3,4,6,8,12 --halt-thresholds 0.2,0.1,0.05 --max-loops 12
```

Foundation held-out continuation loss (256 validation stories) by loop count: `long-02` (fixed 6) 1.746, 1.597, 1.563, 1.548, **1.538**, 1.541, 1.554 at k = 1, 2, 3, 4, 6, 8, 12; `depth-01` (random 1..8) **1.500 at every k**. Its state change after loop 1 is 0.02 to 0.05: freed from a fixed depth, it reaches a fixed point after one loop, because TinyStories continuation needs no iterative computation.

Verifier reward on the 128 instruction prompts, greedy:

| Setting | `story-sft-05` (fixed 6) | `story-sft-depth-01` (random 1..8) |
| --- | --- | --- |
| 1 loop | 0.401 | 0.439 |
| 2 loops | 0.435 | 0.447 |
| 3 loops | **0.495** | 0.442 |
| 4 loops | 0.479 | 0.459 |
| 6 loops (old default) | 0.410 | **0.460** |
| 8 loops | 0.321 | 0.457 |
| 12 loops | 0.299 | 0.409 |
| auto, halt < 0.2 | 0.341 (7.2 loops used) | 0.447 (2.0 loops used) |
| auto, halt < 0.1 | 0.304 (11.6 loops) | 0.438 (2.0 loops) |
| auto, halt < 0.05 | 0.299 (12 loops) | 0.454 (7.0 loops) |

What it means:

- **Variable-depth training buys robustness and self-stopping, not more intelligence.** The depth-trained model scores the same from 1 to 8 loops and stops itself after 2 loops at no cost (0.447 vs 0.460), a 3x saving in encoder compute. It does not get better with more loops: this task has nothing for the recurrence to iterate on.
- **The fixed-depth model was mis-operated.** Its best operating point is 3 loops (0.495), not the 6 it trained at (0.410). `play.py` now defaults to `medium`. Any halting rule is useless on it because its state never converges.
- **Beyond the trained range, both degrade** (12 loops), so the halting cap should stay inside it.
- A learned halting head (PonderNet or ACT with a ponder cost) is only worth adding for a task where more loops demonstrably help; on this data they do not.

## Rejection-sampling fine-tuning: the step that made it smarter

Evolution only moves a 1,024-value offset. To move the whole model toward the verifier, train it on its own verified successes (expert iteration): sample 8 stories per training prompt at T=1.0, keep a sample only if it uses every required word, has dialogue when asked, ends with EOS and repeats no more than 8% of its 4-grams, then fine-tune on those stories (weighted 2 to 3x) together with the dataset stories and continuation replay, and repeat from the improved model.

```powershell
uv run python -m src.rft --checkpoint checkpoints/loop-transformer-10m-story-sft-depth-01.pt --output data/rft_round1.jsonl --samples 8 --loops 4 --temperature 1.0 --top-p 0.9 --seed 42
uv run python main.py sft --init-checkpoint checkpoints/loop-transformer-10m-story-sft-depth-01.pt --output checkpoints/loop-transformer-10m-story-sft-depth-rft1.pt --datasets jsonl:data/rft_round1.jsonl,jsonl:data/rft_round1.jsonl,jsonl:data/rft_round1.jsonl,tinystories_instruct,tinystories_continue --per-source 12000 --validation-fraction 0.05 --epochs 3 --lr 3e-5 --warmup-steps 100 --min-lr-ratio 0.1 --batch-size 16 --max-input-tokens 128 --max-target-tokens 256 --teacher-tokens 220 --teacher-temperature 0.7 --teacher-batch-size 16 --allow-ungated --max-foundation-regression 0.35 --thinking-effort high --loop-range 1 8 --seed 42 --device cuda
# round 2, 3, ...: sample from the new checkpoint (new --seed), fine-tune with the new file weighted 2x plus the earlier files,
# and add --selection last: on a validation set made of the model's own samples the LM loss cannot improve, the verifier reward is the objective
```

Verifier reward on the 128 held-out prompts, greedy. Each round takes about 15 minutes on the RTX 5070 Ti (7 minutes of sampling with the KV cache, 5 of training, 2 of evaluation).

| Checkpoint | Verified samples kept | Reward, 4 loops | Words used | All 3 words | Ended | Reward, auto (loops used) |
| --- | --- | --- | --- | --- | --- | --- |
| base `long-02` | - | 0.214 | 0.17 | 0 | 116 | - |
| `story-sft-depth-01` | - | 0.459 | 0.35 | 4 | 95 | 0.438 (2.0) |
| `story-sft-depth-rft1` | 861 / 7,634 (11%) | 0.511 | 0.43 | 10 | 87 | 0.471 (2.0) |
| `story-sft-depth-rft2` | 1,263 / 7,620 (17%) | 0.580 | 0.49 | 13 | 98 | 0.547 (2.0) |
| `story-sft-depth-rft3` | 1,759 / 7,653 (23%) | **0.605** | 0.51 | **21** | 100 | **0.600** (2.0) |

The yield rises every round because the model it samples from is better, and every round so far has stayed inside the foundation-retention limit. The self-stopping checkpoint keeps stopping at 2 loops while gaining the same amount, so the gain is in the weights, not in extra compute. `play.py --best-of N` adds test-time selection on top: N samples, the verifier picks.

## Story-grounded Q&A

The model can also answer questions about a story it is given. That is extraction, not recall: the answer is in the prompt, which is what makes it reachable at 10M parameters. Gemma writes one question and a short answer per passage; a verifier keeps the pair only if the question is a real question, the answer is at most 12 words, and at least 60% of the answer's content words appear in the passage. 63% of Gemma's pairs pass.

The training prompt is `Read the story and answer the question.\n\nStory: <passage>\n\nQuestion: <question>` and the target is the short answer.

```powershell
uv run python -m src.qa_build --source tinystories_qa --per-source 6000 --output data/qa_train.jsonl --seed 42
uv run python -m src.qa_build --source tinystories_qa_valid --per-source 400 --output data/qa_valid.jsonl --seed 7
uv run python main.py sft --init-checkpoint checkpoints/loop-transformer-10m-story-sft-depth-rft3.pt --output checkpoints/loop-transformer-10m-story-qa-02.pt --datasets jsonl:data/qa_train.jsonl,jsonl:data/rft_round3.jsonl,jsonl:data/rft_round3.jsonl,tinystories_instruct,tinystories_continue --per-source 12000 --validation-fraction 0.05 --epochs 3 --lr 3e-5 --warmup-steps 100 --min-lr-ratio 0.1 --batch-size 16 --max-input-tokens 128 --max-target-tokens 256 --teacher-tokens 220 --teacher-temperature 0.7 --teacher-batch-size 16 --allow-ungated --max-foundation-regression 0.35 --thinking-effort high --loop-range 1 8 --selection last --seed 42 --device cuda
uv run python -m src.instruct_eval <checkpoints...> --task qa --prompts 128 --loops 4
```

119 held-out Q&A prompts, greedy at 4 loops. `F1` is token overlap with the reference answer, `grounded` the share of answer words found in the passage, `words` the mean answer length.

| Checkpoint | Reward | F1 | Exact | Grounded | Ended | Words |
| --- | --- | --- | --- | --- | --- | --- |
| `story-sft-depth-rft3` (stories only) | 0.087 | 0.05 | 0 | 0.29 | 89 | 93.2 |
| `story-qa-01` (Q&A weighted 2x) | 0.308 | 0.26 | 6 | 0.57 | 119 | 6.4 |
| `story-qa-02` (Q&A 1x, story self-samples 2x) | 0.306 | 0.26 | 6 | 0.57 | 119 | 6.6 |

Story reward on the same 128 story prompts, to show the cost of the second task:

| Checkpoint | Story reward | All 3 words |
| --- | --- | --- |
| `story-sft-depth-rft3` | 0.605 | 21 |
| `story-qa-01` | 0.536 | 16 |
| `story-qa-02` | **0.579** | 17 |

Before Q&A training the model answered every question with a 93-word story and ignored the question; after it, it produces a short answer, stops cleanly on every prompt, and takes more than half its words from the passage. Learning the second task costs some story quality: weighting Q&A 2x dropped story reward to 0.536, and weighting the story self-samples higher instead recovered it to 0.579 while Q&A stayed flat. `story-qa-02` is the model to use for both tasks; `story-sft-depth-rft3` remains the story specialist at 0.605.

What this does not add: world knowledge (`What is the capital of France`) or arithmetic, which need a model far larger than 10M and were abandoned early in this project, and chitchat such as `hello`, for which there is no data in the mix at all.

## Scope

- A 10M model is a simple-story continuer at best. Do not use it for medical, legal, financial, or safety-critical decisions.
- More loops add compute; they do not create missing knowledge.
- Never train on benchmark test splits. Follow every dataset card and Gemma's [terms](https://ai.google.dev/gemma/terms).
