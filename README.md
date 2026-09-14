# LoopTransformer

A **10M-parameter** text-only recurrent encoder-decoder trained from random weights. `google/gemma-3-270m-it` is the project's only teacher; it is frozen and used only to write SFT targets after the student has a healthy base checkpoint.

The stages follow the usual sequence at toy scale: foundation -> quality annealing -> SFT -> verifier-grounded optimization. Every stage has been run on the TinyStories base. The foundation gate itself never passed, so SFT used `--allow-ungated`; read the results below before trusting any checkpoint. [CHANGELOG.md](CHANGELOG.md) records the full history, including what was tried and rejected.

## Quick start

```powershell
uv sync
uv run python play.py --checkpoint checkpoints/loop-transformer-10m-story-evolved-03.pt --max-new-tokens 160
```

Then type a prompt in the trained format, for example `Write a short story for young children. Use the words: dog, ball, happy. Include dialogue.` Optional additions: `The story is about: <one line>` and `Include this sentence: <sentence>`. `/temp 0.7` gives more coherent, less varied stories; `/temp 0` is greedy. Checkpoints are not in git; the commands below rebuild them in about two hours on an RTX 5070 Ti.

## Repository layout

| Path | Role |
| --- | --- |
| `main.py` | Dispatches `pretrain`, `anneal`, `sft`/`posttrain`, `evolve`. |
| `play.py` | Interactive runner with sampling controls. |
| `src/model.py` | The loop transformer: recurrent encoder, latent thoughts, decoder with prompt prefill, sampling helper. |
| `src/pretrain.py` | Foundation trainer: streamed mixture loader, windowing, quality gate, anchor, rollout-guarded selection. |
| `src/pretrain_distill.py` | Shared training loop, checkpoint save/load, the SFT stage with its foundation-retention guard. |
| `src/data.py` | SFT data sources (TinyStoriesInstruct story prompts, continuation replay, GSM8K/ARC/CommonsenseQA/finance/Dolly) and the story verifier. |
| `src/gemma_teacher.py` | Gemma 3 270M IT access, batched generation, append-only target cache, story mode. |
| `src/evolution.py` | CEM over the latent-thought offset with the answer reward and the rule-verified story reward. |
| `src/quant.py` | TurboQuant KV-cache compression: random rotation, Lloyd-Max codebooks, 1-bit QJL residual, bit-packed storage. |
| `src/kv_bench.py` | KV-cache report: exactness, speed-up, fidelity per bit width, bytes per token. |
| `src/story_eval.py` | Frozen 32-story continuation report for foundation checkpoints. |
| `src/instruct_eval.py` | Verifier reward and variety report for SFT/evolution checkpoints (produces the table below). |
| `src/student_tokenizer.py` | 8k byte-level BPE trainer, embedded in checkpoints. |
| `configs/tinystories_foundation.json` | The pinned foundation corpus. |
| `tests/` | 93 tests; `uv run python -m pytest -q`. |

## What the model is

| Part | Current implementation (`--architecture compact-10m`) |
| --- | --- |
| Tokenizer | 8,192-token byte-level BPE trained on the foundation corpus and embedded in the checkpoint (`--tokenizer compact-bpe`). Gemma's 262k vocabulary would alone exceed the parameter budget; GPT-2's 50k vocabulary spent 80% of the weights on the embedding table. |
| Encoder | Input tokens plus four learned latent-thought tokens; one 256-wide block reused at every loop. |
| Decoder | Six causal blocks that cross-attend to the final encoder state. The last 24 prompt tokens are also copied into the decoder (`--decoder-prompt-prefill-tokens 24`) so the first generated word does not depend on cross-attention alone. |
| Parameters | 10,015,296, tied embedding and LM head. |
| Context | Architecture limit 512 tokens. Trained on 128 input + 160 target tokens; `play.py` truncates longer prompts to the trained limit. |

`/effort low`, `medium`, and `high` run 1, 3, and 6 encoder loops. Loops never stop early; there is no halting head. Generation stops at EOS or the token limit.

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

## Scope

- A 10M model is a simple-story continuer at best. Do not use it for medical, legal, financial, or safety-critical decisions.
- More loops add compute; they do not create missing knowledge.
- Never train on benchmark test splits. Follow every dataset card and Gemma's [terms](https://ai.google.dev/gemma/terms).
