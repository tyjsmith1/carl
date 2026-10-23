# Training a custom "hey Carl" wake word

MVP ships with the pre-trained `hey_jarvis` model from OpenWakeWord (see `docs/DECISIONS.md` ADR-016 for why). This guide walks through training a custom "hey Carl" model so Carl actually responds to its own name.

**Total time**: ~1 hour of mostly-idle Colab runtime + ~5 minutes of Carl config.
**Prerequisites**: a Google account (free Colab is enough), nothing installed locally.
**Cost**: $0 if you stay in Colab free tier.

## What you get out of it

- A `.onnx` model file (~1-2 MB) at `config/wake_words/hey_carl.onnx`
- Wake phrase changes from "hey Jarvis" to "hey Carl"
- Everything else about Carl unchanged

## Step 1 — Open the training notebook

Open OpenWakeWord's simplified training notebook in Colab:

**https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing**

(If that link ever rots, the canonical source is linked from the "Training New Models" section of the [openWakeWord README](https://github.com/dscripka/openWakeWord).)

**Runtime settings**: at the top of Colab, `Runtime → Change runtime type → T4 GPU`. Free tier gives you enough. The notebook will run without GPU, just slower.

## Step 2 — Set the target phrase

Find the cell that defines the training phrase (usually near the top, labeled something like "Set your target wake word"). Change it to:

```python
target_phrase = "hey carl"
```

Lowercase — the model doesn't care about case, and the synthesis engine sometimes trips on Proper Nouns.

## Step 3 — Run all cells

`Runtime → Run all` (or Ctrl+F9). The notebook will:

1. **Install dependencies** (~2 min) — `openwakeword`, `piper-tts`, `torch`, etc.
2. **Generate synthetic training data** (~15-30 min) — synthesizes ~1000-3000 "hey Carl" utterances via Piper TTS across many voices/accents, plus a matched pool of "negative" utterances (things that should NOT trigger).
3. **Train the classifier** (~15-30 min) — fine-tunes a small classifier on top of OpenWakeWord's frozen feature extractor.
4. **Evaluate on a held-out test set** — reports accuracy, false-accept rate, false-reject rate.
5. **Package the model** — outputs a `.onnx` (and often `.tflite`) file to `/content/models/` in the Colab filesystem.

**What to watch for while it runs:**
- The synthesis step often prints Piper voices as it loops through them — normal, ignore.
- If Colab kicks you off (idle timeout), just re-run the failing cell.
- The final evaluation should show accuracy >95% on the test set. If it's lower, see Troubleshooting below.

## Step 4 — Download the model

Once training finishes, the notebook typically has a cell that either auto-downloads the `.onnx` or gives you a path in the Colab filesystem. If the latter:

1. Open Colab's file browser (left sidebar, folder icon)
2. Navigate to the models directory (usually `/content/models/` or similar)
3. Right-click the `hey_carl.onnx` file → **Download**

Save it to your local `~/Downloads/hey_carl.onnx`.

## Step 5 — Drop it into the repo

```bash
cd ~/dev/carl
mkdir -p config/wake_words
mv ~/Downloads/hey_carl.onnx config/wake_words/hey_carl.onnx
```

Note: wake-word model files are gitignored (`config/wake_words/*.onnx` / `*.tflite` / `*.ppn`) because they're per-user — no need to commit.

## Step 6 — Point the config at it

Edit `config/carl.yaml`:

```yaml
wake_word:
  openwakeword:
    model_name: hey_carl                                    # descriptive only when model_path is set
    model_path: config/wake_words/hey_carl.onnx             # ← this is what activates the custom model
    threshold: 0.5                                          # start here; tune per Step 8
    inference_framework: onnx
```

Only `model_path` matters — the rest is metadata for logging/config validation.

**Alternative**: put the same override in `config/carl.local.yaml` (also gitignored) so the checked-in `carl.yaml` stays on `hey_jarvis` for anyone else who clones the repo.

## Step 7 — Verify

```bash
.venv/bin/carl
```

You should see this in the startup log:

```
openwakeword_loaded  framework=onnx  model=hey_carl  threshold=0.5
IDLE. listening for "hey carl"... (Ctrl-C to quit)
```

Say **"Hey Carl"** into your mic. Expected event:

```
wake_detected  model=hey_carl  score=0.9ish
[FSM] IDLE → RECORDING
```

If the score is below 0.5 (or Carl doesn't fire at all), skip to Troubleshooting.

## Step 8 — Threshold tuning

Custom-trained models often need a threshold different from the pretrained default. Two failure modes to balance:

| Symptom | Meaning | Fix |
|---|---|---|
| Fires on TV, music, other speech | False accepts | Raise threshold (0.6, 0.7) |
| You say "hey Carl" and nothing happens | False rejects | Lower threshold (0.4, 0.3) |

Best method: use it normally for a day. Grep the log for `wake_detected` and look at the `score` values — for the wake attempts that worked, note the range; for missed wakes, you won't have a log entry, but you'll remember.

For reference: the pretrained `hey_jarvis` reliably scored 0.94-0.99 in our testing at threshold 0.5. A well-trained custom model should be similar; if yours is consistently in the 0.6-0.8 range on clean utterances, drop the threshold to 0.4 and it'll feel more responsive.

## Troubleshooting

**Training accuracy < 90%**
The phrase might be too short or acoustically ambiguous. "Hey Carl" is short (2 syllables), which is on the harder end for wake words (compare to "hey Jarvis" — 3 syllables, more phonetic distinctiveness). Options:
- Increase training data volume: bump the number of synthesized samples in the notebook's data-gen cell (`num_samples = 2000` → `5000`)
- Try a slightly longer phrase: "hey Carl please" or "okay Carl"
- Add more negative samples (Colab has a cell for this) so the model learns what's NOT the wake word

**Model file too small (< 500 KB) or won't load**
Training probably didn't complete. Re-run the training cell, watch for Python errors.

**Carl loads but never fires wake_detected**
- Confirm the mic is working: `.venv/bin/carl audio list` and speak; should show input activity
- Threshold too high: temporarily drop to 0.2 and see if it fires on anything, then walk it back up
- Model file corrupt: re-download from Colab

**Carl fires on random noise all the time**
Threshold too low OR training data too narrow. Raise threshold first; if that doesn't help, retrain with more diverse negative samples (background music, TV audio, other speech).

**Colab session disconnected mid-training**
Free tier has a ~12-hour idle limit. If you disconnect: reopen the notebook, `Runtime → Run all` again. The synthesis step is idempotent (regenerates from scratch); training will restart but shouldn't take long since data-gen is cached.

## Optional refinements (later)

- **Voice-specific tuning**: modify the notebook to weight Piper voices toward gender/accent similar to yours. Small quality bump on your specific voice; small quality drop on everyone else's.
- **Household negative samples**: record ~30 min of your typical background audio (TV, kitchen, kids, whatever) and add it to the notebook's negative-sample pool. Big reduction in false accepts.
- **Retraining periodically**: if OpenWakeWord ships new feature-extractor versions (they periodically do), retrain against the newer base for better accuracy.

## Reverting to hey_jarvis

If the custom model disappoints, revert `config/carl.yaml`:

```yaml
wake_word:
  openwakeword:
    model_name: hey_jarvis
    # model_path: (delete this line — falls back to pretrained by name)
    threshold: 0.5
    inference_framework: onnx
```

Restart carl. `hey_carl.onnx` stays on disk so you can re-enable it just by adding `model_path` back.

## Related

- Design decision: `docs/DECISIONS.md` ADR-016 (why OpenWakeWord over Porcupine)
- Backlog: `docs/BACKLOG.md` "v2+" section — this guide implements that item
- Spec: `docs/specs/pipeline.md` "OpenWakeWord implementation" section
