# Vanilla Zero-Shot Baseline

This baseline measures direct GPT performance without FiDeLiS retrieval or reasoning-path search. It is meant to identify examples where zero-shot GPT is weak, so later router/RAG experiments can focus on queries that plausibly need larger retrieval space.

## Configure

Do not put a real API key in `config.example.json`. That file is a checked-in template.
Create a local ignored config instead:

```json
{
  "OPENAI_API_KEY": "your-key",
  "model": "gpt-3.5-turbo-0125"
}
```

Save it as `vanilla_baseline/config.local.json`. The repository `.gitignore` excludes this file and other key-bearing JSON names under `vanilla_baseline/`.

## Dry run sampling

```bash
python -m vanilla_baseline.run_zero_shot \
  --datasets webqsp cwq crlt \
  --sample_size 300 \
  --seed 17 \
  --dry_run
```

This creates persistent manifests under `results/vanilla_zero_shot/manifests/`. Increasing `--sample_size` later extends the same manifest without reselecting existing examples.

## Run GPT zero-shot

```bash
python -m vanilla_baseline.run_zero_shot \
  --datasets webqsp cwq crlt \
  --sample_size 300 \
  --seed 17 \
  --config_path vanilla_baseline/config.local.json \
  --model gpt-3.5-turbo-0125
```

Outputs:

- `results/vanilla_zero_shot/manifests/{dataset}_seed17.jsonl`
- `results/vanilla_zero_shot/predictions/predictions_{dataset}.jsonl`
- `results/vanilla_zero_shot/summaries/{dataset}_summary.json`
- `results/vanilla_zero_shot/plots/{dataset}_{metric}_hist.png`

Each prediction record keeps the exact question, ground-truth answers, parsed predictions, Hit, F1, precision, recall, accuracy, API usage, and status. Existing successful records are skipped by default, so interrupted runs can resume.
