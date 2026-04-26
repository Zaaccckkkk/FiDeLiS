# FiDELIS: Faithful Reasoning of Large Language Model on Knowledge Graph Question Answering

This repo contains the implementation of paper ["FiDELIS: Faithful Reasoning of Large Language Model on Knowledge Graph Question Answering"](https://arxiv.org/abs/2405.13873). **FiDELIS** is a novel framework that synergizes Large Language Models (LLMs) with Knowledge Graphs (KGs) to enable faithful and interpretable reasoning.

## Why FiDELIS?

Large Language Models (LLMs) often struggle with hallucinations, especially in complex reasoning tasks. While Knowledge Graphs (KGs) can help mitigate this issue, existing KG-enhanced methods face challenges in:
- Accurately retrieving knowledge
- Efficiently traversing KGs at scale
- Maintaining interpretability of reasoning steps

FiDELIS addresses these challenges through its innovative approach to KG-enhanced reasoning.

## Key Features

- **Planning-Retrieval-Reasoning Framework**: Combines LLM capabilities with KG structure for faithful reasoning (i.e., all the answers are anchored in KG facts and are verified step by step)
- **Fast Reasoning w/ Path-RAG Module**: Pre-selects candidate sets to reduce computational costs; cuts search space & 1.7x faster than agents
- **Training-free Approach**: Efficient and generalizable solution without requiring model training & Scales to massive KGs
- **Transparent Steps**: Full reasoning paths & Audit-ready explanations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Y-Sui/FiDeLiS.git fidelis
cd fidelis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in `config.json`:
```json
{
    "OPENAI_API_KEY": "your-api-key-here"
}
```

## Quick Start

Run FiDELIS on the WebQSP dataset:
```bash
python main.py --d RoG-webqsp --model_name gpt-3.5-turbo-0125
```

## Usage

### Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--d` | Dataset choice (RoG-webqsp, RoG-cwq, or CL-LT-KGQA) | RoG-webqsp |
| `--model_name` | LLM model to use | gpt-3.5-turbo-0125 |
| `--top_n` | Number of top candidates to consider | 30 |
| `--top_k` | Beam size for search | 3 |
| `--max_length` | Maximum path length | 3 |
| `--strategy` | Search strategy | discrete_rating |
| `--verifier` | Verification method | deductive+planning |

### Advanced Configuration

- **Beam Search**: Adjust `--top_k` to control the beam width
- **Path Length**: Modify `--max_length` to handle longer reasoning chains
- **Verification**: Choose between different verification strategies with `--verifier`

## Project Structure

```
fidelis/
├── src/                    # Source code
│   ├── llm_navigator.py    # Core LLM navigation logic
│   └── utils/             # Utility functions
├── datasets/              # Dataset files
├── scripts/              # Helper scripts
├── main.py              # Main execution script
└── requirements.txt     # Project dependencies
```

## How It Works

1. **Planning Phase**: Generates relation paths grounded by KGs
2. **Retrieval Phase**: Uses Path-RAG to efficiently fetch relevant knowledge
3. **Reasoning Phase**: Conducts step-wise beam search with deductive verification
4. **Verification**: Validates each reasoning step and halts when the answer is deducible

## Automated Evaluation Pipeline

### Prerequisites

For inference-only experiments, install the lightweight runtime dependencies:

```bash
pip install -r requirements-inference.txt
```

On Windows, use `py` for the commands below. In Colab or Linux/macOS, use `python`.

### API Key

Set your OpenAI API key without committing it:

```powershell
$env:OPENAI_API_KEY="your-key-here"
```

or create a local `config.json` from `config.example.json`. The real `config.json`, `.env`, caches, and generated results are ignored by git.

### Smoke Test

Run the final smoke test before larger experiments:

```powershell
py scripts/final_smoke_test.py
```

Optional three-sample smoke test:

```powershell
py scripts/final_smoke_test.py --run_three_sample
```

### Mini Grid

Start with a tiny grid before spending money on a larger run:

```powershell
py scripts/run_eval_pipeline.py --sample_size 3 --datasets CL-LT-KGQA --top_k_values 1,2 --max_length_values 1,2 --output_path results_grid_test --resume
```

### Full Grid

After the mini grid passes, run the full grid:

```powershell
py scripts/run_eval_pipeline.py --sample_size 50 --top_k_values 1,2,3 --max_length_values 1,2,3 --output_path results_grid --resume
```

Colab example:

```bash
python scripts/run_eval_pipeline.py --sample_size 50 --top_k_values 1,2,3 --max_length_values 1,2,3 --output_path /content/drive/MyDrive/fidelis/results_grid --resume
```

### Resume Interrupted Runs

Use `--resume` to skip grid cells that already have valid, non-empty prediction files with enough rows and no duplicate ids:

```powershell
py scripts/run_grid.py --sample_size 50 --top_k_values 1,2,3 --max_length_values 1,2,3 --output_path results_grid --resume
```

### Output Files

Grid runs write timestamped run folders under:

```text
<output_path>/<model_name>/
```

The top-level manifest is:

```text
<output_path>/grid_manifest.jsonl
```

Validation writes:

```text
<output_path>/grid_output_check.csv
```

Aggregation writes:

```text
<output_path>/grid_eval_run_summary.csv
<output_path>/grid_eval_per_sample_all_runs.csv
<output_path>/grid_eval_best_config_per_sample.csv
<output_path>/grid_eval_best_config_per_sample.jsonl
```

`grid_eval_run_summary.csv` has one row per run/configuration with mean metrics. `grid_eval_per_sample_all_runs.csv` has every sample under every run and prediction field. `grid_eval_best_config_per_sample.csv` selects the best configuration per `(dataset, id)` using hit, F1, accuracy, then smaller depth/beam settings.

Accuracy or F1 of 0 is not a pipeline crash. Pipeline failures are crashes, missing outputs, empty prediction files, invalid JSONL, duplicate ids, or missing evaluation artifacts.

API calls can cost money and may hit rate limits. Run `sample_size=3` mini grids before `sample_size=50` or `sample_size=100`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{sui2025fidelisfaithfulreasoninglarge,
       title={FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering}, 
       author={Yuan Sui and Yufei He and Nian Liu and Xiaoxin He and Kun Wang and Bryan Hooi},
       year={2025},
       eprint={2405.13873},
       archivePrefix={arXiv},
       primaryClass={cs.AI},
       url={https://arxiv.org/abs/2405.13873}, 
}
```
