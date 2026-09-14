
[日本語版 README はこちら / Japanese README](README_Ja.md)
> SGIC (SMBC Global Investment & Consulting Ltd.) 


<h3 align="center"><i><b>
SGIC FinEval: Unified framework for evaluating LLMs on English and Japanese financial benchmarks.
</b></i></h3>

<p align="center">
SGIC FinEval is a unified framework for evaluating large language models on English and Japanese financial benchmarks. <br>
It supports three model types (pre-trained `base` / instruction-tuned `inst` / instruction-trained `reasoning` models that emit a reasoning trace), so you can select the evaluation setup that matches a model's training stage.
</p>


The implementation in this repository is the evaluation framework used in the following paper (in Japanese), presented at FIT2026:

> 汎用的な能力を維持した金融特化型LLMの開発. \
> 齋藤 幸史郎 (東京科学大学), 白方 健司, 西塔 明, 山田 裕文 (SMBC Global Investment & Consulting Ltd.), Ma Youmi (東京科学大学), 岡崎 直観 (東京科学大学 / 産業技術総合研究所 / NII LLMC). \
> 第25回情報科学技術フォーラム (FIT2026), 2026.

<details>
  <summary><strong>Contents</strong></summary>

- [📊 Included Benchmarks](#-included-benchmarks)
- [💻 Requirements](#-requirements)
- [🔧 Initial Setup](#-initial-setup)
   - [📥 Clone this repository](#-clone-this-repository)
   - [🔑 Create `.env`](#-create-env)
   - [📦 Build the environment](#-build-the-environment)
- [🚀 Running Evaluations](#-running-evaluations)
   - [📁 Results](#-results)
- [🧩 Patches to Existing Benchmarks](#-patches-to-existing-benchmarks)
- [📜 License](#-license)
- [📚 Citations](#-citations)

</details>


# 📊 Included Benchmarks

| Benchmark | Task | Metrics | Repo / Dataset |
| -- | -- | -- | -- |
| [CFA](https://huggingface.co/datasets/TheFinAI/flare-cfa) | Financial Knowledge QA (CFA exam–level finance and calculation questions) | Acc, F1 | [PIXIU](https://github.com/The-FinAI/PIXIU) |
| [FinQA](https://huggingface.co/datasets/TheFinAI/flare-finqa) | Financial Numerical Reasoning / QA (numerical reasoning over financial statements) | Acc | [PIXIU](https://github.com/The-FinAI/PIXIU) |
| [FPB](https://huggingface.co/datasets/TheFinAI/en-fpb) | Financial Sentiment Analysis (sentiment classification of financial news) | Acc, F1 | [PIXIU](https://github.com/The-FinAI/PIXIU) |
| [chabsa](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/chabsa) | Aspect-Based Sentiment Analysis (aspect-level sentiment in Japanese financial texts) | Acc, F1 | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [CMA Basics](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/cma_basics) | Financial Knowledge QA (management accounting / CMA basics) | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [CPA Audit](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/cpa) | Financial Knowledge QA (CPA-level auditing knowledge) | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [FP2](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/fp2) | Financial Knowledge QA (Japanese FP Level 2 exam knowledge) | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [Security Sales 1](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/security_sales_1) | Financial Knowledge QA (securities sales representative exam knowledge) | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| dr-finance-instruction | Japanese Financial Instruction Following (LLM-as-judge accuracy) | Acc | [DeepResaerch/dr-finance-instruction](https://huggingface.co/datasets/DeepResaerch/dr-finance-instruction) |

All three benchmark suites (`PIXIU`, `japanese-lm-fin-harness`,
`dr-finance-instruction`) support all three model types.


# 💻 Requirements

- Linux with an NVIDIA GPU (inference runs on [vLLM](https://github.com/vllm-project/vllm); PyTorch wheels are pinned to CUDA 12.8)
- [uv](https://docs.astral.sh/uv/) (Python 3.10.14)
- `git`, network access at setup time (benchmark repositories are not vendored; they are fetched from GitHub)
- A Hugging Face token (to download models/datasets)
- An OpenAI API key (only for `dr-finance-instruction`, which uses an OpenAI model as judge)


# 🔧 Initial Setup

## 📥 Clone this repository

```sh
git clone https://github.com/pakapaka333/SGIC_FinEval.git
cd SGIC_FinEval
```

## 🔑 Create `.env`

Edit `.env_template` and set the following:

| Var | Description |
| -- | -- |
| `PROJECT_ROOT` | Required: absolute path to the root of this repository. |
| `HF_TOKEN` | Required: your Hugging Face token. |
| `OPENAI_API_KEY` | Optional: your OpenAI API key (required for `dr-finance-instruction`). |
| `UV_CACHE_DIR` | Optional: cache dir for uv. |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | Optional: cache dirs for Hugging Face. |

Once configured, copy it to `.env`:

```sh
cp .env_template .env
```

## 📦 Build the environment

```sh
bash scripts/build_env.sh
```

This script (i) fetches the upstream benchmark repositories into `benchmarks/` at the commits pinned in [benchmarks.tsv](benchmarks.tsv), (ii) applies the patches under [patches/](patches), and (iii) creates the project venv.


# 🚀 Running Evaluations

```sh
# Evaluate on a single specified benchmark
$ bash scripts/evaluate.sh <model_name> <benchmark_name> <model_type>

# Evaluate on all benchmarks
$ bash scripts/run_all.sh <model_name> <model_type> [benchmark ...]
```

- `model_name`: a Hugging Face model ID (e.g. `Qwen/Qwen3-14B`) or a local path
- `benchmark_name`: `PIXIU` | `japanese-lm-fin-harness` | `dr-finance-instruction`
- `model_type`:
  - `base` — for pre-trained models. Few-shot examples are included to elicit answers.
  - `inst` — for instruction-tuned models. The model's chat template is applied. No few-shot examples.
  - `reasoning` — for instruction-tuned models that emit a reasoning trace. Like `inst`, plus a large generation budget (32k tokens) and stripping of thinking segments (e.g. `</think>`) before scoring.

Example:

```sh
bash scripts/evaluate.sh tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 PIXIU inst

bash scripts/run_all.sh tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 reasoning
```

## 📁 Results

Per-benchmark results are written to:

```
results/<model_name>/<model_type>/<benchmark_name>/results_<jobid>.json
```

After each run, `scripts/aggregate_results/aggregate_results.py` also collects the latest result across all benchmarks for that model into:

```
results/<model_name>/<model_type>/aggregate_results.jsonl
```


# 🧩 Patches to Existing Benchmarks

This repository does **not** vendor the code of the existing financial benchmarks. \
The commits used are pinned in [benchmarks.tsv](benchmarks.tsv), and the changes required by this repository are maintained as patches:

| Patch | Applies to | What it does |
| -- | -- | -- |
| [patches/PIXIU.patch](patches/PIXIU.patch) | [The-FinAI/PIXIU](https://github.com/The-FinAI/PIXIU) | Adds `*_inst` task variants (chat-style prompts) for FPB/CFA/FinQA and plumbs chat-mode options through the evaluator. |
| [patches/PIXIU--financial-evaluation.patch](patches/PIXIU--financial-evaluation.patch) | [The-FinAI/financial-evaluation](https://github.com/The-FinAI/financial-evaluation) | Extends the vLLM model wrapper with chat-template generation, stop words, larger generation limits, and thinking-segment stripping for reasoning models. |
| [patches/japanese-lm-fin-harness.patch](patches/japanese-lm-fin-harness.patch) | [pfnet-research/japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) | Adds generation-style task variants (`*_gen`) so reasoning models can be scored on free-form answers. |


The dr-finance-instruction data is fetched entirely from the Hugging Face Hub at run time.


# 📜 License

The code in this repository is released under the [MIT License](LICENSE). \
The patches under `patches/` are derivative works of the upstream projects (all MIT-licensed) — see [NOTICE](NOTICE) for details. \
Benchmark contents fetched at setup time remain under their own upstream licenses.


# 📚 Citations


```
## PIXIU
@misc{xie2023pixiu,
      title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance}, 
      author={Qianqian Xie and Weiguang Han and Xiao Zhang and Yanzhao Lai and Min Peng and Alejandro Lopez-Lira and Jimin Huang},
      year={2023},
      eprint={2306.05443},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{xie2024FinBen,
      title={The FinBen: An Holistic Financial Benchmark for Large Language Models}, 
      author={Qianqian Xie and Weiguang Han and Zhengyu Chen and Ruoyu Xiang and Xiao Zhang and Yueru He and Mengxi Xiao and Dong Li and Yongfu Dai and Duanyu Feng and Yijing Xu and Haoqiang Kang and Ziyan Kuang and Chenhan Yuan and Kailai Yang and Zheheng Luo and Tianlin Zhang and Zhiwei Liu and Guojun Xiong and Zhiyang Deng and Yuechen Jiang and Zhiyuan Yao and Haohang Li and Yangyang Yu and Gang Hu and Jiajia Huang and Xiao-Yang Liu and Alejandro Lopez-Lira and Benyou Wang and Yanzhao Lai and Hao Wang and Min Peng and Sophia Ananiadou and Jimin Huang},
      year={2024},
      eprint={2402.12659},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


## Japanese Language Model Financial Evaluation Harness
@preprint{Hirano2023-pre-finllm,
  title={{金融分野における言語モデル性能評価のための日本語金融ベンチマーク構築}},
  author={平野, 正徳},
  doi={10.51094/jxiv.564},
  year={2023}
}
@inproceedings{Hirano2023-finnlpkdf,
  title={{Construction of a Japanese Financial Benchmark for Large Language Models}},
  author={Masanori Hirano},
  booktitle={Joint Workshop of the 7th Financial Technology and Natural Language Processing (FinNLP), the 5th Knowledge Discovery from Unstructured Data in Financial Services (KDF), and The 4th Workshop on Economics and Natural Language Processing (ECONLP)},
  pages={1-9},
  doi={10.2139/ssrn.4769124},
  url={https://aclanthology.org/2024.finnlp-1.1},
  archivePrefix={arXiv},
  arxivId={2403.15062},
  year={2024}
}
```
