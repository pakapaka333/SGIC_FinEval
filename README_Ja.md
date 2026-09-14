
[English README is here](README.md)
> SGIC (SMBC Global Investment & Consulting Ltd.) 


<h3 align="center"><i><b>
SGIC FinEval: 英語・日本語の金融ベンチマークで大規模言語モデルを評価するための統合フレームワーク
</b></i></h3>

<p align="center">
SGIC FinEval は英語・日本語の金融ベンチマークで大規模言語モデルを評価するための統合フレームワークです。 <br>
3種類のモデルタイプ（事前学習済み `base` / 指示学習済み `inst` / 推論過程付き指示学習済み `reasoning`）に対応しているため、学習段階に応じて評価設定を選択できます。
</p>

　\
なお、本リポジトリの実装は、FIT2026 で発表した以下の論文で使用した評価フレームワークです:

> 汎用的な能力を維持した金融特化型LLMの開発. \
> 齋藤 幸史郎 (東京科学大学), 白方 健司, 西塔 明, 山田 裕文 (SMBC Global Investment & Consulting Ltd.), Ma Youmi (東京科学大学), 岡崎 直観 (東京科学大学 / 産業技術総合研究所 / NII LLMC). \
> 第25回情報科学技術フォーラム (FIT2026), 2026.

<details>
  <summary><strong>目次</strong></summary>

- [📊 収録ベンチマーク](#-収録ベンチマーク)
- [💻 動作要件](#-動作要件)
- [🔧 初回セットアップ](#-初回セットアップ)
   - [📥 このリポジトリを clone](#-このリポジトリを-clone)
   - [🔑 `.env` の作成](#-env-の作成)
   - [📦 環境構築](#-環境構築)
- [🚀 評価のやり方](#-評価のやり方)
   - [📁 結果の出力先](#-結果の出力先)
- [🧩 既存ベンチマークへのパッチ](#-既存ベンチマークへのパッチ)
- [📜 ライセンス](#-ライセンス)
- [📚 引用](#-引用)

</details>


# 📊 収録ベンチマーク

| ベンチマーク | タスク | 指標 | リポジトリ / データセット |
| -- | -- | -- | -- |
| [CFA](https://huggingface.co/datasets/TheFinAI/flare-cfa) | 金融知識QA（CFA試験レベルの金融・計算問題） | Acc, F1 | [PIXIU](https://github.com/The-FinAI/PIXIU) |
| [FinQA](https://huggingface.co/datasets/TheFinAI/flare-finqa) | 金融数値推論 / QA（財務諸表に対する数値推論） | Acc | [PIXIU](https://github.com/The-FinAI/PIXIU) |
| [FPB](https://huggingface.co/datasets/TheFinAI/en-fpb) | 金融感情分析（金融ニュースの感情分類） | Acc, F1 | [PIXIU](https://github.com/The-FinAI/PIXIU) |
| [chabsa](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/chabsa) | 観点別感情分析（日本語金融テキスト） | Acc, F1 | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [CMA Basics](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/cma_basics) | 金融知識QA（管理会計 / 証券アナリスト基礎） | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [CPA Audit](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/cpa) | 金融知識QA（公認会計士レベルの監査知識） | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [FP2](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/fp2) | 金融知識QA（FP技能検定2級レベル） | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| [Security Sales 1](https://github.com/pfnet-research/japanese-lm-fin-harness/tree/main/jlm_fin_eval/datasets/security_sales_1) | 金融知識QA（証券外務員一種レベル） | Acc | [japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) |
| dr-finance-instruction | 日本語金融指示追従（LLM-as-judge による正答率） | Acc | [DeepResaerch/dr-finance-instruction](https://huggingface.co/datasets/DeepResaerch/dr-finance-instruction) |

3つのベンチマークスイート（`PIXIU`, `japanese-lm-fin-harness`, `dr-finance-instruction`）はいずれも、3種類のモデルタイプすべてに対応しています。


# 💻 動作要件

- NVIDIA GPU を搭載した Linux 環境 (推論は [vLLM](https://github.com/vllm-project/vllm) を使用。PyTorch は CUDA 12.8 用 wheel に固定)
- [uv](https://docs.astral.sh/uv/) (Python 3.10.14)
- `git` およびセットアップ時のネットワーク接続 (ベンチマークのリポジトリは同梱されておらず、GitHub から取得します)
- Hugging Face トークン (モデル・データセットのダウンロード用)
- OpenAI API キー (`dr-finance-instruction` のみ。judge に OpenAI モデルを使用します)


# 🔧 初回セットアップ

## 📥 このリポジトリを clone

```sh
git clone https://github.com/pakapaka333/SGIC_FinEval.git
cd SGIC_FinEval
```

## 🔑 `.env` の作成

`.env_template` を編集して以下を設定してください:

| 変数 | 説明 |
| -- | -- |
| `PROJECT_ROOT` | 必須: このリポジトリのルートの絶対パス |
| `HF_TOKEN` | 必須: Hugging Face トークン |
| `OPENAI_API_KEY` | 任意: OpenAI API キー (`dr-finance-instruction` で必須) |
| `UV_CACHE_DIR` | 任意: uv のキャッシュディレクトリ |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | 任意: Hugging Face のキャッシュディレクトリ |

設定したのち `.env` としてコピーしてください:

```sh
cp .env_template .env
```

## 📦 環境構築

```sh
bash scripts/build_env.sh
```

このスクリプトは、(i) [benchmarks.tsv](benchmarks.tsv) に固定されたコミットで上流ベンチマークのリポジトリを `benchmarks/` に取得し、(ii) [patches/](patches) 以下のパッチを適用したうえで、(iii) プロジェクトの venv を作成します。


# 🚀 評価のやり方

```sh
# 1つのベンチマークを指定して評価する場合
$ bash scripts/evaluate.sh <model_name> <benchmark_name> <model_type>

# 全ベンチマークで評価する場合
$ bash scripts/run_all.sh <model_name> <model_type> [benchmark ...]
```

- `model_name`: Hugging Face のモデルID（例: `Qwen/Qwen3-14B`）またはローカルパス
- `benchmark_name`: `PIXIU` | `japanese-lm-fin-harness` | `dr-finance-instruction`
- `model_type`:
  - `base` — 事前学習済みモデル向け。問題に応えるための few-shot あり。
  - `inst` — 指示学習済みモデル向け。モデルの chat template を適用。few-shot なし。
  - `reasoning` — 推論過程付き指示学習済みモデル向け。`inst` に加えて、大きな生成上限（32kトークン）と、採点前の思考部分（`</think>` など）の除去を行う。

例:

```sh
bash scripts/evaluate.sh tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 PIXIU inst

bash scripts/run_all.sh tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 reasoning
```

## 📁 結果の出力先

ベンチマークごとの結果は以下に書き出されます:

```
results/<model_name>/<model_type>/<benchmark_name>/results_<jobid>.json
```

また、各実行の最後に `scripts/aggregate_results/aggregate_results.py` が、そのモデルの全ベンチマークの最新結果を以下に集約します:

```
results/<model_name>/<model_type>/aggregate_results.jsonl
```


# 🧩 既存ベンチマークへのパッチ

このリポジトリは、既存の金融ベンチマークのコードを**同梱しません**。 \
使用するコミットは [benchmarks.tsv](benchmarks.tsv) に固定されており、本レポジトリ都合の変更はパッチとして管理しています:

| パッチ | 適用先 | 内容 |
| -- | -- | -- |
| [patches/PIXIU.patch](patches/PIXIU.patch) | [The-FinAI/PIXIU](https://github.com/The-FinAI/PIXIU) | FPB / CFA / FinQA に chat 形式プロンプトの `*_inst` タスクを追加し、evaluator に chat モードのオプションを追加 |
| [patches/PIXIU--financial-evaluation.patch](patches/PIXIU--financial-evaluation.patch) | [The-FinAI/financial-evaluation](https://github.com/The-FinAI/financial-evaluation) | vLLM モデルラッパーを拡張（chat template による生成、stop words、生成上限の拡大、reasoning モデルの思考部分の除去） |
| [patches/japanese-lm-fin-harness.patch](patches/japanese-lm-fin-harness.patch) | [pfnet-research/japanese-lm-fin-harness](https://github.com/pfnet-research/japanese-lm-fin-harness) | reasoning モデルを自由記述回答で採点するための生成型タスク（`*_gen`）を追加 |


dr-finance-instruction のデータは、すべて実行時に Hugging Face Hub から取得されます。


# 📜 ライセンス

このリポジトリのコードは [MIT License](LICENSE) で公開しています。 \
`patches/` 以下のパッチは各上流プロジェクト（いずれも MIT ライセンス）の派生物です (詳細は [NOTICE](NOTICE) を参照してください)。 \
セットアップ時に取得されるベンチマークの内容は、それぞれの上流のライセンスに従います。


# 📚 引用


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