# C-WAH Benchmark (REVECA)

This directory contains the C-WAH benchmark implementation used in REVECA.

## Prerequisites

1. Clone the [VirtualHome API](https://github.com/xavierpuigf/virtualhome) repository one folder above this repository.
2. Download the simulator from [this link](https://drive.google.com/file/d/1JTrV5jdF-LQVwY3OsV3Jd3r6PRghyHBp/view) and place it in `../executable/`.
3. Set your OpenAI key as an environment variable:

```bash
export OPENAI_KEY=your_openai_api_key
```

## Setup

```bash
cd cwah
conda create -n reveca_cwah python=3.8
conda activate reveca_cwah
pip install -r requirements.txt
```

## Run

```bash
sh ./scripts/symbolic_obs_llm_llm.sh
```

## Configuration

To change model or runtime options, edit:

- `scripts/symbolic_obs_llm_llm.sh`

Common options:

- `--lm_id`: model name/version
- `--base-port`: environment port range
- `--executable_file`: path to VirtualHome executable
