# TDW Multi-Agent Transport (TDW-MAT)

## Codebase Layout

```
|__ tdw-gym/                       main code
|       |__ challenge.py            main evaluation code
|       |__ tdw_gym.py              main environment code
|       |__ lm_agent.py             Ours
|
|__ scene_generator/                dataset generation code
|
|__ dataset/                        dataset configuration & storage
|
|__ transport_challenge_multi_agent/  low-level environment controller
|
|__ scripts/                        experiment scripts
|
|__ REVECA_LLM                      Ours
```

---

## Setup

Run the following commands step by step to set up the default environment:

```bash
cd tdw-mat
conda create -n tdw_mat python=3.9
conda activate tdw_mat
pip install -r requirements.txt
```

---

## Run Experiments

Example scripts are provided under `scripts/`.  

For instance, to run experiments with **two LLM Agents**:

```bash
sh ./scripts/test_LMs-gpt-4o-mini.sh
```

This repository currently includes the script above under `scripts/`.  
