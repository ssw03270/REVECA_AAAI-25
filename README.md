# **REVECA**: Adaptive Planning and Trajectory-based Validation in Cooperative Language Agents using Information Relevance and Relative Proximity

![main](https://github.com/user-attachments/assets/bbfd0d91-c3bf-49f6-8541-71921151314e)

This repository contains the implementation for:
- Paper: [REVECA (AAAI 2025)](https://arxiv.org/abs/2405.16751)

## Repository Structure

This repository now includes two benchmark tracks:

- `cwah/`: C-WAH benchmark code (VirtualHome-based)
- `tdw-mat/`: TDW Multi-Agent Transport benchmark code

Benchmark-specific instructions:

- C-WAH guide: [cwah/README.md](cwah/README.md)
- TDW-MAT guide: [tdw-mat/README.md](tdw-mat/README.md)

## Quick Start

### C-WAH

```bash
cd cwah
conda create -n reveca_cwah python=3.8
conda activate reveca_cwah
pip install -r requirements.txt
sh ./scripts/symbolic_obs_llm_llm.sh
```

### TDW-MAT

```bash
cd tdw-mat
conda create -n reveca_tdw python=3.9
conda activate reveca_tdw
pip install -r requirements.txt
sh ./scripts/test_LMs-gpt-4o-mini.sh
```

## Release Status

- [x] C-WAH setting
- [x] TDW-MAT setting
- [ ] Overcooked-AI setting

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{seo2025reveca,
  title={Reveca: Adaptive planning and trajectory-based validation in cooperative language agents using information relevance and relative proximity},
  author={Seo, SeungWon and Noh, SeongRae and Lee, Junhyeok and Lim, SooBin and Lee, Won Hee and Kang, HyeongYeop},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={22},
  pages={23295--23303},
  year={2025}
}
```

This code is derived from:
- [Building Cooperative Embodied Agents Modularly with Large Language Models](https://arxiv.org/abs/2307.02485) by Zhang et al.
