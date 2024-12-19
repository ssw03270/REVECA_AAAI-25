## **REVECA**: Adaptive Planning and Trajectory-based Validation in Cooperative Language Agents using Information Relevance and Relative Proximity

![main](https://github.com/user-attachments/assets/bbfd0d91-c3bf-49f6-8541-71921151314e)

This repository contains the code for the following paper.
Paper Link: [arxiv](https://arxiv.org/abs/2405.16751)

### Setup Instructions
1. Clone the [VirtualHome API](https://github.com/xavierpuigf/virtualhome) repository one folder above this repository.
2. Download the [simulator](https://drive.google.com/file/d/1JTrV5jdF-LQVwY3OsV3Jd3r6PRghyHBp/view), and place it in an *executable* folder, located one folder above this repository.
3. Create a conda environment and install the required dependencies:

```
conda create --name reveca python=3.8
conda activate reveca 
pip install -r requirements.txt 
```

### Running the Code
To run the code, execute the following command:
```
sh ./scripts/symbolic_obs_llm_llm.sh
```

### Changing the LLM Version
If you want to change the version of the LLMs, navigate to the *scripts* folder and edit the *symbolic_obs_llm_llm.sh* file. 

Modify the *--lm_id* parameter to specify the version of GPT you want to use.

Examples of *--lm_id*:
- gpt-4o-mini-2024-07-18
- gpt-3.5-turbo-0125

### TODO List
- [x] C-WAH setting.
- [ ] TDW-MAT setting.
- [ ] Overcooked-AI setting.

### Citation
If you find this work useful in your research, please cite:
> ```bibtex
> @article{seo2024llm,,
>   title={REVECA: Adaptive Planning and Trajectory-based Validation in Cooperative Language Agents using Information Relevance and Relative Proximity},
>   author={SeungWon Seo and SeongRae Noh and Junhyeok Lee and SooBin Lim and Won Hee Lee and HyeongYeop Kang},
>   journal={arXiv preprint arXiv:2405.16751v2},
>   year={2024}
> }
> ```

This code is derived from the code of the following paper:
- [Building Cooperative Embodied Agents Modularly with Large Language Models](https://arxiv.org/abs/2307.02485) by Zhang et al.
