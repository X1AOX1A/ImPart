# [[ACL 2025] ImPart: Importance-Aware Delta-Sparsification for Improved Model Compression and Merging in LLMs](https://arxiv.org/abs/2504.13237)

The official repository containing the introduction and code for our ACL 2025 paper: [ImPart: Importance-Aware Delta-Sparsification for Improved Model Compression and Merging in LLMs](https://arxiv.org/abs/2504.13237).

<p align="center">|
  <a href="#-news"> 🔥 News</a> |
  <a href="#-motivation">💡 Motivation</a> |
  <a href="#-seqar">🔖 Method</a> |
</p>

<p align="center">|
  <a href="#️-quick-start"> ⚡️ Quick Start </a> |
  <a href="#-citation">📓 Citation</a> | 
  <a href="https://arxiv.org/abs/2407.01902">📃 Paper </a>|
</p>

# 🔥 News
- **Jan 2025**: Our paper has been accepted by **NAACL 2025 main conference**.
- **Sep 2024**: We released our code and quick start.
- **Jul 2024**: We released our paper on [**arxiv**](https://arxiv.org/abs/2407.01902).

# 💡 Motivation
- LLMs are more susceptible to distractions when responding as multiple characters sequentially.
- Different characters specialize in distinct malicious instructions. The combination of different characters further amplifies the effectiveness of attack.

# 🔖 SeqAR
### SeqAR: **Seq**uential **A**uto-generated cha**R**acters 
- Design and optimize the jailbreak templates automatically.
- Jailbreak target LLMs by asking them to act as malicious characters sequentially.

<span id="SeqAR"></span>
![SeqAR](./assets/imgs/SeqAR_overview.png)


# ⚡️ Quick Start
## Requirments
Install all the packages from **requirments.txt**
```
conda create -n seqar python=3.10 -y
conda activate seqar
git clone https://github.com/sufenlp/SeqAR.git
cd SeqAR
pip install -r requirements.txt
```

## Data
* 
* You can add more datasets in [./data]() referring to the existed csv or jsonl files.

## Model & Benchmark
| Task | Fine-tuned                                                                   | Backbone                                                        | Benchmark                                             | Benchmark      |
|------|------------------------------------------------------------------------------|-----------------------------------------------------------------|-------------------------------------------------------|----------------|
| Math | [WizardMath-13B-V1.0](https://huggingface.co/vanillaOVO/WizardMath-13B-V1.0) | [LLaMA-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MATH           |
| Code | [WizardCoder-13B]()                                                          | [CodeLlama-13B](codellama/CodeLlama-13b-hf)                     | HumanEval                                             | MBPP           | 
| Chat | [LLaMA-2-13B-Chat]()                                                         | [LLaMA-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) | [IFEval](https://huggingface.co/datasets/google/IFEval)                                            | [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) |
| Chat | [LLaMA-2-7B-Chat]()                                                          | [LLaMA-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) | [IFEval](https://huggingface.co/datasets/google/IFEval)                                            | [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) |
| Chat | [LLaMA-3-8B-Instruct]()                                                      | [LLaMA-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) | [IFEval](https://huggingface.co/datasets/google/IFEval)                                            | [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) |


## 

## ImPart
- ImPart 
* **Optimization**: Get the jailbreak characters.
```
PYTHONPATH=your_project_pth \
python more_character.py \
    -p "folder with your experiment config"
```
* **Evaluation**: Test the jailbreak performance of characters' combination.
```
PYTHONPATH=your_project_pth \
python evaluate.py \
    -p "folder with experiment finished" \
    -m "llama_vllm" \
    -c 2
```

## ImPart + Quantization
- Apply GPTQ to the sparse singular vector of $\Delta W$, detailed in Section 7.1 and Appendix B.1.
- Extend GPTQ to accommodate sparse weight matrix as the following algorithm (Algorithm 2 in paper).
![impart_qt_algorithm](./assets/imgs/impart_qt_algorithm.png)
- The code is modified based on the implementation of [Delta-CoMe](https://github.com/thunlp/Delta-CoMe).
### Run Quantization
```shell
PYTHONPATH=your_project_pth \
python evaluate.py \
    -p "folder with experiment finished" \
    -m "llama_vllm" \
    -c 2
```
### Run Evaluation
- Same as 

### Compression Performance
![impart_qt_algorithm](./assets/imgs/impart_qt_results.png)
- **ImPart-Qt** achieves **nearly lossless performance** in the **Compression Ratio (CR) of 32**.

---

## IMPART + Model Merging

# 📓 Citation
If you find this repo useful for your research, please cite us as:
```bibtex
@misc{yang2025impart,
      title={ImPart: Importance-Aware Delta-Sparsification for Improved Model Compression and Merging in LLMs}, 
      author={Yan Yang and Yixia Li and Hongru Wang and Xuetao Wei and James Jianqiao Yu and Yun Chen and Guanhua Chen},
      year={2025},
      eprint={2504.13237},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.13237}, 
}
```