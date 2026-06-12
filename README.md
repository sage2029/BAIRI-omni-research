# Integrating Natural Language Processing with Deep Knowledge Tracing in ITS for Low-Resource Languages

[📝 Paper Status: Research Square](https://www.researchsquare.com/article/rs-9991485/v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end framework integrating tokenization-free character-level text features with a gated recurrent (LSTM) Deep Knowledge Tracing network core tailored specifically for non-segmented scripts like Burmese STEM curricula.

## Core Features
* **Tokenization-Free NLP Module:** Bypasses word segmentation dependencies via bounded character n-grams ($2$ to $4$) processed by a calibrated Linear SVM.
* **Recurrent Latent Core Engine:** Natively concatenates multimodal features ($v_{text} \parallel e(s_t) \parallel c_t$) into sequential LSTM cells.
* **Empirical Validation:** Extends tracking performance to an AUC of 0.824, yielding a +0.062 absolute improvement over traditional text-blind DKT baselines.

## Repository Layout
* `src/model.py`: Core neural pipeline architecture.
* `data/README.md`: Setup requirements for the simulated interaction datasets.

## Quick Start
```bash
# Clone and install dependencies
git clone [https://github.com/sage2029/cs-ai-research.git](https://github.com/sage2029/cs-ai-research.git)
cd its-hybrid-nlp-dkt-burmese
pip install -r requirements.txt

# Verify code implementation pipeline
python src/model.py

@article{oo2026integrating,
  title={Integrating Natural Language Processing with Deep Knowledge Tracing in Intelligent Tutoring Systems for Low-Resource Languages: A Case Study on Burmese STEM Curricula},
  author={Oo, Aung Ko Ko},
  journal={Preprint Under Review},
  year={2026}
}