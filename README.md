# Collaborative Ideation with LLM: A Distributed Bayesian Inference Framework for Analyzing and Predicting Human-AI Co-Creation

**Paper Information**

This repository contains the experimental code for our paper:

* **Title:** "Collaborative Ideation with LLM: A Distributed Bayesian Inference Framework for Analyzing and Predicting Human-AI Co-Creation" 
* **Authors:** Momoha Hirose, Masatoshi Nagano, and Tadahiro Taniguchi 
* **Note:** This work is a substantially extended and revised version of our preliminary work presented at ARSO 2025.

## Description

This repository provides the experimental code for our paper, which formulates human-LLM collaborative ideation—where an LLM proposes ideas and a human selects them—as an instance of **Distributed Bayesian Inference**. Specifically, we interpret the iterative proposal-selection loop through the lens of the **Sampling-Importance-Resampling (SIR)** algorithm.

This framework unifies human preferences and LLM knowledge as probability distributions. We conducted three controlled experiments using LLM agents as proxies for human evaluators to investigate the dynamics of this interaction.

This repository includes codes for:

* **Experiment I: Discrete Domain (Toy Model)** 
    Validates the SIR framework in a fixed symbolic space. The task involves identifying a specific target item (playing card) from a closed set to verify knowledge integration dynamics between agents with partial knowledge.
* **Experiment II: Open Domain (Travel Ideation)** 
    Investigates how an LLM's internal bias (Proposer) shapes the decision trajectory in a language-mediated brainstorming task.
* **Experiment III: Continuous Domain (Color Ideation)** 
    Examines the "sharpening dynamics" of the collective decision distribution in a continuous latent space ($LCh$ color space) to test for convergence and satisficing behaviors.

## Environment Setup (Installation)

To run the code in this repository, you will need Python (version 3.8 or higher is recommended) and an OpenAI API key.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Tanichu-Laboratory/collaborative_ideation.git](https://github.com/Tanichu-Laboratory/collaborative_ideation.git)
    cd collaborative_ideation
    ```

2.  **Install required libraries:**
    Install the necessary Python packages listed in `requirements.txt` (e.g., openai, numpy, pandas, scipy, etc.).
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This repository contains scripts for three distinct experiments corresponding to the sections in the paper.

### 1. Experiment I: Discrete Domain (Toy Model)

```bash
# Run the discrete domain simulation
OPENAI_API_KEY="your-key" python experiment1/trial.py
```

### 2. Experiment II: Open Domain (Travel Ideation)

```bash
# Run the open-domain ideation task
OPENAI_API_KEY="your-key" python experiment2/trial_all.py
```

### 3. Experiment III: Continuous Domain (Color Ideation)

```bash
# Run the continuous domain simulation
OPENAI_API_KEY="your-key" python experiment3/trial.py
```

## License

This repository’s code is distributed under the terms of the Apache License 2.0. A copy of the license can be found in the LICENSE file in the root directory of this repository.

## Contact

For any questions regarding this research or codebase, please contact Momoha Hirose at hirose.momoha.68e@st.kyoto-u.ac.jp.
