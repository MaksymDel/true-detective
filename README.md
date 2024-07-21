## True Detective: A Challenging Benchmark for Deep Abductive Reasoning in Large Language Models

This repository contains code and data for our paper [True Detective: A Challenging Benchmark for Deep Abductive Reasoning in Large Language Models](https://aclanthology.org/2023.starsem-1.28/).
It introduces a challenging (as far as GPT-4 is concerned) benchmark consisting of short stories of detective puzzles with a golden chain of thought traces for each puzzle.

The data is sourced from [5minutemystery](https://www.5minutemystery.com/) platform and can only be used for academic research.

#### Abstract
Large language models (LLMs) have demonstrated solid zero-shot reasoning capabilities, which is reflected in their performance on the current test tasks. This calls for a more challenging benchmark requiring highly advanced reasoning ability to be solved. In this paper, we introduce such a benchmark, consisting of 191 long-form (1200 words on average) mystery narratives constructed as detective puzzles. Puzzles are sourced from the “5 Minute Mystery” platform and include a multiple-choice question for evaluation. Only 47% of humans solve a puzzle successfully on average, while the best human solvers achieve over 80% success rate. We show that GPT-3 models barely outperform random on this benchmark (with 28% accuracy) while state-of-the-art GPT-4 solves only 38% of puzzles. This indicates that there is still a significant gap in the deep reasoning abilities of LLMs and humans and highlights the need for further research in this area. Our work introduces a challenging benchmark for future studies on reasoning in language models and contributes to a better understanding of the limits of LLMs’ abilities.

#### How to cite
```
@inproceedings{del-fishel-2023-true,
    title = "True Detective: A Deep Abductive Reasoning Benchmark Undoable for {GPT}-3 and Challenging for {GPT}-4",
    author = "Del, Maksym  and
      Fishel, Mark",
    editor = "Palmer, Alexis  and
      Camacho-collados, Jose",
    booktitle = "Proceedings of the 12th Joint Conference on Lexical and Computational Semantics (*SEM 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.starsem-1.28",
    doi = "10.18653/v1/2023.starsem-1.28",
    pages = "314--322",
    abstract = "Large language models (LLMs) have demonstrated solid zero-shot reasoning capabilities, which is reflected in their performance on the current test tasks. This calls for a more challenging benchmark requiring highly advanced reasoning ability to be solved. In this paper, we introduce such a benchmark, consisting of 191 long-form (1200 words on average) mystery narratives constructed as detective puzzles. Puzzles are sourced from the {``}5 Minute Mystery{''} platform and include a multiple-choice question for evaluation. Only 47{\%} of humans solve a puzzle successfully on average, while the best human solvers achieve over 80{\%} success rate. We show that GPT-3 models barely outperform random on this benchmark (with 28{\%} accuracy) while state-of-the-art GPT-4 solves only 38{\%} of puzzles. This indicates that there is still a significant gap in the deep reasoning abilities of LLMs and humans and highlights the need for further research in this area. Our work introduces a challenging benchmark for future studies on reasoning in language models and contributes to a better understanding of the limits of LLMs{'} abilities.",
}
```
