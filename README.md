<div align="center">

# Sutskever's List

**The 30 fundamental papers and resources that represent 90% of what matters in modern AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

_"If you really learn all of these, you'll know 90% of what matters today."_  
— Ilya Sutskever (Co-founder of OpenAI, Former Chief Scientist)

[About](#about) • [Reading Roadmap](#reading-roadmap) • [Papers by Topic](#papers-by-topic) • [Getting Started](#getting-started) • [Contributing](#contributing)

</div>

---

## About

In a conversation with **John Carmack** (legendary game developer and AI researcher), **Ilya Sutskever** shared a curated reading list of approximately 30 foundational papers, blog posts, and resources that form the backbone of modern deep learning and artificial intelligence.

This list is exceptional because it:

- **Focuses on fundamentals** that have stood the test of time
- **Covers the full spectrum** from basic neural networks to cutting-edge architectures
- **Includes theory and practice** - mathematical foundations alongside practical implementations
- **Represents real depth** - these aren't just popular papers, they're the ones that truly matter

This repository organizes Sutskever's recommendations into a structured learning path, provides context for each resource, and maintains an active community of learners working through the material together.

### Why This List Matters

Unlike most AI reading lists that grow indefinitely, Sutskever's list is deliberately constrained. The "90% of what matters" philosophy means these papers provide maximum insight with minimal redundancy. Master these, and you'll have a foundation that enables you to:

- Understand and implement modern architectures from scratch
- Read cutting-edge research papers with confidence
- Make informed architectural decisions in your own work
- Grasp the theoretical underpinnings of why things work

---

## Reading Roadmap

This roadmap organizes the list by difficulty and learning objectives. Each path builds on previous knowledge, so follow the progression unless you already have the prerequisites.

### Beginner Path

**Prerequisites:** Basic programming, calculus, and linear algebra  
**Goal:** Understand what neural networks are and how they work  
**Time commitment:** 4-6 weeks

1. **CS231n: Convolutional Neural Networks for Visual Recognition**  
   Stanford's legendary course. Start here for the best introduction to deep learning fundamentals, covering everything from backpropagation to CNNs.  
   [Course Website](https://cs231n.github.io/)

2. **The Unreasonable Effectiveness of RNNs** — _Andrej Karpathy_  
   An intuitive, code-driven introduction to recurrent networks. Shows what RNNs can do before diving into how they work.  
   [Blog Post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

3. **Understanding LSTM Networks** — _Christopher Olah_  
   The clearest visual explanation of LSTMs ever written. Essential for understanding sequence modeling.  
   [Blog Post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

4. **The Annotated Transformer** — _Harvard NLP_  
   A line-by-line implementation of "Attention is All You Need" with full working code. Learn by building.  
   [Blog Post](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

5. **ImageNet Classification with Deep Convolutional Neural Networks** (AlexNet)  
   The 2012 paper that started the deep learning revolution. Understanding this gives context for everything that followed.  
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

### Intermediate Path

**Prerequisites:** Completed Beginner Path or equivalent knowledge  
**Goal:** Master core architectures and attention mechanisms  
**Time commitment:** 6-8 weeks

6. **Deep Residual Learning for Image Recognition** (ResNet)  
   Skip connections revolutionized deep learning by enabling networks with 100+ layers. This is the paper that made modern deep learning possible.  
   [arXiv](https://arxiv.org/abs/1512.03385)

7. **Attention Is All You Need**  
   The transformer architecture that powers GPT, BERT, and most modern language models. One of the most influential ML papers ever written.  
   [arXiv](https://arxiv.org/abs/1706.03762)

8. **Neural Machine Translation by Jointly Learning to Align and Translate**  
   The paper that introduced attention mechanisms. Essential for understanding transformers.  
   [arXiv](https://arxiv.org/abs/1409.0473)

9. **Pointer Networks**  
   How to handle variable-length outputs and combinatorial problems with neural networks.  
   [arXiv](https://arxiv.org/abs/1506.03134)

10. **Recurrent Neural Network Regularization**  
    The right way to apply dropout to RNNs. Short but crucial for practical implementations.  
    [arXiv](https://arxiv.org/abs/1409.2329)

11. **Generating Sequences With Recurrent Neural Networks**  
    Alex Graves' masterpiece on sequence generation with RNNs. Introduces mixture density networks and shows how to generate handwriting and text.  
    [arXiv](https://arxiv.org/abs/1308.0850)

12. **On the difficulty of training Recurrent Neural Networks**  
    Explains the vanishing/exploding gradient problem in RNNs and introduces gradient clipping. Essential for understanding RNN training challenges.  
    [arXiv](https://arxiv.org/abs/1211.5063)

13. **Identity Mappings in Deep Residual Networks**  
    Improves on ResNet by analyzing exactly how skip connections work. Important for architecture design.  
    [arXiv](https://arxiv.org/abs/1603.05027)

14. **Order Matters: Sequence to Sequence for Sets**  
    Handling unordered data with sequence models. Clever solutions to important problems.  
    [arXiv](https://arxiv.org/abs/1511.06391)

15. **Multi-Scale Context Aggregation by Dilated Convolutions**  
    Efficient receptive fields without pooling. Key technique for semantic segmentation and dense prediction.  
    [arXiv](https://arxiv.org/abs/1511.07122)

### Advanced Path

**Prerequisites:** Strong understanding of architectures from Intermediate Path  
**Goal:** Explore novel architectures and advanced training techniques  
**Time commitment:** 8-10 weeks

16. **Neural Turing Machines**  
    Combining neural networks with external memory. Pioneering work in differentiable programming.  
    [arXiv](https://arxiv.org/abs/1410.5401)

17. **A Simple Neural Network Module for Relational Reasoning**  
    How to reason about relationships between objects. Enables compositional understanding.  
    [arXiv](https://arxiv.org/abs/1706.01427)

18. **Relational Recurrent Neural Networks**  
    Enhancing RNNs with relational inductive biases for improved reasoning.  
    [arXiv](https://arxiv.org/abs/1806.01822)

19. **Variational Lossy Autoencoder**  
    Advanced generative modeling with hierarchical latent variables.  
    [arXiv](https://arxiv.org/abs/1611.02731)

20. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**  
    How to train models that don't fit on a single GPU. Essential for understanding modern large-scale training.  
    [arXiv](https://arxiv.org/abs/1811.06965)

21. **Deep Speech 2: End-to-End Speech Recognition in English and Mandarin**  
    End-to-end speech recognition with RNNs. Shows how to apply deep learning to real-world audio problems.  
    [arXiv](https://arxiv.org/abs/1512.02595)

22. **Neural Quantum Chemistry**  
    Deep learning for molecular property prediction. Example of deep learning beyond traditional ML domains.  
    [arXiv](https://arxiv.org/abs/1704.01212)

23. **Machine Learning: The High-Interest Credit Card of Technical Debt**  
    Essential reading on ML systems engineering. Addresses the hidden costs and long-term maintenance challenges of ML systems in production.  
    [PDF](https://research.google/pubs/pub43146/)

### Theory & Foundations

**Prerequisites:** Strong mathematical background, comfort with information theory  
**Goal:** Understand the theoretical underpinnings of why deep learning works  
**Time commitment:** 10-12 weeks (can be done in parallel with other paths)

24. **The First Law of Complexodynamics** — _Scott Aaronson_  
    Theoretical framework for understanding complexity in physical systems. Provides deep insights into learning and compression.  
    [Blog Post](https://scottaaronson.blog/?p=762)

25. **Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton**  
    Formalizing intuitions about complexity. Important for understanding what neural networks can and cannot learn.  
    [arXiv](https://arxiv.org/abs/1405.6903)

26. **Scaling Laws for Neural Language Models**  
    How model performance scales with compute, data, and parameters. Essential for understanding modern AI capabilities and limitations.  
    [arXiv](https://arxiv.org/abs/2001.08361)

27. **A Tutorial Introduction to the Minimum Description Length Principle**  
    Compression and learning are two sides of the same coin. Fundamental principle underlying many ML algorithms.  
    [arXiv](https://arxiv.org/abs/math/0406077)

28. **Keeping Neural Networks Simple by Minimizing the Description Length of the Weights**  
    Applying MDL to neural networks. Occam's razor formalized.  
    [PDF](https://www.cs.toronto.edu/~fritz/absps/colt93.pdf)

### Doctoral Level

**Prerequisites:** All of the above, plus serious dedication  
**Goal:** Achieve deep theoretical understanding  
**Time commitment:** Several months

29. **Machine Super Intelligence** — _Shane Legg_  
    PhD thesis on formal theories of intelligence and artificial general intelligence. Dense but rewarding.  
    [PDF](http://www.vetta.org/documents/Machine_Super_Intelligence.pdf)

30. **An Introduction to Kolmogorov Complexity and Its Applications** (Chapter 7, page 434 onwards)  
    The mathematical foundations of information, randomness, and learning. Challenging but fundamental.  
    [PDF](https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf)

---

## Papers by Topic

### Foundational Architectures

| Paper                                                                                                                                               | Authors                       | Year | Key Innovation                               | Why It Matters                                                              |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---- | -------------------------------------------- | --------------------------------------------------------------------------- |
| [ImageNet Classification with Deep CNNs](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton | 2012 | Deep CNNs + GPU training + ReLU + Dropout    | Proved deep learning works at scale, sparked the modern AI revolution       |
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385)                                                                                          | He et al.                     | 2015 | Skip connections enabling very deep networks | Made 100+ layer networks trainable, foundation of most modern architectures |
| [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)                                                                     | He et al.                     | 2016 | Analysis and improvement of ResNet design    | Clarified why skip connections work so well                                 |

### Recurrent Networks & Sequential Processing

| Paper                                                                                              | Authors        | Year | Key Innovation                                      | Why It Matters                                      |
| -------------------------------------------------------------------------------------------------- | -------------- | ---- | --------------------------------------------------- | --------------------------------------------------- |
| [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Karpathy       | 2015 | Accessible introduction to RNN capabilities         | Best first introduction to sequence modeling        |
| [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)          | Olah           | 2015 | Clear visual explanation of LSTM architecture       | The definitive LSTM explainer                       |
| [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)                         | Zaremba et al. | 2014 | Proper dropout for RNNs                             | Essential technique for training RNNs               |
| [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)             | Graves         | 2013 | Sequence generation with mixture density networks   | Masterpiece on generating handwriting and text      |
| [On the difficulty of training RNNs](https://arxiv.org/abs/1211.5063)                              | Pascanu et al. | 2012 | Vanishing/exploding gradients and gradient clipping | Essential for understanding RNN training challenges |
| [Deep Speech 2](https://arxiv.org/abs/1512.02595)                                                  | Amodei et al.  | 2015 | End-to-end speech recognition                       | Shows how to apply deep learning to real audio      |

### Attention & Transformers

| Paper                                                                                                    | Authors         | Year | Key Innovation                  | Why It Matters                                           |
| -------------------------------------------------------------------------------------------------------- | --------------- | ---- | ------------------------------- | -------------------------------------------------------- |
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | Bahdanau et al. | 2014 | Attention mechanism for seq2seq | Introduced the attention mechanism used everywhere today |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                            | Vaswani et al.  | 2017 | Transformer architecture        | Powers GPT, BERT, and nearly all modern NLP              |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)                      | Rush et al.     | 2018 | Line-by-line implementation     | Learn transformers by building one                       |

### Novel Architectures & Mechanisms

| Paper                                                                                       | Authors        | Year | Key Innovation                                         | Why It Matters                                         |
| ------------------------------------------------------------------------------------------- | -------------- | ---- | ------------------------------------------------------ | ------------------------------------------------------ |
| [Pointer Networks](https://arxiv.org/abs/1506.03134)                                        | Vinyals et al. | 2015 | Output sequences of discrete tokens pointing to inputs | Handles combinatorial optimization problems            |
| [Neural Turing Machines](https://arxiv.org/abs/1410.5401)                                   | Graves et al.  | 2014 | Differentiable external memory                         | Pioneered neural network + data structure combinations |
| [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391)            | Vinyals et al. | 2015 | Read/process phase for set inputs                      | Elegant solution to permutation invariance             |
| [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) | Yu & Koltun    | 2016 | Dilated/atrous convolutions                            | Exponentially growing receptive fields                 |
| [A Simple Neural Network Module for Relational Reasoning](https://arxiv.org/abs/1706.01427) | Santoro et al. | 2017 | Relation networks                                      | First-class support for reasoning about relationships  |
| [Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822)                    | Santoro et al. | 2018 | Multi-head dot-product attention in RNNs               | Combines recurrence with relational reasoning          |

### Generative Models

| Paper                                                             | Authors     | Year | Key Innovation                          | Why It Matters                          |
| ----------------------------------------------------------------- | ----------- | ---- | --------------------------------------- | --------------------------------------- |
| [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) | Chen et al. | 2016 | Hierarchical VAE with learned inference | Advanced generative modeling techniques |

### Applications & ML Systems

| Paper                                                                                         | Authors        | Year | Domain     | Why It Matters                                           |
| --------------------------------------------------------------------------------------------- | -------------- | ---- | ---------- | -------------------------------------------------------- |
| [Neural Quantum Chemistry](https://arxiv.org/abs/1704.01212)                                  | Gilmer et al.  | 2017 | Chemistry  | Shows deep learning's reach beyond traditional ML        |
| [ML: The High-Interest Credit Card of Technical Debt](https://research.google/pubs/pub43146/) | Sculley et al. | 2015 | ML Systems | Hidden costs and maintenance challenges of production ML |

### Training & Scaling

| Paper                                                                       | Authors       | Year | Key Innovation                           | Why It Matters                                          |
| --------------------------------------------------------------------------- | ------------- | ---- | ---------------------------------------- | ------------------------------------------------------- |
| [GPipe](https://arxiv.org/abs/1811.06965)                                   | Huang et al.  | 2018 | Pipeline parallelism for large models    | Essential for training models that don't fit on one GPU |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al. | 2020 | Power law relationships in model scaling | Predicts performance based on compute/data/parameters   |

### Information Theory & Foundations

| Paper                                                                                                                 | Authors           | Year | Key Concept                                | Why It Matters                                     |
| --------------------------------------------------------------------------------------------------------------------- | ----------------- | ---- | ------------------------------------------ | -------------------------------------------------- |
| [Keeping Neural Networks Simple by Minimizing Description Length](https://www.cs.toronto.edu/~fritz/absps/colt93.pdf) | Hinton & Van Camp | 1993 | MDL principle for neural networks          | Foundational principle: learning is compression    |
| [A Tutorial Introduction to the MDL Principle](https://arxiv.org/abs/math/0406077)                                    | Grünwald          | 2004 | Minimum Description Length                 | Unifies learning and compression theoretically     |
| [Quantifying Complexity in Closed Systems](https://arxiv.org/abs/1306.6730)                                           | Aaronson et al.   | 2013 | Complexity measures in physical systems    | Theoretical foundations for understanding learning |
| [The First Law of Complexodynamics](https://scottaaronson.blog/?p=762)                                                | Aaronson          | 2013 | Entropy and complexity in physical systems | Deep connections between physics and computation   |

### Books & Theses

| Resource                                                                                    | Author       | Year | Topic                  | Why It Matters                                           |
| ------------------------------------------------------------------------------------------- | ------------ | ---- | ---------------------- | -------------------------------------------------------- |
| [Machine Super Intelligence](http://www.vetta.org/documents/Machine_Super_Intelligence.pdf) | Shane Legg   | 2008 | Formal theories of AGI | Rigorous treatment of intelligence from first principles |
| [Kolmogorov Complexity](https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf) (p.434+)         | Li & Vitányi | 2013 | Information theory     | Mathematical foundations of information and randomness   |

### Courses

| Resource                                                         | Institution | Topic                      | Why It Matters                                 |
| ---------------------------------------------------------------- | ----------- | -------------------------- | ---------------------------------------------- |
| [CS231n: CNNs for Visual Recognition](https://cs231n.github.io/) | Stanford    | Deep learning fundamentals | The best introduction to deep learning, period |

---

## Getting Started

### For Self-Learners

**Start with the fundamentals:** If you're new to deep learning, begin with CS231n and work through the Beginner Path sequentially. Don't skip ahead—each paper builds on previous concepts.

**Implement as you learn:** The best way to understand these papers is to implement them. Start with The Annotated Transformer as a template for how to read a paper while coding.

**Take notes:** Maintain a learning journal where you summarize each paper in your own words. This active processing dramatically improves retention.

**Join the community:** Use GitHub Discussions to ask questions, share insights, and find study partners.

### For Study Groups

**Structure:** Meet weekly, with each person responsible for presenting one paper. Budget 2-3 hours per paper for reading and preparation.

**Discussion format:**

- 10 min: Context and motivation
- 20 min: Main technical content
- 15 min: Implementation details
- 15 min: Discussion and connections to other papers

**Shared implementation:** As a group, implement 3-5 of the most important papers from scratch. This shared codebase becomes a learning resource for everyone.

### For Researchers

**Use as a foundation:** If you're entering a new research area, this list provides the essential background. Use the Theory section to ground your intuitions.

**Connect to your work:** As you read each paper, explicitly note connections to your research interests. This builds a mental map of the field.

**Contribute back:** Once you've mastered the list, consider contributing summaries, implementations, or teaching others.

---

## Implementation Resources

### Code Repositories

Several papers have excellent reference implementations:

- **The Annotated Transformer**: Complete working implementation in PyTorch
- **CS231n Assignments**: Hands-on coding exercises for CNNs, RNNs, and transformers
- **AlexNet**: Available in all major frameworks
- **ResNet**: Official implementation and countless tutorials

### Recommended Tools

- **PyTorch**: Best for research and learning (dynamic computation graphs)
- **JAX**: Excellent for understanding autograd and transformations
- **NumPy**: Implement from scratch for deepest understanding

### Learning Projects

After mastering the core papers, build these projects to solidify understanding:

1. **Character-level language model** (RNNs + LSTMs)
2. **Image classifier from scratch** (CNNs + ResNets)
3. **Machine translation system** (Attention + Transformers)
4. **Neural Turing Machine** (External memory)

---

## Contributing

We welcome contributions from the community! This repository improves when people share their insights and implementations.

### How to Contribute

**Paper summaries**: Write clear, concise summaries (300-500 words) explaining the key ideas in accessible language.

**Code implementations**: Share clean, well-documented implementations with links to your repos.

**Learning notes**: Contribute your notes, diagrams, or explanations that helped you understand difficult concepts.

**Fix issues**: Report broken links, errors in descriptions, or suggest improvements.

**Translations**: Help make this resource accessible to non-English speakers.

### Contribution Guidelines

- Keep summaries technical but accessible
- Include code that actually runs
- Cite sources properly
- Follow the existing format
- Be respectful and constructive in discussions

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Progress Tracking

Create your own checklist by copying this into a GitHub issue or personal document:

```
## Beginner Path
- [ ] CS231n Course
- [ ] The Unreasonable Effectiveness of RNNs
- [ ] Understanding LSTM Networks
- [ ] The Annotated Transformer
- [ ] ImageNet Classification with Deep CNNs

## Intermediate Path
- [ ] Deep Residual Learning (ResNet)
- [ ] Attention Is All You Need
- [ ] Neural Machine Translation (Attention)
- [ ] Pointer Networks
- [ ] RNN Regularization
- [ ] Generating Sequences With RNNs
- [ ] On the difficulty of training RNNs
- [ ] Identity Mappings in ResNets
- [ ] Order Matters (Sets)
- [ ] Dilated Convolutions

## Advanced Path
- [ ] Neural Turing Machines
- [ ] Relational Reasoning Module
- [ ] Relational RNNs
- [ ] Variational Lossy Autoencoder
- [ ] GPipe
- [ ] Deep Speech 2
- [ ] Neural Quantum Chemistry
- [ ] ML: The High-Interest Credit Card of Technical Debt

## Theory & Foundations
- [ ] First Law of Complexodynamics
- [ ] Coffee Automaton (Complexity)
- [ ] Scaling Laws for LLMs
- [ ] MDL Tutorial
- [ ] MDL for Neural Networks

## Doctoral Level
- [ ] Machine Super Intelligence
- [ ] Kolmogorov Complexity (Chapter 7)
```

---

## Community

### Discussions

Share insights, ask questions, and learn together in [GitHub Discussions](../../discussions).

**Popular topics:**

- Paper reading groups
- Implementation help
- Conceptual questions
- Research connections

### Study Groups

Find or start a study group:

- Post in [Issues](../../issues) with tag `study-group`
- Include your timezone, pace, and preferred communication platform
- Many groups meet weekly via video call

### Social

Share your progress and connect with others:

- Tag your posts with `#SutskeversListChallenge`
- Share implementations and learning notes
- Help others who are earlier in their journey

---

## FAQ

**Q: Do I need to read every paper cover-to-cover?**  
A: No. Read for understanding, not completion. Some papers you'll skim, others you'll implement. Let your learning goals guide you.

**Q: What math background do I need?**  
A: Calculus, linear algebra, probability. CS231n has good review materials if you're rusty.

**Q: How long does it take to complete?**  
A: 6-12 months working steadily. The goal isn't speed—it's deep understanding.

**Q: Can I skip the theory papers?**  
A: You can, but you shouldn't. They provide crucial intuitions for why techniques work.

**Q: Are there video lectures?**  
A: CS231n has excellent videos. For most papers, you'll need to read the actual paper.

---

## License

This curated list is available under the [MIT License](LICENSE).

Individual papers and resources are subject to their respective licenses and copyrights.

---

## Acknowledgments

- **Ilya Sutskever** for curating this essential reading list and sharing it with the community
- **John Carmack** for the conversation that brought this list to public attention
- **All the researchers** who authored these foundational papers
- **The contributors** who help maintain and improve this repository
- **The AI/ML community** for building on these foundations

---

<div align="center">

**[⬆ Back to top](#sutskever-list)**

</div>
