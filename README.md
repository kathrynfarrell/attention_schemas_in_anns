## Attention Schemas in Artificial Neural Networks

Code accompanying the paper "Improving How Agents Cooperate: Attention Schemas in Artificial Neural Networks." 

Growing evidence suggests that the brain uses an “attention schema” to monitor, predict, and help control attention. It has also been suggested that an attention schema improves social intelligence by allowing one person to better predict another. Given their potential advantages, attention schemas have been increasingly tested in machine learning. Here we test small deep learning networks to determine how the addition of an attention schema may affect performance on a range of tasks. First, we found that an agent with an attention schema is better at judging or categorizing the attention states of other agents. Second, we found that an agent with an attention schema develops a pattern of attention that is easier for other agents to judge and categorize. Third, we found that in a joint task where two agents paint a scene together and must predict each other’s behavior for best performance, adding an attention schema improves that performance. Finally, we find that the performance improvements caused by an attention schema are not a non-specific result of an increase in network complexity. Not all performance, on all tasks, is improved. Instead, improvement is specific to “social” tasks involving judging, categorizing, or predicting the attention of other agents. These results suggest that an attention schema may be useful in machine learning for improving cooperativity and social behavior.

Experiments 1, 2, and 3 can be run with the following notebooks:
* `attentiontask.ipynb`: the attention tensor discrimination task in Experiment 1
* `imageclassification.ipynb`: the general image classification tasks (A, B, and C) in Experiment 1, and the transfer learning task in Experiment 2
* `jointcoloring.ipynb`: the cooperative MARL task ("coloring") in Experiment 3

The full dataset **(5.26 GB)** containing all training and validation images and attention tensors is available [here](https://drive.google.com/file/d/1wczGhJcCfDbC3Nf6vlzCpdiZMA22LR9Q/view?usp=sharing). 

CSV files containing experimental results, as well as the code used to generate figures and statistics, can be found in the `figures` directory.
