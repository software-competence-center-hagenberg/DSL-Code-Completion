# DSL-Code-Completion
DSL-Code-Completion

This repository contains the preprocessing and training code for the paper "Using AI-Based Code Completion for Domain Specific Languages". 
We investigated machine learning architectures and adapted two architectures to fit TTI input files. The first one is Pythia. 

Svyatkovskiy, A., Zhao, Y., Fu, S., Sundaresan, N.: Pythia: AI-assisted code completion system. 
In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. p. 2727–2735. KDD ’19, Association
for Computing Machinery, New York, NY, USA (2019)

The training and preprocessing code for the pythia implementation is based on https://github.com/motykatomasz/Pythia-AI-code-completion.
Second, we evaluated Code Completion Neural Attention and Pointer Networks (NAPN).

Li, J., Wang, Y., King, I., Lyu, M.R.: Code completion with neural attention and pointer networks. 
In: Proceedings of the 27th International Joint Conference on
Artificial Intelligence. p. 4159–4165. IJCAI’18, AAAI Press (2018)

The training and preprocessing code for the NAPN implementation is based on https://github.com/oleges1/code-completion.

# Using AI-Based Code Completion for Domain Specific Languages
Code completion is a very important feature of modern integrated development environments. Research has been done for years to improve code completion systems for general-purpose languages. However, only little literature can be found for (AI-based) code completion for domain specific languages (DSLs). 
A DSL is a special-purpose programming language tailored for a specific application domain. In this paper, we investigate whether state-of-the-art code completion approaches can also be applied for DSLs. This is demonstrated using the domain-specific language TTI. TTI is used in transformer construction in an industrial context, where an existing code completion shall be replaced by an advanced machine learning approach. For this purpose, implementations of two code completion systems are adapted to our needs. One of them shows very promising results and achieves a top-5 accuracy of 95 percent. To evaluate the practical applicability, the approach is integrated into the existing editor of a power transformer manufacturer.