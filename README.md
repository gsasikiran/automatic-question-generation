# AUTOMATIC QUESTION GENERATION

## Authors
  - [Sasi Kiran Gaddipati](https://github.com/gsasikiran)
  - [Desiana Dien Nurchalifah](https://github.com/desinurch)

## Overview

  An approach to generate questions and answers from comprehensions, by implementing a similar technique of [Zhou, et al.](https://arxiv.org/pdf/1704.01792.pdf) trained on [Pytorch](https://pytorch.org/).
  
  Project link: https://github.com/gsasikiran/automatic-question-generation
 
 ## Dataset: 
  - Stanford Question Answering Dataset ([SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/))
  
  - Dataset comprises of comprehensions, questions and answers designed for question answering task
  
  - The following excerpt is taken from [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/Southern_California.html) website
  
  ![Excerpt of Dataset from here](/images/squad_dataset.PNG)
  
 
 ## Requirements
  - Language : Python
  - Packages
    - torch : 1.5.0
    - matplotlib :3.2.1
    - pandas : 1.0.5
    - spacy : 2.2.4
    - numpy : 1.18.5
    - torchtext : 0.3.1
    - nltk : 3.5
 
 For more information of code and training check [main/README.md](https://github.com/gsasikiran/automatic-question-generation/tree/master/main)
 
 ## Description
 
 ### Preprocessing Steps
   We implement:
   *  Case normalization    
   * Tokenization    
   * Named entity recognition (NER)    
   * POS-Tagging    
   * IOB-Tagging    
   * Pairing input and output
    
 ### Model Architecture
  
   We input the preprocessed input and output pairs through the Seq2Seq attention architecture. The architecture comprises of encoder and decoder. Encoder is built upon the bi-directional GRU and the decoder consists of stacked GRU architectures. The attention mechanism calculates the context vectors at every step of decoder. Attention is computed by taking the softmax of most related hidden vectors of the  encoder to the current hidden vector at the decoder. The model architecture is shown below.
   
   ![Model architecture](/images/AQG_model.png) 
   
   
 
 
 ## Results
 
 ### Qualitative Results
 ![](/images/totally_crct.PNG) 
 ---------------------------------------------------------------------------
 ![](/images/partially_crct.PNG) 
 ---------------------------------------------------------------------------
 ![](/images/wrong_1.PNG) 
 ---------------------------------------------------------------------------
 ![](/images/wrong_2.PNG)
 ---------------------------------------------------------------------------
 ### Quantitative Results
 
 #### Human Evaluation:
  We created [Google forms](https://forms.gle/TFLSjGU843XSTac39) with random 25 excerpts and questions to vote from 1-3 as follows:
  - 3 : Question is meaningful and relates to the paragraph
  - 2 : Question is more or less meaningful and may relate to the paragraph
  - 1 : Question do not carry any meaning
  
  The results through human evaluation are defined below. The mean score and correspondability between evaluators (Fleiss' Kappa Score) are given below
  - **Mean score:** 1.750 (greater than [1])
  - **Fleiss' Kappa score:** 0.238 (Fair agreement between evaluators)
  
  #### Automatic Evaluation:
  - **Meteor Score:** 0.1819
  - **BLEU Score:** 0.0216
  
  
 
 ## References:
 [1] Zhou, Qingyu, et al. "Neural question generation from text: A preliminary study." National CCF Conference on Natural Language Processing and Chinese Computing. Springer, Cham, 2017.
 
 [2] Fleiss, Joseph L. "Measuring nominal scale agreement among many raters." Psychological bulletin 76.5 (1971): 378.
 
 [3] https://github.com/bentrevett/pytorch-seq2seq

 
 
  
 
  
