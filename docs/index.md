## Description
Sequence labeling is a task that assigns categorial label to each element in an input sequence. Many problems can be formalized as sequence labeling task, including speech recognition, video analysis and various problems in NLP (e.g. POS tagging, NER, Chunking, etc.). Traditionally sequence labeling requires large amount of hand-engineered features and domain-specific knowledge, but recently neural approaches have achieved state-of-the-art performance on several sequence labeling benchmarks. A common data format for sequence labeling task is IOB (Inside-Outside-Beginning), although other alternative formats (e.g. IO, IOBES, BMEWO, BMEWO+, BILOU, etc.) might be used.
![ner_iob_example]({{ site.url }}/sequence_labeling_tf/ner.iob.example.png){:width="800px"}

*Figure 1: An NER example in IOB format*

## DataSet
* [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/) is a multi-task dataset, which contains 3 sub-tasks, POS tagging, syntactic chunking and NER. For NER sub-task, it contains 4 types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
* [OntoNotes5](https://catalog.ldc.upenn.edu/LDC2013T19) is a multi-task dataset, which contains several sub-tasks, including POS tagging, word sense disambiguation, coreference, NER and others. For NER sub-task, it contains 18 types of named entities: PERSON, LOC, ORG, DATE, MONEY and others. This dataset can be converted into CoNLL format using [common tool](http://conll.cemantix.org/2012/data.html).
* [Treebank3](https://catalog.ldc.upenn.edu/LDC99T42) is a distributed release of Penn Treebank (PTB) project, which selected 2,499 stories from a three year Wall Street Journal (WSJ) collection of 98,732 stories for syntactic annotation, including POS tagging and constituency parsing.
* [GloVe](https://nlp.stanford.edu/projects/glove/) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Experiment
### Bi-LSTM + Char-CNN + Softmax
![bilstm_charcnn_softmax_arch]({{ site.url }}/sequence_labeling_tf/BiLSTM_CharCNN_Softmax.architecture.png){:width="500px"}

*Figure 2: Bi-LSTM + Char-CNN + Softmax architecture*

|    CoNLL2003 - NER  |    F1 Score   |   Precision   |     Recall    |
|:-------------------:|:-------------:|:-------------:|:-------------:|
|         Dev         |     94.92     |     94.97     |     94.87     |
|        Test         |     91.29     |     90.41     |     92.18     |

*Table 1: The performance of Bi-LSTM + Char-CNN + Softmax on CoNLL2003 NER sub-task with setting: num layers = 2, unit dim = 200, window size = [3]*

|   OntoNotes5 - NER  |    F1 Score   |   Precision   |     Recall    |
|:-------------------:|:-------------:|:-------------:|:-------------:|
|         Dev         |     86.22     |     84.21     |     88.32     |
|        Test         |     85.09     |     82.66     |     87.67     |

*Table 2: The performance of Bi-LSTM + Char-CNN + Softmax on OntoNotes5 NER sub-task with setting: num layers = 2, unit dim = 200, window size = [3,5]*

|   Treebank3 - POS   |    Accuracy   |
|:-------------------:|:-------------:|
|         Dev         |     97.36     |
|        Test         |     97.58     |

*Table 3: The performance of Bi-LSTM + Char-CNN + Softmax on Treebank3 POS tagging sub-task with setting: num layers = 2, unit dim = 200, window size = [3]*

### Bi-LSTM + Char-CNN + CRF
![bilstm_charcnn_crf_arch]({{ site.url }}/sequence_labeling_tf/BiLSTM_CharCNN_CRF.architecture.png){:width="500px"}

*Figure 3: Bi-LSTM + Char-CNN + CRF architecture*

|    CoNLL2003 - NER  |    F1 Score   |   Precision   |     Recall    |
|:-------------------:|:-------------:|:-------------:|:-------------:|
|         Dev         |     94.93     |     94.92     |     94.93     |
|        Test         |     91.30     |     90.47     |     92.15     |

*Table 4: The performance of Bi-LSTM + Char-CNN + CRF on CoNLL2003 NER sub-task with setting: num layers = 2, unit dim = 200, window size = [3]*

|   OntoNotes5 - NER  |    F1 Score   |   Precision   |     Recall    |
|:-------------------:|:-------------:|:-------------:|:-------------:|
|         Dev         |     86.45     |     84.11     |     88.93     |
|        Test         |     85.25     |     82.57     |     88.11     |

*Table 5: The performance of Bi-LSTM + Char-CNN + CRF on OntoNotes5 NER sub-task with setting: num layers = 2, unit dim = 200, window size = [3,5]*

|   Treebank3 - POS   |    Accuracy   |
|:-------------------:|:-------------:|
|         Dev         |     97.27     |
|        Test         |     97.51     |

*Table 6: The performance of Bi-LSTM + Char-CNN + CRF on Treebank3 POS tagging sub-task with setting: num layers = 2, unit dim = 200, window size = [3]*

## Reference
* Zhiheng Huang, Wei Xu, and Kai Yu. [Bidirectional LSTM-CRF models for sequence tagging](https://arxiv.org/abs/1508.01991) [2015]
* Jason PC Chiu and Eric Nichols. [Named entity recognition with bidirectional lstm-cnns](https://arxiv.org/abs/1511.08308) [2015]
* Xuezhe Ma and Eduard Hovy. [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs CRF](https://arxiv.org/abs/1603.01354) [2016]
* Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, KazuyaKawakami, and ChrisDyer. [Neural architectures for named entity recognition](https://arxiv.org/abs/1603.01360) [2016]
