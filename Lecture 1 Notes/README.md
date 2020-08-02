# Speech Recognition

* [Overview](#Overview)
    * [Different Variations of Tokens](#Different-Variations-of-Tokens)
        * [Phoneme](#Phoneme)
        * [Grapheme](#Grapheme)
        * [Word](#Word)
        * [Morpheme](#Morpheme)
        * [Byte](#Bytes)
        * [Percentage of usage of different type of tokens](#Percentage-of-usage-of-different-type-of-tokens)
    * [Acoustic Feature](#Acoustic-Feature)
        * [How to obtain acoustic feature](#How-to-obtain-acoustic-feature)
        * [Percentage of usage of different type of acoustic feature](#Percentage-of-usage-of-different-type-of-acoustic-feature)
    * [Amount of Data](#Amount-of-Data)

* [Speech Recognition Models](#Speech-Recognition-Models)


# Overview

* Speech recognition converts **speech** to **text**
* Speech can be represented using sequence of vectors with length **T** and number of dimension **D**
* Text can be represented using sequence of **tokens** with length **N**
* **V** denotes the number of different tokens, it is a fixed number
* Usually, **T** > **N**

## Different Variations of Tokens

### Phoneme
* A unit of sound 
* For example: The phonemes which make up <code>One Punch Man</code>
> * <code>W AH N P AH N CH M AE N</code>
* Requires **lexicon** which is basically a dictionary of words to phonemes
* Prior to Deep Learning era, phoneme is a widely used choice of token because it is directly related to sound signal
* Disadvantage: requires experts such as linguists to build the lexicon for every words

### Grapheme
* Smallest unit of a writing system
* For English, smallest units of writing system are 26 English alphabets
* Needs to add a symbol **_** which represents blank spacing between different words
* For example:
>*  one_punch_man
* It has **N** = 13, **V** = 26 and more
* In addition, punctuation marks such as <code> , </code>
* For writing system such as Chinese, the grapheme are the Chinese characters
* For example, One Punch Man consists of 4 characters
>* 一 , 拳 , 超 , 人
* On the other hand, grapheme for Chinese does not require spacing
* Advantages: 
    * Lexicon free. Does not require linguists to build the lexicons
    * The model may learn to construct words that do not exist in the training set
* Disadvantage:
    * Does not have clear or direct relationship with sound
    * For example, the first phoneme for words *cat* and *Kate* is  **/kæt/**, whereas, the first grapheme are *c* and *k* respectively
    * Huge challenge for the model to learn the complicated relationship between sound and graphemes

### Word
* For example:
    * One Punch Man
    * **N** = 3 because it consists of 3 words
    * usually **V** > 100K
    * Names for people, places are also words
* For some language, using words as token is difficult
* For example, Chinese words are formed by stacking characters together side by side
* Chinese does not use spacing to differentiate between different words
    * 一拳，超人 are two words in Chinese
* For some languages, using words as token is not suitable
* This is because **V** is too large
* For example, agglutinative languages such as Turkish
    
### Morpheme
* The smallest meaningful unit
* Greater than grapheme, but smaller than word
* For example, the English word *unbreakable* can be broken down into 3 morphemes: *un*, *break* and *able*

### Bytes
* Symbols for different language can be represented using UTF-8
* Texts can be represented using sequence of bytes
* The system can be language independent
* **V** is always 256
* Paper [Li, et al., ICASSP 2019]

### Percentage of usage of different type of tokens

| Token Type | Percentage |
|------------|------------|
| Phoneme    | 32%        |
| Grapheme   | 41%        |
| Word       | 10%        |
| Morpheme   | 17%        |


* After going through more than 100 papers from ICASSP 2019, INTERSPEECH 2019, ASRU 2019
* Grapheme is the most widely used choice because it doesn't require lexicons
* Phoneme is the second most widely used choice because it is directly related to sound
* The least used is word

Others 

|              Input              |                           Output                           |
|:-------------------------------:|:----------------------------------------------------------:|
| Speech                          | Word embeddings                                            |
| Speech                          | Speech recognition + Translation                           |
| Speech :<br>One ticket to Taipei  | Speech recognition + Intent classification<br><buy_ticket> |
| Speech : <br>One ticket to Taipei | Speech recognition + Slot filling<br>NA NA NA LOC    |

## Acoustic Feature

<img src="images/i1.png" width="400"/>

* Use a sliding window of size 25ms, with step size 10ms to segment the speech signal
* A frame consists of 400 sample points if the sampling frequency is 16KHz
    > 16Khz x 25ms = 400
* Represent the 25ms speech signal with a vector:
    * Use 400 sample points (400-dim vector)
    * 39-dim MFCC
    * 80-dim filter bank output
* MFCC is widely used before deep learning era
* Filter banks are becoming popular
* After obtaining vector, move the sliding window by 10ms, take another 25ms frame and represent it using vector
* 1 second of speech signal results in 100 frames
* There is a overlap between neigboring frames
* Due to the overlap, some of the values of a vector is repeated in vector of neighboring frame
* Can improve the model architecture and reduce cost by targetting this characteristic

### How to obtain acoustic feature

<img src="images/i2.png" width="400"/>

* The first step is to obtain spectogram from speech waveform using Fourier Transform
* Waveform is too complicated to be used directly as input for Speech Recognition
* This is because the same sound can result in very different looking waveforms
* On the other hand, it is possible for human to guess what was spoke by looking at the spectrogram
* The spectrogram has a more direct relationship with sound compared to waveform
* After that, the spectrogram is passed through filter banks
* Usually, the output of filter banks are **log**ed
* Then after doing DCT, we will get **MFCC**

### Percentage of usage of different type of acoustic feature

| Type of acoustic feature | Percentage |
|:------------------------:|:----------:|
| filter bank output       |        75% |
| MFCC                     |        18% |
| Spectrogram              |         2% |
| Waveform                 |         4% |
| Other                    |         1% |

* After going through more than 100 papers from ICASSP 2019, INTERSPEECH 2019, ASRU 2019
* Prior to deep learning era, **MFCC** is the most widely used acoustic feature
* Filter bank output has overtaken MFCC as choice for acoustic feature

### Amount of Data 

* Requires speech data labelled with texts
* **TIMIT** is the "MNIST" of Speech Recognition

| Data                                               | Length in time |
|----------------------------------------------------|----------------|
| MNIST                                              | "49 min"       |
| CIFAR-10                                           | "2 hr 40 min"  |
| TIMIT                                              | 4 hr           |
| WSJ                                                | 80 hr          |
| Switchboard                                        | 300 hr         |
| Librispeech                                        | 960 hr         |
| Fisher                                             | 2000 hr        |
| ISLVRC (ImageNet)                                  | "4096 hr"      |
| Google Voice Search<br>[Chiu, et al., ICASSP,2018] | 12000 hr ++    |
| FB Video<br>[Huang, et al., arXiv 2019]            | 13000 hr ++    |

* Treat MNIST as if it is speech data
    * 28 x 28 x 1 x 60000 / (16000 x 60) = 49 min
* CIFAR-10:
    * 32 x 32 x 3 x 50000 / (16000 ) = 2 hr 40 min
* Both MNIST and CIFAR-10 are smaller than TIMIT
* Research papers show that Google Voice Search uses 12000 hr plus of data
* However, commercial products use more than that, probably about 10 or 20 times more dataset

* Two points of views for Speech Recognition: 
    1. Seq-toSeq model
    2. Hidden Markov Model (HMM)

# Speech Recognition Models 
* Listen, Attend, and Spell [Chorowski, et al.,NIPS'15]
* Connectionist Temporal Classification (CTC) [Graves, et al., ICML'06]
* RNN Transducer (RNN-T) [Graves, ICML workshop'12]
* Neural Transducer [Jaitly, et al., NIPS'16]
* Monotonic Chunkwise Attention (MoCha) [Chiu, et al., ICLR'18]

| Model      | Percentage |
|------------|------------|
| LAS        | 40%        |
| CTC        | 24%        |
| LAS + CTC  | 11%        |
| RNN-T      | 10%        |
| HMM-hydrid | 15%        |

* LAS is a Seq2Seq model, still widely used
* Neural Transducer and MoCha are relatively lastest techniques
* HMM hybrid combines HMM and deep learning technique
* Nowadays, almost no people do Speech Recognition without using deep learning 

# Listen, Attend, and Spell (LAS)
* LAS is a typical Seq2Seq model with attention
* Listen refers to the encoder part 
* Spell refers to the decoder part
* Attend refers to the attention part
* Why is it called LAS, not Seq2Seq model ?
    * There was a time when it's popular to use three verbs as the title of paper, now it's outdated
    * Other Speech Recognition models are also Seq2Seq model
    * People will not know which model you refer to if you called it Seq2Seq model

## Listen: Encoder Choice
* Listen refers to the encoder part

<img src="images/i3.png" width="400"/>

* Encoder is a Seq2Seq model which takes the sequence of acoustic features as input and outputs high level representations of the acoustic feature sequence
* **x** denotes the input sequence of acoustic features
* **h** denotes the output sequence of vectors
* We hope that the encoder extracts content information and remove speaker variance and noises
* Speaker variance means variations in acoustic features due to different speakers
* The choice of encoder:
    * RNN
    * 1D CNNs

<img src="images/i4.png" width="400"/>

* Using 1D Convolutional Neural Network (CNN), the triangle represents the filter
* The filter performs convolution on acoustic features which is in window
* Then the filter is slided rightwards, another convolution on another window of acoustic features
* A value is contributed at each time step when moving the filter right 
across time steps

<img src="images/i5.png" width="400"/>

* There are more than one filters (different colors)
* Each filter contributes to a value at each time step 
* Filters at higher level can consider longer sequence of acoustic features
* It is common to combine CNN and RNN

## Listen: Down Sampling
* Down Sampling is important in Speech Recognition
* For acoustic feature, 1 seconds of speech results in 100 frames or vectors
* This results in very long sequence
* Self-attention for each time step consider entire sequence in
* This is okay for Machine Translation, but not okay for Speech Recognition because the sequence is too long
* Neighboring vectors share info due to overlap of sliding window
* Down Sampling increases efficiency and reduce computational cost
* Very important, the author mentioned unsuccesful to train LAS without down sampling

**Down Sampling Methods :**

<img src="images/i6.png" width="400"/>

* Pyramid RNN: Each time step of higher layer takes several output of time steps of lower layer as input
* Pooling over time: Each time step of higher layer skips some time step of lower layer

<img src="images/i7.png" width="400"/>

* Time-delay DNN (TDNN) : Similar to 1D CNN, but only consider first and final frame 
* Truncated Self-attention : 
    * Apply attention only to part of the sequence in the range
    * Usually, self-attention will look at whole sequence
    * For example, <code>x3</code> to <code>h3</code> will give attention to <code>x1</code> to <code>x4</code>
    * Truncate the range, now it only looks at <code>x2</code> to <code>x4</code>
    * The range is a hyperparameter

## Attention

<img src="images/i8.png" width="300"/>

* Encoder transforms the input sequence into a sequence of vectors **h**
* There is a vector denoted by **z0**
* Self-attention can be seen as doing a Google Search
* Where **z0** is the keyword and sequence h1, h2, h3, h4 is the database
* There is a **match** function which takes h1 and z0 as input and outputs a scalar <code>alpha_0_1</code>
* Match function can be seen as calculating the similarity between the vectors h1 and z0
* There are two types of match function

**Dot Product Attention**

<img src="images/i10.png" width="200"/>

* Do linear transform to both **h** and **z**
* Results in 2 vectors with same dimensions
* Then do dot product, which results in a scalar **α**

**Additive Attention**

<img src="images/i11.png" width="200"/>

* Perform linear transform to both **h** and **z**
* Add both transformed vectors
* Apply **tanh** resulting in a vector
* Multiply with **W** results in a scalar **α**

<img src="images/i12.PNG" width="400"/>

* Then the <code>alpha</code> are passed through a softmax layer
* The output of the softmax layer sums to one 
* <code>alpha</code> can be seen as a measure of similarity between <code>z</code> and <code>h</code>
* The context vector <code>c_0</code> is a weighted average of the **h** vectors at each time step, weighted by the softmax's output
* Only those **h_t** with lots of similarity with **z0**  contribute to the context vector
* Note that describing self-attention with analogy of google search and similarity is just for understanding
* The context vector is the input for the RNN decoder

## Spell

<img src="images/i13.PNG" width="400"/>

* The decoder is a RNN
* At the first time step of decoder, **z0** and **c0** are the input, the hidden state output is **z1**
* **z1** then undergoes some transform to output distribition over all tokens
* The distribution is a vector of size **V**
* Assigning probabilities to each of the **V** possible tokens
* As shown in the above figure, at the first time step, the decoder assigns 0.0 to a, 0.1 to b and so on
* There are two ways to select a token based on probabilities
* Selecting the token which maximizes the distribution at each time step is called greedy search approach
* Using greedy approach, the token to be selected at the first time step is *c*
* Another method is called beam search which is usually used, will be discussed later

<img src="images/i15.PNG" width="400"/>

* The output of the hidden state at first time step is **z1**, this vector is used to compute **c1** 
* At the second time step, calculate the distribution vector with **z0**,**c1** and the token *c* as input
* The token to be selected at second time step is *a*

<img src="images/i16.PNG" width="400"/>

* Repeat the step, until <EOS> is selected. <EOS> is a token which symbolizes **end of sentence**

### Beam Search

<img src="images/i17.PNG" width="400"/>

* To illustrate beam search, assume there are two tokens **A** and **B** only
* The way to look at the tree in the figure is from the bottommost
* At the first time step, there are two choices, the values at the edge is the output of the decoder, which can be interpreted as assigned probability for each token
* In practice, we will log the probability. This is because the values might be too small if **V** is large
* If **A** is selected at the first time step, this leads to another two possible choices at the second time step
* Similarly, if we choose any of the token at any time step
* The red path is when we use greedy approach, selecting the token with maximum probability at each time step
* Using greedy approach during decoding does not promise or help to get output sequence with the highest probabilty
* For green path, the first time step is B with 0.4, the second time step is 0.9 and the third step 0.9
* Selecting lower probability choice currently, but in the long run may result in better path
* Beam search keeps **B** best pathes at each time step
* If **B** = 2, 2 paths with the highest probability multiplied together will be kept at each time step
* This is better than greedy aprroach in decoding
* **B** is a hyperparameter to be set

## Training

<img src="images/i18.PNG" width="200"/>

* During training, one-hot vector is used as target 
* The prediction from previous time step is not used as input in current time step
* Instead, the ground truth for previous time step is used as input for current time step
* The goal is to have the output vector to be as close as possible to the one-hot vector
* In other words, minimize cross entropy or p(a) to be as large as possible
* This is called **teacher forcing**

Why teacher forcing ?
* Let say the model is supposed to output *cat*
* Initially, the model outputs 'x' instead of 'c' because the model is very bad at the beginning of training
* If we used'x' as input to second time step, the model might learn to generate 'a' when given 'x', not generating 'a' when given 'c'
* In other words, the model will be learning something wrong, this is wasting training time

## Attention

<img src="images/i19.PNG" width="400"/>

* Two types of attention
* Bahdanau: The calculated context vector is used as input of the next time step
* Luong: The context vector is used in current time step as input

<img src="images/i20.PNG" width="200"/> 

* Chan: Use both of the attention together
* The author of LAS thinks that attention has too much flexibility
* At each time step, attention looks at entire sequence
* This might be necessary for Machine Translation
* where the first word in the output sequence might be due to final word in the input sequence
* For Speech Recognition, it is not needed
* Intuitively, when decoding a token at a time step, the focus should be on neighboring acoustic features only
* As the model decodes from left to right, the attention weightage should move from left to right also like in the following figure

<img src="images/i21.PNG" width="400"/> 

* But since attention can consider entire sequence
* The attention weightage may jump around instead of moving from left to right when decoding
* The author added a component called Location Aware Attention

### Location Aware Attention

<img src="images/i22.PNG" width="400"/> 

* Define a window, for example, the window or range is 3 neighboring acoustic features in the figure above
* In short, the attention weights undergo some transform called process history
* Then, it is used as input to the match function
* The model learns to move attention weightage from left to right when decoding

### Does LAS work ?

<img src="images/i23.PNG" width="400"/> 

* In the original paper, the author evaluated it using TIMIT dataset
* The performance does not surpass the hybrid of HMM and deep learning
* However, this result might not be true for actual Speech Recognition using other type of tokens
* TIMIT uses phoneme as token

<img src="images/i24.PNG" width="400"/> 

* Experiments by other researchers showed that LAS beat HMM
* Chiu shows that using end-to-end model like LAS reduced the size of the model
* The conventional LFR system which combines separate models such as Language Model (LM) is much larger than LAS model
* LAS includes Language Model as a end-to-end model
* Of course, adding an extra language model might improve the performance of LAS a little bit

<img src="images/i25.PNG" width="400"/> 

* The figure above shows that LAS can learn focus from left to right when decoding even when Location aware attention is not used

<img src="images/i26.PNG" width="500"/> 

* The figure above shows an interesting result
* The ground truth is <code>aaa</code>
* The second best path decodes into <code>triple a</code>
* They sound very different
* But it seems LAS is able to learn very complicated relationship between input and output sequence and also the relationship between tokens

## Limits of LAS

* LAS start decoding after listening to the whole input
* User may expect on-line speech recognition
* Which means decoding as user is speaking, instead of decoding after user finished speaking

# Connectionist Temporal Classification

* CTC was introduced by Alex Graves at ICML 2006
* CTC can be used for on-line streaming speech recognition
* Consists of a encoder
* Encoder is some type of uni-directional RNN
* If use bi-directional RNN, cannot perform on-line speech recognition

<img src="images/i27.PNG" width="400"/> 

* At each time step, there is a linear classifier
* Hidden state output **h** is transformed and pass through a softmax layer
* The output of classifier is the token distribution vector with size **V**
* **V** is the vocabulary size
* Each frame consists of only 10ms of acoustic feature
* It is very short
* Difficult to have 1-to-1 relationship between input sequence and output sequence, ignoring down sampling
* May need use more than one frame to decode one token
* Addition of null, ∅
* ∅ means the model does not know what to output
* It wants to listen more before decoding, therefore the decoded token will be ∅
* The size of the distribution vector becomes **V + 1**
* Since CTC uses ∅, it requires post-processing
* Post-processing:
    * Removing ∅
    * Merge duplicate tokens
* For example,
    >∅ ∅ d d ∅ e ∅ e ∅ p →	deep
* Output sequence may be shorter than input sequence
* During training, how to prepare training data ?
* For example, we have 4 frames <code>x1, x2, x3, x4</code> paired to the word *we*
* How do we know which frame is responsible for which token ?
    * We do not know !
* Possible labelling or **alignment**
    * w ∅ ∅ e
    * w w e e
    * ∅ w ∅ e
* So how ?
    * Enumerate all possible alignments and use them during training !

## Does CTC works ?

<img src="images/i28.PNG" width="500"/> 

* The y-axis shows the assigned probability
* The dotted line is for ∅
* It kinda works !

<img src="images/i29.PNG" width="600"/> 

* The figure above showed the result of a CTC with V = 7000 on a unseen test speech data
* The vocabulary does not have the word "dietary"
* The model outputs "diet" and "terry" instead
* The "community" has two consecutive peaks, the model "stutters"
* Increase **V** improves performance

<img src="images/i30.PNG" width="500"/>

* The figure above shows that CTC without post-processing have bad performance
* Due to CTC requires post-processing, there are some with opinion that it should not be considered an end-to-end model

### Issue with CTC

* The classifier at each time step decodes independently
* For example, if the model is supposed to decode only one *c*
* Instead, the output sequence is:
    > <code> c ∅ c </code>
* At the third time step, it outputs *c* again 
* The model "stutters"
* The classifier does not know ∅ is generated at second time step
* This is only an exaggerated example
* The encoder is a RNN, it still can pass information from previous time steps

# RNN Transducer (RNN-T)

* Before taking about RNN-T, let's talk about Recurrent Neural Aligner (RNA)

## Recurrent Neural Aligner
* CTC decoder takes one vector or frame as input and outputs one token
* RNA adds dependency by changing it to a RNN decoder

<img src="images/i31.PNG" width="200"/> 

* As shown in the figure above, the previous token distribution is used as input in current time step
* Similar to LAS decoder
* Can one vector maps to multiple tokens ?
* For example, can a vector maps to "th" ?
* Of course, you can add "th" as a new token in practice
* But we want our model to be flexible 
* RNN-T is the solution to this question !

## RNN Transducer

<img src="images/i32.PNG" width="400"/> 

* At a time step, continues to take in the same hidden state output and  outputs token until ∅ is outputted
* After the model outputs ∅, go to the next frame
* This enable the model to output multiple tokens for a frame
* Have the same alignment problem as CTC, how to utilize all of the possible alignments during training ?
* Addition of an extra RNN which acts as Language Model

<img src="images/i33.PNG" width="400"/> 

* The additional RNN takes the generated output as input 
* The output of the RNN is consumed as input by RNN-T to generate the next tokens
* This RNN will ignore ∅
* Why need to have an extra RNN at decoder ?
    * It acts as Language Model, can easily collect texts and train
    * It is critical for training algorithm
    * We need an algorithm which can enumerate all possible alignments during training, which has a pre-condition of ignoring ∅

# Neural Transducer

* RNN-T reads acoustic feature one at a time 
* Neural transducer reads a bunch of acoustic features 

<img src="images/i34.PNG" width="400"/> 

* It applies attention to a window of acoustic features, generates tokens until ∅ is generated
* Then, the window is moved, the same thing is repeated with new acoustic features in the window
* The decision to focus on which acoustic features in the window is determined by attention mechanism

<img src="images/i35.PNG" width="500"/> 

* The figure above shows the effects of window size
* Without using attention: only take in the last acoustic feature in the window
    * The model performance deteriorates as window size increases
* LSTM-attention : the attention weights at current time step is passed to a LSTM
* The results show that as long as there is attention mechanism, the window size does not matter

# Monotonic Chunkwise Attention (MoCHA)

* Similar to Neural Transducer, with some modification at the attention
* Dynamicly shift the window

<img src="images/i36.PNG" width="300"/> 

* Have an extra component which decides to place window at certain time step or not
* If yes, place the window and vice versa
* This model only output one token for each window
* Does not output ∅
* The output of **here?** is binary, cannot different binary, how to deal with this, refer to the paper [Chiu, et al., ICLR’18]

# Summary

| MODEL             | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| LAS               | Seq2Seq                                                                                    |
| CTC               | Seq2Seq with linear classifier as decoder                                                  |
| RNA               | Seq2Seq, 1-to-1                                                                            |
| RNN-T             | Seq2Seq, 1-to-many                                                                         |
| Neural Transducer | Seq2Seq, input one window<br>Takes in multiple acoustic features, outputs multiple tokens  |
| MoCHA             | NT, dynamic shift window<br>Takes in multiple acoustic features, output 1 token per window |

# Hidden Markov Model
## Overview
* Hidden Markov Model (HMM) is used in speech recognition

<img src="images/i37.PNG" width="230"/> 

* where **X** denotes the acoustic features , **Y** denotes the text
* The equation means that if we can calculate **P(Y|X)**
    * Enumerate all possible Y
    * Calculate P(Y|X) for each Y
    * Select the Y with the highest P(Y|X)
* **Y*** denotes the Y with highest P(Y|X)
* This is known as **Decode**
* Using Bayes' theorem :

<img src="images/i38.PNG" width="230"/> 

* P(X) is ignored because unrelated
* **P(X|Y)** : Acoustic model
    * For example, HMM
* **P(Y)** : Language model
* HMM models **P(X|Y)**, therefore it must be used together with language model
* Deep learning end-to-end models such as LAS and CTC model **P(Y|X)**
* Theoretically, deep learning models do not need language model
* However, adding language model or P(Y) does help with the performance

## Phoneme, Tri-phone and states
* HMM does not use sequence of tokens
* Instead, it uses sequence of states, denoted by **S**
* Instead of P(X|Y), HMM utilizes P(X|S)
* Sequence of tokens is converted to sequence of states

<img src="images/i39.PNG" width="400"/> 

* Firstly, phoneme is converted to tri-phone 
* Phoneme is the unit of sound
* Tri-phone is a unit smaller than phoneme
* A phoneme's pronunciation might be affected by neighboring phoneme
* For example, there are 2 occurrence phoneme **uw** have different pronunciations due to previous phonemes **d** and **y**
* Combine **uw** with neigboring phonemes to form more specific token
* For example, **t-d+uw** and **d-uw+y**
* The next step is to convert tri-phone into states
* Number of states for each tri-phone can be 3 or 5, a value to be set 
* If it's 3
* Then we further split each tri-phone to 3 states such as **t-d+uw1**, **t-d+uw2** and **t-d+uw3**
* These 3 states are more specific sound unit
* Combining these 3 states will result in the **t-d+uw** sound

## Transition Probability and Emission Probability
<img src="images/i40.PNG" width="400"/> 

* Assuming the states *a*, *b* and *c* correspond to the acoustic feature vectors as shown in the figure above
* Requires two type of probabilities
* Transition Probability : Probability to transition from one state to another
    * For example, *a* and *b* are two states
    * Transtion probability from *a* to *b* = <code>P(b|a)</code>
* Emission Probability <code>P(x|s)</code>: Given a state,  what's the probability of producing a sequence of acoustic features
    * Assume every state produces sound with fixed distribution
    * For example, <code>P(x | "t-d+uw1")</code>
    * Each state's sound distribution can be represented or modelled with a Gaussian Mixture Model (GMM)
    * That's why HMM does not use tokens such as grapheme because the sound or pronunciation of character differs depending on usage 
* About 30 phonemes, then convert to tri-phone and then each tri-phone to 3 states
    * Result in <code>(30^3)*3</code> states
    * Too many states !
* One critical concept to solve problem of too many states is **tied-state**
* Tied-state : 
    * Originally, each state has a GMM
    * Assume some states have same sound
    * So, some states share a GMM
    * In practice, these states share the pointer, which point to the address of the shared GMM
    * The ultimate form of tied-state is **Subspace GMM**

## Alignment
* Even we have transition probabilities and emission probabilities, we still cannot calculate <code>P(X|S)</code>
* Before this, we assume a certain correspondence between state sequence and sequence of acoustic features
* But really, we have no idea of the correct **alignment** of state sequence 
* This part is **hidden** part of HMM
* In other words, we do not know which state produces each acoustic feature
* There are many possible alignments as shown in following figure 

<img src="images/i41.PNG" width="500"/> 

* In the figure, we have 3 states and 6 acoustic feature vectors
* Alignment is denoted using **h**
* One alignmnment is <code>aabbcc</code>
* The second alignment is <code>abbbbc</code>
* The probability of a acoustic feature given an alignment <code>P(X|h)</code>can be calculated by multiplying the transition probabilities and emission probabilities togther
* Different alignment will result in different **P(X|h)**

<img src="images/i42.PNG" width="400"/> 

* **P(X|S)** can be calculated by enumerating all possible alignments, and summing their P(X|h)
* <code> h = align(S)</code> means that the alignments have to be valid
* It has to follow the sequence *a* to *b* to *c*
* Examples of invalid alignment:
    * h = abccbc ( *c* occurs before *b* )
    * h = abbbbb ( without *c* )

## Using Deep Learning

### Method 1 Tandem

<img src="images/i43.PNG" width="500"/> 

* The first method is called **tandem**
* This method does not change or modify the HMM usage
* A deep neural network is added as a layer between the acoustic features and the HMM
* Train a deep neural network as a state classifier which predict the acoustic feature belongs to which state
* The output of the classifier can be interpreted as a vector which consists of the conditional probability distribution of states given the acoustic feature
* As shown in the figure above, the states are *a*, *b*, *c* and more
* Given an acoustic feature's vector as input <code>x_i</code>, the state classifier generates a vector which consists <code>P(a|x_i), P(b|x_i) </code> and more
* The output vector is used to substitute MFCC as a new acoustic feature for HMM
* Besides using the output of classifier, one may also opt to use the output of hidden layer of the state classifier as new acoustic feature of HMM


### Method 2 DNN-HMM Hybrid

<img src="images/i44.PNG" width="500"/> 

* For HMM, each state has a GMM which represents the distribution of acoustic feature given that state 
* As shown in the figure above, let's say we have a state *a*, the GMM models the <code>P(x|a)</code>
* The idea of DNN-HMM hybrid is to **replace GMM with a deep neural network** 
* GMM models **P(x|a)**
* Whereas, the deep neural network models **P(a|x)**
* They are modelling different things
* As shown in the figure above, **P(x)** is irrevelant and **P(a)** can be counted from training data
* Based on the expression, the DNN output can be thought of as the scaled version of P(a|x) by a factor of P(a)
* The deep neural network predict the state of a given acoustic feature

**Summary:**
* **P(x|a)** can be written in terms of **P(a|x)**
* P(a) can be counted from training data
* P(x) is not related, so it's ignored
* Basically, can still derive P(x|a) from DNN's output

**Opinions** 
* Advantages of why researchers claimed it is better than HMM in performance, note are not good reasons:
    * Deep neural network can do discriminative training
        * However, HMM can also do discriminative training
    * Deep neural network has many parameters
        * GMM can also have many parameters

* Prof. Lee's opinion :
    * Both of the reasonings up there are not adequate to explain the performance
    * Originally, each state must their own GMM with means and variances
    * Now every state shares a DNN 
    * The DNN plays a role similar to the Subspace GMM 

### Method to train a state classifier
* Do alignment and get the alignment with the highest probability
* Train DNN to predict the acoustic feature's state 
* This creates better aligment 
* Then, train the DNN again with this better alignment

**Process**
* Begin with data and labels but without alignment
* Initially, the alignment of acoustic features and states is not known
* Train HMM-GMM model first
* Select alignment with the highest probability 
* Train a deep neural network (named *DNN 1*) based on this alignment
* Realignment: *DNN 1* is used to do alignment again
* Train another deep neural network (named *DNN 2*) with realigned data
* Repeat these steps

## Alignment of HMM, CTC and RNN-T
### LAS Does Not Require Alignment
<img src="images/i46.PNG" width="200"/> 

* End-to-end models such as LAS can be thought of as modeling P(Y|X)
* For example, Y can be tokens such as words and phonemes which are larger than states
* X refers to the acoustic feature

<img src="images/i47.PNG" width="300"/> 

* During decoding, we want to get the Y with the highest probability denoted by Y*
* However, we cannot really get Y* because we cannot solve the optimization problem


<img src="images/i48.PNG" width="300"/>

* LAS directly computes P(Y|X)
* *c* denotes the context vector
* The *superscript* denotes the time step
* <code>*c*0</code> denotes the context vector for the first time step

**Process :**
* Both *z*0 and *c*0 are used to calculated *z*1, which goes to some transform to generate a vector of size **V**
* That vector is the probability distribution of Y
* For example, *a* is selected because its probability is the highest
* The vector representing *a* is used as input to calculate *z*2, which generates the distribution vector 
* Multiplying the probabilities together to get P(Y|X)
    * <code>P(Y|X) = P(a|X) * P(b|a,X) *....</code>
* Training does not solve the decoding optimization problem, it uses an approximation instead
* It means that we don't really maximize Y with largest P(Y|X) during training
* During training, an approximation is used. This is different from finding the maximal P(Y|X) of decoding optimization problem
    * During training, we find the maximum of distribution vector at each time step. This is different from P(Y|X). Therefore, it's written as <code>Y^</code>
    * During training, find parameters so that the probability <code>Y^</code> given X as large as possible
    * Therefore, have to use beam search to find a better result


## Alignment of CTC and RNN-T
* LAS does not need alignment because it can directly compute P(Y|X)
    * <code>P(Y|X) = P(a|X) * P(b|a,X) *....</code>
* Other speech recognition models such as **CTC** and **RNN-T** need **alignment** to calculate P(Y|X)
* For example:
    * Acoustic features : x1, x2, x3, x4
    * Tokens : a,b
* Duplicate tokens or add ∅ so that the sequence length matches the sequence length of acoustic features
* We can list down alignments. For example :
    * Aligment h1 = a, ∅, b, ∅ 
* However, we can only compute probability for certain alignment denoted by <code>P(h|X)</code>
* Incapable of computing the probability for certain acoustic features <code>P(Y|X)</code>
* The solution for calculating <code>P(Y|X)</code> for CTC and RNN-T is same as the method of calculating P(X|S) for HMM
    * Summing up the probabilities for all possible valid alignments

<img src="images/i49.PNG" width="500"/>

* <code>align(Y)</code> means valid aligment 
* Summing up all the probabilities for every possible valid alignments denoted by <code>P(h|X)</code> to obtain <code>P(Y|X)</code>

### Content
1. Enumerate all the possible alignments
2. How to sum over all the alignments
3. Training
4. Testing (Inference,decoding)
### 1. Enumerate all possible alignments

<img src="images/i50.PNG" width="500"/>

* As shown in the image, there are 6 vectors for acoustic features
* There are 3 characters
* Alignment for HMM : Duplicate the characters to length T
    * Example alignments: c c a a a t , c a a a a t
* Alignment for CTC :
   * Duplicate character AND
   * Add ∅ to length **T**
   * Example alignment : c ∅ a a t t
* Alignment for RNN-T :
    * Add ∅ x T
    * In the figure, **T**=6
    * Remember that when RNN-T generates ∅, it means it is ready to see the next acoustic feature
    * So have to generate 6 ∅ to go through all 6 acoustic features 

**Showing Alignment using Trellis Graph**

**HMM**

<img src="images/i51.PNG" width="500"/>

* As shown by the pseudocode in the figure above, alignment for HMM is done by duplicating the tokens
    * The sum of number of times to duplicate each token is equal to T
    * Each token must be used at least once
* Trellis Graph visualizes the alignments
* There are many possible alignments

**CTC**

<img src="images/i52.PNG" width="500"/>

* Alignment strategy for CTC :
    * Duplicate character AND
    * Add ∅ to length **T**
    * The sum of the number of times each token is duplicated and the number of times ∅ is inserted must be equal to **T**
    * Must add least use each token once
    * Can opt to not insert ∅

<img src="images/i53.PNG" width="400"/>

* If currently at a token, can have 3 options
    1. Duplicate token
    2. Insert ∅
    3. Next token
* Can skip ∅
* The alignment can ends with **t** or **∅**

<img src="images/i54.PNG" width="400"/>

* If currently at ∅, can have 2 options 
    1. Duplicate ∅
    2. Next token
* Cannot skip any token
* The alignment can ends with **t** or **∅**

<img src="images/i55.PNG" width="400"/>

* The figure above shows the summary for the options available for 2 situations

<img src="images/i56.PNG" width="400"/>

* Two possible alignments 

<img src="images/i57.PNG" width="400"/>

* The exception is when we have repeating tokens
* For example, there are two **e** for the word **see**
* In this situation, must insert at least a ∅ between two **e**

**RNN-T**

<img src="images/i58.PNG" width="500"/>

* As shown in the figure above, have to insert at least ∅ **T** times
    * Each token is inserted one time
    * Optional to insert some ∅ before the final token
    * After the final token, insert at least one ∅

<img src="images/i59.PNG" width="500"/>

* Two possible alignments

<img src="images/i60.PNG" width="500"/>

* Visualization for HMM, CTC and RNN-T

### How to sum over all the alignments 

#### Score computation for an alignment

* Lets go through the calculation of score **P(h|X)** for an alignment *h*

<img src="images/i61.PNG" width="500"/>

* The figure above illustrates an alignment <code>*h* =  ∅ c ∅ ∅ a ∅ t ∅ ∅</code>
* The score P(h|X)is a product of conditional probabilities

<img src="images/i62.PNG" width="500"/>

* The figure above shows a RNN-T
* RNN-T has an extra RNN 
    * Represents the relationship between tokens
    * Doesn't eat ∅ as input

**Process :**
* To differentiate between both seq2seq models for RNN-T, we are going to call the main model and the extra RNN as seq2seq model and RNN respectively
* A unique token < BOS > which means start of sentence is taken in by the RNN
* The RNN spits out *l*0
* At the first time step, seq2seq model takes in *l*0 and *h*1 as input and generates a distribution vector denoted by <code>p1,0</code>
* The value for ∅ is the highest in the distribution vector <code>p1,0</code>, it is interpreted as generating a ∅
* For RNN-T, ∅ means it wants to see the second acoustic feature, we denote it as *x*2 here
* *h*2 is the hidden state's output after eating *x*2
* seq2seq model then generates distribution vector denoted by <code>p2,0</code> . The value for the token *c* is the highest 
* RNN eats *c* and activation from previous time step to generate *l*1
* *l*1 then gets consumed by seq2seq model to generate ∅
* The process goes on and on until **T** number of ∅ is generated. This means that ∅ has done eating all the **T** number of acoustic features
* Notice that RNN does not eat **∅** as input, instead ∅ is ignored

<img src="images/i63.PNG" width="500"/>

* Each grid corresponds to a distribution vector
* For example, <code>p4,2</code> means that it has seen 4 acoustic feature vectors and produced 2 tokens
    * subscript 1 : Number of acoustic features seen
    * subscript 2 : Number of tokens produced
* Each arrow which points from a grid to another grid correspond to a probability from the distribution vector 
* <code>p4,2(∅)</code> is the probability of ∅ from the distribution vector denoted by <code>p4,2</code>

<img src="images/i64.PNG" width="500"/>

* The figure above wishes to illustrates the fact that the distribution vector <code>p4,2</code> is fixed or does not change, no matter how you get there
* Why is it so ? The answer is given in the figure
* To the right of the Trellis graph, there is a RNN that's side by side and matching the token rows of the Trellis graph
* That RNN is the additional RNN of RNN-T
* No matter what path you take, you have to go through 4 acoustic features and generates 2 tokens
* Since RNN ignores ∅, the input to the RNN will surely be the second token **a**
* RNN's previous step activation also remain the same no matter how many ∅ is generated before 
* Both inputs remain the same, thus the RNN will spit out the same *l*2 regardless of the paths taken
* By the time to generate <code>p4,2</code>, seq2seq model will be taking *h*4 and *l*2 as input
* This results in the same <code>p4,2</code> to be generated regardless of the paths taken

### 2. How to sum over all alignments

<img src="images/i65.PNG" width="500"/>

* <code>α_*i*,*j* </code> is the summation of the summation of the scores of all the alignments that reads *i*-th acoustic features and output *j*-th tokens
* Each grid corresponds to a **α**
* α of an immediate grid is the weighted average of all the α of grids which are one step away of that grid and are valid option to transition to that immediate grid, weighted by the probabilities that represented by the arrows of the grids leading to the immediate grid
* As shown in the figure, α4,2 can be calculated using α3,2 and α4,1 and are weighted by p4,1(a) and p3,2(∅)
* The α_i,j at the bottom right is equal to P(Y|X) = sum of all the P(h|X) for every possible valid alignments of Y denoted by <code>h ∈ align(Y) </code>
* It means you can compute the summation of the scores of all the alignments 
* How to compute ?
* Using the idea of dynamic programming, one can calculate α for every grid, starting from the top left grid

### 3. Training

* The following expression is the objective function for training :

<img src="images/i66.PNG" width="200"/>

* During training, we want find a set of parameters denoted by <code>θ*</code> which maximizes objective function
* Given sequence of acoustic features denoted by **X** and the annotation for **X**, we want to maximzes the *log* of <code>P(Y^|X)</code>
* The following expression shows how to obtain <code>P(Y^|X)</code>

<img src="images/i67.PNG" width="200"/>

* Basically, summing over all the scores of all the valid alignments
* The score of an alignment is denoted by <code>P(h|X)</code>, it is the product of all the probabilities
* The following figure shows an example 

<img src="images/i68.PNG" width="500"/>

* Remember the Trellis graph ? Each of the grid of the Trellis graph represents a distribution vector such as <code>p1,0 , p2,0 ...</code>
* The arrow of the Trellis graph represents a probability from the distribution vector
* For example, <code>p1,0(∅)</code> is the probability of ∅ from the distribution vector <code>p1,0</code>. 
* Essentially, score of an alignment can be obtained by multplying the probabilities represented by the arrows leading from the start to end of alignment in Trellis graph
* The following figure shows that all the alignments

<img src="images/i69.PNG" width="500"/>

* Each alignment is a path from start to end 
* P(Y^|X) can be obtained by multiplying the probabilities for all the paths and sum them up together
* The loss function needs to be differentiated againts parameters to obtain the gradients denoted by <code>dP(Y^|X)/dθ</code> so that we can solve it using gradient descent

<img src="images/i70.PNG" width="300"/>

* P(Y^|X) is influenced by the probabilities such as p4,1(a) and p3,2(∅) which in turn are influenced by the parameters of the model θ
* The following figure shows that derivative of P(Y^|X) with respect to the parameter can be obtained using chain rule

* It is the sum of partial derivative of the P(Y^|X) with respect to each intermediate priobability multiplied by the partial derivative of the intermediate probability wirh respect to the parameter

<img src="images/i71.PNG" width="500"/>

* The partial derivative of intermediate probability with respect to θ can be obtained using backpropagation through time as shown in the following figure 

<img src="images/i73.PNG" width="500"/>

* The process is same as backpropagation for normal seq2seq model

<img src="images/i74.PNG" width="500"/>

* The figure above shows that the partial derivative of P(Y^|X) against the intermediate probability p4,1(*a*) can be expressed as the sum of scores for all the alignments which have p4,1(*a*) scaled by <code>1/p4,1(*a*)</code>

<img src="images/i75.PNG" width="500"/>

* Introducing new stuff here, <code>β*i*,*j*</code> is the summation of the score of all the alignments starting from *i*-th acoustic feature and *j*-th token to the end of alignment
* Each grid *i*,*j* has a β*i*,*j*
* β is the opposite to the α
* Let say, we want to calculate β4,2
* From the grid for p4,2 , there are two options which will lead us into two different grids:
    * Generate *t* leads to the grid 4,3
    * Read *x*5 and generate ∅ leads to the grid 5,2
* β4,2 is the weighted average of β4,3 and β5,2 weighted by the probabilities obtained from the distribution vector p4,2

<img src="images/i76.PNG" width="500"/>

* Going back to calculate the sum of scores for all the alignments with p4,2(a)
* This sum is <code>α4,1 * p4,1(*a*) * β4,2</code>
* α4,1 sums up score for all the alignments from the start to the grid 4,1
* If we generate *a* when at grid 4,1 , we would reach the grid 4,2
* β4,2 sums up score for all the alignments from the grid 4,2 to the end
* Multiplying together these terms we can obtain the partial derivative
* At this point, we know how to compute the gradients during training