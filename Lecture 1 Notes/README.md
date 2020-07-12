## Speech Recognition

* [Overview](###Overview)
    * [Output: Different Variations of Tokens](###Output:-Different-Variations-of-Tokens)
        * [Phoneme](####-Phoneme-:)
        * [Grapheme](####-Grapheme-:)
        * [Word](###-Word-:)
        * [Morpheme](###-Morpheme-:)
        * [Byte](###-Bytes-:)
        * [Percentage of usage of different type of tokens](###-Percentage-of-usage-of-different-type-of-tokens)
    * [Input: Acoustic Feature](###-Input:-Acoustic-Feature)
        * [How to obtain acoustic feature](####-How-to-obtain-acoustic-feature-:)
        * [Percentage of usage of different type of acoustic feature](###-Percentage-of-usage-of-different-type-of-acoustic-feature)
    * [Amount of Data](###-Amount-of-Data)

* [Speech Recognition Models](###-Speech-Recognition-Models)


### Overview

* Speech recognition converts **speech** to **text**
* Speech can be represented using sequence of vectors with length **T** and number of dimension **D**
* Text can be represented using sequence of **tokens** with length **N**
* **V** denotes the number of different tokens, it is a fixed number
* Usually, **T** > **N**

### Output: Different Variations of Tokens

#### Phoneme :
* A unit of sound 
* For example: The phonemes which make up <code>One Punch Man</code>
> * <code>W AH N P AH N CH M AE N</code>
* Requires **lexicon** which is basically a dictionary of words to phonemes
* Prior to Deep Learning era, phoneme is a widely used choice of token because it is directly related to sound signal
* Disadvantage: requires experts such as linguists to build the lexicon for every words

#### Grapheme :
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

### Word :
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
    
### Morpheme :
* The smallest meaningful unit
* Greater than grapheme, but smaller than word
* For example, the English word *unbreakable* can be broken down into 3 morphemes: *un*, *break* and *able*

### Bytes :
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

### Input: Acoustic Feature

<img src="images/1.PNG" width="400"/>

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

#### How to obtain acoustic feature :

<img src="images/2.PNG" width="400"/>

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

### Speech Recognition Models 
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

### Listen, Attend, and Spell (LAS)
* LAS is a typical Seq2Seq model with attention
* Listen refers to the encoder part 
* Spell refers to the decoder part
* Attend refers to the attention part
* Why is it called LAS, not Seq2Seq model ?
    * There was a time when it's popular to use three verbs as the title of paper, now it's outdated
    * Other Speech Recognition models are also Seq2Seq model
    * People will not know which model you refer to if you called it Seq2Seq model

### Listen: Encoder Choice
* Listen refers to the encoder part

<img src="images/3.PNG" width="400"/>

* Encoder is a Seq2Seq model which takes the sequence of acoustic features as input and outputs high level representations of the acoustic feature sequence
* **x** denotes the input sequence of acoustic features
* **h** denotes the output sequence of vectors
* We hope that the encoder extracts content information and remove speaker variance and noises
* Speaker variance means variations in acoustic features due to different speakers
* The choice of encoder:
    * RNN
    * 1D CNNs

<img src="images/4.PNG" width="400"/>

* Using 1D Convolutional Neural Network (CNN), the triangle represents the filter
* The filter performs convolution on acoustic features which is in window
* Then the filter is slided rightwards, another convolution on another window of acoustic features
* A value is contributed at each time step when moving the filter right 
across time steps

<img src="images/5.PNG" width="400"/>

* There are more than one filters (different colors)
* Each filter contributes to a value at each time step 
* Filters at higher level can consider longer sequence of acoustic features
* It is common to combine CNN and RNN

### Listen: Down Sampling
* Down Sampling is important in Speech Recognition
* For acoustic feature, 1 seconds of speech results in 100 frames or vectors
* This results in very long sequence
* Self-attention for each time step consider entire sequence in
* This is okay for Machine Translation, but not okay for Speech Recognition because the sequence is too long
* Neighboring vectors share info due to overlap of sliding window
* Down Sampling increases efficiency and reduce computational cost
* Very important, the author mentioned unsuccesful to train LAS without down sampling

**Down Sampling Methods :**

<img src="images/6.PNG" width="400"/>

* Pyramid RNN: Each time step of higher layer takes several output of time steps of lower layer as input
* Pooling over time: Each time step of higher layer skips some time step of lower layer

<img src="images/7.PNG" width="400"/>

* Time-delay DNN (TDNN) : Similar to 1D CNN, but only consider first and final frame 
* Truncated Self-attention : 
    * Apply attention only to part of the sequence in the range
    * Usually, self-attention will look at whole sequence
    * For example, <code>x3</code> to <code>h3</code> will give attention to <code>x1</code> to <code>x4</code>
    * The range is a hyperparameter

### Attention

<img src="images/8.PNG" width="300"/>

* Encoder transforms the input sequence into a sequence of vectors **h**
* There is a vector denoted by **z0**
* Self-attention can be seen as doing a Google Search
* Where **z0** is the keyword and sequence h1, h2, h3, h4 is the database
* There is a **match** function which takes h1 and z0 as input and outputs a scalar <code>alpha_0_1</code>
* Match function can be seen as calculating the similarity between the vectors h1 and z0
* There are two types of match function

**Dot Product Attention**

<img src="images/10.PNG" width="200"/>

* Do linear transform to both **h** and **z**
* Results in 2 vectors with same dimensions
* Then do dot product, which results in a scalar **α**

**Additive Attention**

<img src="images/11.PNG" width="200"/>

* Perform linear transform to both **h** and **z**
* Add both transformed vectors
* Apply **tanh** resulting in a vector
* Multiply with **W** results in a scalar **α**