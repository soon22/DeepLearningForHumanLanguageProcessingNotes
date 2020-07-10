## Speech Recognition

### Overview

* Speech recognition converts **speech** to **text**
* Speech can be represented using sequence of vectors with length **T** and number of dimension **D**
* Text can be represented using sequence of **tokens** with length **N**
* **V** denotes the number of different tokens, it is a fixed number
* Usually, **T** > **N**

### Different Variations of Tokens

#### Phoneme :
* A unit of sound 
* For example:The phonemes which make up <code>One Punch Man</code>
> * <code>W AH N P AH N CH M AE N</code>
* Requires **lexicon** which is basically a dictionary of words to phonemes
* Prior to Deep Learning era, phoneme is a widely used as tokens because it is directly related to sound signal
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
>* 一,拳,超,人
* On the other hand, grapheme for Chinese does not require spacing
* Advantages: 
    * Lexicon free. Does not require linguists to build the lexicons, can start training right away
    * The model may learn to construct words that do not exist in the training set
* Disadvantage:
    * Does not have clear or direct relationship with sound signal
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
* Greater than grapheme but smaller than word
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


* After going through more than 100 papers, ICASSP 2019, INTERSPEECH 2019, ASRU 2019
* Grapheme is the most widely used choice because it doesn't require lexicons
* Phoneme is the second widely used choice because it is directly related to sound
* The least used is word

Others 

|              Input              |                           Output                           |
|:-------------------------------:|:----------------------------------------------------------:|
| Speech                          | Word embeddings                                            |
| Speech                          | Speech recognition + Translation                           |
| Speech :<br>One ticket to Taipei  | Speech recognition + Intent classification<br><buy_ticket> |
| Speech : <br>One ticket to Taipei | Speech recognition + Slot filling<br>NA NA NA LOC    |

### Acoustic Feature