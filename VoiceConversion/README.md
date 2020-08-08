# Voice Conversion

* Input is speech and output is speech also
* The content of the speech is preserved
* Many aspects of the speech can be changed while preserving the content. For example :
    * Change the voice of speaker
* This was achieved using Deepfake, which can fool speaker verification system
* This can be done using personalized TTS

## Speaker 
* Same sentence spoke by different person varies in terms of :
    * Speaking style : 
    1. Emotion : Change the emotion of the speaker [Gao, INTERSPEECH'19]
    2. Normal to Lombard [Seshadri, ICASSP'19
    3. Whisper to Normal [Patel, SSW'19]
    4. Singers vocal technique conversion [Luo, ICASSP'20]

## Applications
### Data Augmentation 
* Data augmentation means increasing training data by introducing some modifications
* Using image as an example, rotation of images is a data augmentation method. A slighted rotated face image is still a face image
* If we can transform speech data of a person to another person using voice conversion, we can double the amount of data [Keskinm ICML'19]
* The other way of augmenting speech is by adding noise
* We can either use voice conversion to remove noise or adding noise to clean speech to augment data [Mimura, ASRU'17]

### Improving Intelligibility
* Intelligibility means state or quality of being able to be understood or comprehensible
* We can improve the speech intelligibility of speech of patients [Biadsy, INTERSPEECH'19]
* Accent conversion : [Zhao, INTERSPEECH'19]
    * Change the accent of a non-native speaker 

## Real Implementation
* Input speech (size *T*) → Voice Conversion → Output Speech (size *T*')
* The sequence of input and output speech can vary
* Usually, it is *T* = *T*' in research papers because it's simpler
* If the lengths are the same, Sequence-to-Sequence model is not necessary
* We cannot feed acoustic features directly to the voice conversion model
* Acoustic features cannot be converted into speech directly
* For example, if we use spectrogram
* Spectrogram still lack phase information, or in other words the information of time
* In order to produce speech, we require phase information
* **Vocoder** is used to convert acoustic feature vectors to speech
    * Rule-based : Griffin-Lim algorithm
    * Deep Learning : WaveNet
* Vocoders are used in voice conversion, Text-to-Speech (TTS) and Speech Separation
* **Vocoder will not be covered here**
* The problem lies in data collection
* It is hard to collect large amount of paired data / parallel data
* There are two categories of data namely parallel data and unparalleled data
### Parallel Data
* Parallel data means paired data : The content is the same. Different speakers spoke the same sentence
* If we use parallel data, we would suffer from lack of training data. Two strategies to deal with this :
    1. Use pre-trained models
    2. Then, use a small training set to fine tune the model for specific domain
* In a way, it's similar to image style transfer
* Spectrogram is a matrix which can be visualized as an image
* We can use some ideas of image style transfer on voice conversion with parallel data

### Unparalleled Data
* Unparalleled data : The content of speech are not the same for different speakers
* Application :
    1. Feature disentangle
    2. Direct Transformation

# Feature Disentangle 

* Speech data may consists of information about speaker, content, background, etc
* Feature disentangle is a task of extracting certain attributes or features from the speech data
* For example, we can use it to take away or replace speaker info. 
    * Can transform the speech to make it sounds like spoken by another person
* Can also be used in accent conversion, emotion conversion of the speech

<img src="images/i2.png" width="500"/>

* The figure above shows a possible application of feature disentangle
* We can use unparalleled data of male speech and female speech to train a content encoder, speaker encoder and a decoder
* **Content encoder** extracts the content and phonetic info, filtering out other information such as the male speaker info
* **Speaker encoder** extracts the speaker info while filtering out other information such as content
* Using the content info from the content encoder "Do you want to study Phd ?" and the female speaker info, the **decoder** can generate female speech of "Do you want to study Phd ?"
* The architecture looks like auto-encoder
* Requires extra thing in addition to auto-encoder to train a encoder for speaker and content

## Training Content Encoder

<img src="images/i4.png" width="500"/>
<img src="images/i5.png" width="500"/>

* Don't learn speaker encoder 
* Represent each speaker with one-hot vectors
* For example, represent speaker A with [1,0] and speaker B with [0,1]
* The content encoder encodes the input audio into a vector, then the decoder reconstructs the speech using:
    * encoded vector and 
    * One-hot vector representating the speaker
* This method is assuming we know the utterances belong to which speaker
* We hope that if we train it this way and provides info about speaker using one-hot vector, the content encoder might not need to encode the speaker info
* Issue with using one-hot vector for each speaker : Difficult to consider new speakers

## Speaker Embedding
* Train a speaker encoder first
* The speaker encoder generates an embedding vector which represent the speaker
* Can use a publicly available pre-trained model as speaker encoder
* Is there a way to ensure the content encoder only encodes content ?

## 1. Speech recognition
* Insert a speech recognition model as a content encoder
* Speech recognition converts speech to text, it removes the speaker info entirely while leaving the words which is the content of the speech
* However, cannot use entire speech recognition model directly
* Use part of speech recognition model to join to the decoder
* For example, the state classifier of the HMM-DNN hybrid can be used as content encoder
* The state classifier is used to encode acoustic features into states for HMM speech recognition
* It can also be used to encode content of a speech

## 2. Adversarial Training
<img src="images/i6.png" width="500"/>
* The training is similar to Generative Adversarial Network (GAN)
* A speaker classifier is added 
* It is similar to the discriminator of GAN
* Speaker classifier's job is to decide the encoded vector of content encoder comes from which speaker
* Content encoder's job is to fool the speaker classifier 
* The encoder and speaker classifier are trained iteratively
* After training is done, the speaker classifier would be unable to identify the speaker
* This means that the content encoder's vector does not have speaker information anymore 

## 3. Designing Network Architecture
<img src="images/i10.png" width="500"/>

* Add **Instance Normalization** and **Adaptive Instance Normalization**

### Instance Normalization

* Instance normalization is applied at the content encoder 
* The content encoder is a sequence-to-sequence model which consists of many layers of CNN

<img src="images/i8.png" width="300"/>

* There are more than 1 filters
* Each filter contributes a row or channel

<img src="images/i9.png" width="400"/>

* Each channel is normalized so that each channel has zero mean and unit variance
* Why addition of instance normalization might remove speaker info for the content encoder ?
* Imagine each CNN filter captures certain sound characteristic or pattern
    * For example, high frequency or low frequency
* Each row corresponds to existence of each sound pattern
* If the value is large, it means that the sound pattern is present in the speech
* If the value is small, it means that the sound pattern is absent in the speech
* After instance normalization, the value will not be too large
* This means that the variation in sound characteristic or pattern is diminished
* Hence, removes the speaker info 

### Adaptive Instance Normalization

* We hope the vector which encodes speaker info generated by speaker encoder can influence decoder's info about speaker without affecting or changing content

<img src="images/i11.png" width="400"/>

* Addaptive Instance Normalization is applied in the decoder 
* The vector of speaker encoder undergoes two transform to produce γ and β
* Expect γ and β to embed speaker info to *z* to obtain *z*' 

#### Performance
* Training with the VCTK dataset :

|      | Without IN | With IN |
|------|------------|---------|
| Acc. | 0.375      | 0.658   |

* The results show that adding Instance Normalization does help

<img src="images/i13.png" width="400"/>

* It can also work on unseen data, it can classify the speech correctly 
* As shown in figure above, can classify male speech from female speech

#### Issues

<img src="images/i14.png" width="400"/>

* The problem with using auto-encoder is that the generated speech's quality might not be good
* During training, the input speech to the content encoder and speaker's one-hot vector to the decoder belong to the same speaker
* In other words, the model wasn't trained to do voice conversion
* During testing, we would want the decoder to generate the speech as if A is reading sentence of B
* The generated speech would be **low in quality**
* We need a second stage training to get good voice conversion result

#### Second Stage Training

<img src="images/i15.png" width="400"/>

* Adds a **discriminator**  and a **speaker classifier** after the decoder
* The discriminator determines whether the speech generated by decoder is real or fake
* The decoder now has two tasks :
    * Help speaker classifier 
    * Fool discriminator

<img src="images/i16.png" width="400"/>

* Add a patcher 
* Only learn the patcher in the second stage of training
* The problem is that we do not have ground truth for voice conversion of different speaker
* Even we have no idea of what's the ground truth voice, but at least the converted voice is more realistic

## Direct Transformation

* Transform voice of a speaker to another speaker directly
* Two model architectures :
    * Cycle GAN : 1 to 1 voice conversion of speaker
    * Star GAN : More than 1 speakers

### Cycle GAN

<img src="images/i17.png" width="500"/>

* The generator <code>Gx-y</code> transform the speech of speaker X to speaker Y without changing the content
* The discriminator <code>Dy</code> determines whether the generated speech belongs to speaker Y or not. It outputs a scalar. The larger the value of the scalar, the more confident <code>Dy</code> is.
* Generator <code>Gy-x</code> transforms the speech converted to speaker Y's voice back to speaker X's voice
* During training, it is part of the objective to minimize the difference between the input speech and reconstructed speech
* This is to ensure that the generated speech and the original speech does not vary in content
* <code>Gx-y</code> not only has to generate realistic enough speech in Y's voice, it also has to make sure that the content of the speech does not change

### Star GAN

<img src="images/i18.png" width="400"/>

* Cycle GAN can only change the voice of the speech from one speaker to another
* For Cycle GAN, if there are *N* speakers, we would need *N* x (*N* - 1) generators
* Star GAN's generator can convert the speech to multiple different speakers' voice

<img src="images/i19.png" width="400"/>

* Each speaker is represented as a vector. It tells the Generator which speaker's voice we want to convert the speech to
* We feed the generator the audio of a speaker <code>i</code> and tells it which speaker's voice we want to transform the speech's voice to 
* One-hot vector to represent speaker
* Or use pree-trained speaker encoder which can encode each speaker info into a vector
* The discriminator's role is modified compared to original GAN
* The discriminator's task is to determine whether the generated speech belongs to the input speaker or not

<img src="images/i20.png" width="400"/>

* The difference between Star GAN and Cycle GAN is that the former's discriminator takes in the generated speech and the vector which represents the speaker we intend to convert the voice to as input
* This is because Star GAN is developed for more than a speakers
* Note that Star GAN should consists of a classifier, the classifier is ignored here
* There's no significant conflict between speech disentangle and direct transformation
* We can change the generator to an auto-encoder for speech disentangle