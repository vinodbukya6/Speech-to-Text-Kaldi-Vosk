# Speech-to-Text-Kaldi-Vosk

Kaldi is an open source toolkit made for dealing with speech data. Kaldi is written mainly in C/C++, but the toolkit is wrapped with Bash and Python scripts.

The three parts of Kaldi
1. Preprocessing 
2. Feature Extraction
   MFCC.
   
   CMVNs: which are used for better normalization of the MFCCs.
   
   I-Vectors: For understanding of both channel and speaker variances.
   
   MFCC and CMVN are used for representing the content of each audio utterance and I-Vectors are used for representing the style of each audio utterance or      speaker.
3. The Model

    Acoustic Model: Which used to be GMM but now it was wildly replaced by Deep neural networks. That model will transcribe the audio features that we created into some sequence of context dependent phonemes.
    
    Decoding Graph(WFST): Which takes the phonemes and turns them into lattices. This is generally the output that you want to get in a speech recognition system.      


For Implementation purpose Tensorflow Speech Recognition Challenge dataset used from Kaggle and followed different sources for training. References are mentioned below. 

# Kaldi Installation Steps

1. git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
2. cd kaldi/tools: In tools directory run following two commands one by on extras/check_dependencies.sh and make
3. cd kaldi/src: In src directory run following commands one by one. We need to compile kaldi with GPU for training DNN Model so run configure with use cuda command(sample given here. Whereis cuda: /usr/local/cuda/)  
./configure --use-cuda --cudatk-dir=/usr/local/cuda/, make depend, make

# Training DNN Model(mini librispeech data)

1. Go to egs/minilibrispeech/s5/ and run following commands
2. Change cmd.sh file 'queue.pl' to run.pl (If you have no queueing system and want to run on a local machine, you can change all instances) and run ./run.sh
3. Change num_jobs_initial and num_jobs_final to number of GPUs we have in the train.sh file. (num_jobs_final =1, num_jobs_initial =1 in my case)
4. If any error check solution on kaldi github repo. Possible errors are ‘out of memory’- solution: https://github.com/kaldi-asr/kaldi/issues/4374

# Kaldi from Scratch

For getting Data ready from scratch see following steps. Example dataset used is wsj data. This example implements STT for mobile applications using Kaldi and Vosk libraries.

# Implementation Steps

# 1. Data Preprocessing 

Preparing data for Kaldi needs three files. Each line of file follows a pattern.

Training and Validation data files: 
    wav.scp: utterance_id path_to_auio
    text: utterance_id transcript
    utt2spk: utterance_id speaker
    
Test data files: 
    wav.scp and utt2spk 

Prepare language data: 
    lexicon.txt: Contain every word in the dataset and its phonemes
    nonsilenece_phones: Every phonemes you have 
    optional_phones: List of optional silence phone
    silence_phones: List of silence phone
    Example lexicon.txt: 
                        bed b eh d,
                        bird b er d,
                        cat k ae t,
                        dog d ao g,
                        
                        
# 2. Training

1. Save above prepared dataset files in ‘kaldi/egs/wsj/s5/your_ folders/files’. 
2. Your folders are ‘data’ folder contains train, test, validation, local, lang folders. Your files are ‘path.sh’ and ‘run.sh’
3. ‘path.sh’ and ‘run.sh’ code need to change according to file names we have. Check ‘KALDI_ROOT’ path in ‘path.sh’ file
4. After saving the files run ‘./path.sh’ after running successfully run the ‘./run.sh --stage 0’ running from stage 0
5. There are different stages 0 - 4 for different training processes.
6. After successfully training results are saved in the ‘exp’ folder.

# Next Steps

Inference from Kaldi trained custom-model and make Mobile Compatible using Vosk API from Kaldi trained files

# Inference using Vosk

For inference using Vosk we need to run ‘local/nnet3/run_ivector_common.sh’  (it contains the common feature preparation and iVector-related parts of the script) and local/chain/run_tdnn.sh (main dnn training process). Change the scripts according to our needs (change training/eval/test set names and other related)

Check if kaldi is compiled with CUDA since these scripts require a GPU to run.

Implementation steps are the same using CPU and GPU, only difference is installation process is different in GPU (setting CUDA process). See this link for more details: http://jrmeyer.github.io/asr/2017/10/13/Kaldi-AWS.html

Installed Kaldi on AWS instance and trained DNN model.

# Conclusions
Trained Kaldi on AWS and got all the files required for mobile android-app. For required files check: https://github.com/alphacep/vosk-android-demo/tree/master/models/src/main/assets/sync/model-android


# Required files are saved in following directories:

1. cp exp/chain/tdnn1*_sp_online/ivector_extractor/final.dubm "$dir/model/ivector"
2. cp exp/chain/tdnn1*_sp_online/ivector_extractor/final.ie "$dir/model/ivector"
3. cp exp/chain/tdnn1*_sp_online/ivector_extractor/final.mat "$dir/model/ivector"
4. cp exp/chain/tdnn1*_sp_online/ivector_extractor/global_cmvn.stats "$dir/model/ivector"
5. cp exp/chain/tdnn1*_sp_online/ivector_extractor/online_cmvn.conf "$dir/model/ivector"
6. cp exp/chain/tdnn1*_sp_online/ivector_extractor/splice_opts "$dir/model/ivector"
7. cp exp/chain/tdnn1*_sp_online/conf/splice.conf "$dir/model/ivector"
8. cp exp/chain/tdnn1*_sp_online/conf/mfcc.conf "$dir/model"
9. cp exp/chain/tdnn1*_sp_online/final.mdl "$dir/model"
10. cp exp/chain/tree_sp/graph_tgsmall/HCLG.fst "$dir/model"
11. cp exp/chain/tree_sp/graph_tgsmall/words.txt "$dir/model"
12. cp exp/chain/tree_sp/graph_tgsmall/phones/word_boundary.int "$dir/model"



# References

1. Kaldi slides: https://www.clsp.jhu.edu/wp-content/uploads/2016/06/Building-Speech-Recognition-Systems-with-the-Kaldi-Toolkit.pdf
2. Kaldi-yesno-tutorial: https://libraries.io/github/keighrim/kaldi-yesno-tutorial and Github repo: https://github.com/keighrim/kaldi-yesno-tutorial
3. Dataset Preparation: https://www.kaggle.com/minhnq/tutorial-how-to-train-asr-on-kaldi-lb-75?scriptVersionId=16804869 and Github repo: https://github.com/minhnq97/asr-commands
4. Vosk-api: HMM Model https://medium.com/@qianhwan/understanding-kaldi-recipes-with-mini-librispeech-example-part-1-hmm-models-472a7f4a0488
5. Vosk-api: DNN Model
https://medium.com/@qianhwan/understanding-kaldi-recipes-with-mini-librispeech-example-part-2-dnn-models-d1b851a56c49
6. Kaldi-Chain Model: https://kaldi-asr.org/doc/chain.html#chain_model
