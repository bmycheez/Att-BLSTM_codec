# Att-BLSTM_codec
This is a PyTorch implementation of the JOURNAL OF BROADCAST ENGINEERING paper, 
[Video Compression Standard Prediction using Attention-based Bidirectional LSTM](http://www.kibme.org/resources/journal/20191015155044174.pdf).

If you find our project useful in your research, please consider citing:
~~~
@inproceedings{kim2019codec,
  title={Video Compression Standard Prediction using Attention-based Bidirectional LSTM},
  author={Kim, Sangmin and Park, Bumjun and Jeong, Jechang},
  booktitle={JOURNAL OF BROADCAST ENGINEERING},
  year={2019}
~~~

# Dependencies
Python 3.6
PyTorch 1.4.0

# Data
We used 18 YUV files and converted them to 
M2V (MPEG-2), H263, 264, MP4 (HEVC), BIT (IVC), WEBM (VP8), JPG, J2K (JPEG2000), BMP, PNG and TIFF.
The list of video and image files are on ADD_video_set.xlsx in Korean.

# Training
Use the following command to use our training codes
~~~
python main.py
~~~
There are other options you can choose.
Please refer to utils.py.

# Test
Use the following command to use our test codes
~~~
python test.py
~~~
The pretrained model Bi-LSTM_96.09.pth is ready.
96.09% is the accuracy of the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).


# Options: scenario
We use the proposed algorithm for classifying the codec of bitstreams encoded with particular scenarios.
When the value, training_scenario is,
0 : training samples not encoded
1 : training samples not encoded and bitwise inversed
2 : training samples not encoded and bitwise xor-d
3 : training samples not encoded, bitwise inversed, bitwise xor-d, endian-swaped every 2 bytes
4 : training samples not encoded and endian-swaped every 2 bytes
When the value, test_scenario is,
0 : test samples not encoded
1 : test samples bitwise inversed
2 : test samples bitwise xor-d
3 : test samples endian-swaped every 2 bytes

# Results
When we tested using data with 3 codecs, MPEG-2, H.263, and H.264, we made the network with accuracy 99.39%.
When we tested using data with all 11 codecs, we made the network with accuracy 96.09%.

# Contact
If you have any question about the code or paper, feel free to ask me to <ksmh1652@gmail.com>.
