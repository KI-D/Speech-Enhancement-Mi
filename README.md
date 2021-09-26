# Speech-Enhancement-Mi
Install:
git clone https://github.com/KI-D/Speech-Enhancement-Mi.git
cd Speech-Enhancement-Mi
ln -s /nas/datasets/Chinese_Speech_Datasets/data Chinese_data


python == 3.8

pytorch:
pip install torch==1.7.1+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

gpuRIR:
pip install https://github.com/DavidDiazGuerra/gpuRIR/zipball/master

pesq:
pip install https://github.com/ludlows/python-pesq/archive/master.zip

others:
pip install -r requirements.txt



1. FullSubNet:
主体网络在fullsubnet.py中,名称为FullSubNet,只用到了继承的BaseModel的unfold函数,其他函数均未用到.
网络参数集中于两个子网络fb_model和sb_model,均为SequenceModel.
这里SequenceModel的结构为两层的单向LSTM+Linear.

Train:
CUDA_VISIBLE_DEVICES=1,2 python train_fullsubnet.py ./config.yaml --user_defined_name "ANY NAME" [--resume "ANY NAME EXISTED"]

Test:
CUDA_VISIBLE_DEVICES=1,2 python predict_fullsubnet.py ./config.yaml FullSubNet --user_defined_name fullsubnet

2. TCN:
xxx

3. GTSA:
xxx

4. HiFi-GAN:
xxx