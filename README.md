# Install
git clone https://github.com/KI-D/Speech-Enhancement-Mi.git
cd Speech-Enhancement-Mi
ln -s /nas/datasets/Chinese_Speech_Datasets/data Chinese_data

**python == 3.8**
For pytorch:
pip install torch==1.7.1+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

For gpuRIR:
pip install https://github.com/DavidDiazGuerra/gpuRIR/zipball/master

For pesq:
pip install https://github.com/ludlows/python-pesq/archive/master.zip

For others:
pip install -r requirements.txt

# 注意：
1、我最近修改过模型训练时的断点继续的功能，如果modules里没有optimizer.pth等部分，则不支持当前版本代码下的resume继续训练。

2、如果要做正常训练策略下的网络复现，可以直接复制CRN.py文件, 并修改网络名称，"__init__"部分和"forward"部分，如果使用这种方法，那么其他所有部分的代码依然能够服用。只要compute_loss和realtime_process两个函数相应的输入和返回值不变，就不怎么需要修改train.py和predict.py,只需要在两个文件里面import “你的网络”就可以。另外需要在config.yaml里加入网络参数配置（包括默认参数）。

3、目前训练部分的实现是实时处理语音降噪的情况，所以即使在训练时也是长度为3200的语音输入进行串行训练。所以训练起来非常慢，正常基于CNN和RNN的网络收敛大概需要半个月（单卡），一旦加上时序attention后速度几乎是无法接受的慢。之所以没有多卡实现，首先因为预处理会占用一张卡，如果多卡训练那么预处理时会出现问题。第二是我天天在服务器上占那么多卡一训就是半个月整的我很不好意思。如果有兴趣可以看看怎么样来提高训练的效率。


# 1. FullSubNet:
基于《FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement》.

**原始论文3s语音输入效果：SISDR: 18.69，STOI:	0.9123， PESQ:	2.6646**

**经过改进满足实时性的效果还行,就是PESQ太低：SISDR: 20.34，	STOI: 0.9208， PESQ: 2.5201**

主体网络在fullsubnet.py中,名称为FullSubNet,只用到了继承的BaseModel的unfold函数,其他函数均未用到.
网络参数集中于两个子网络fb_model和sb_model,均为SequenceModel.
这里SequenceModel的结构为两层的单向LSTM+Linear.

Train:  CUDA_VISIBLE_DEVICES=1,2 python train_fullsubnet.py ./config.yaml --user_defined_name "ANY NAME" [--resume "ANY NAME EXISTED"]

Test:   CUDA_VISIBLE_DEVICES=1,2 python predict_fullsubnet.py ./config.yaml FullSubNet --user_defined_name fullsubnet

# 2. CRN:
这个网络不在四篇论文范围内, 但目前我做实验下来是效果最好的.从CRN开始，只要接口相同，训练文件均为train.py,测试文件均为predict.py.

**测试结果：SISDR: 20.29，	STOI: 0.9225， PESQ: 2.6518**

Train:  CUDA_VISIBLE_DEVICES=1,2 python train.py TemporalCRN ./config.yaml --user_defined_name "ANY NAME" [--resume "ANY NAME EXISTED"]

Test:   CUDA_VISIBLE_DEVICES=1,2 python predict.py ./config.yaml TemporalCRN --user_defined_name crn
---------------------------------------------------------------------------------------------------------------------------------------
增加了修改版CRN_ELU, 训练测试脚本与上面一致，但CRN.py直接被取代了。
增加了distillation_crn.py, 用于对CRN蒸馏。
增加了predict_distillation.py, 用于测试蒸馏结果, 也可以测量化结果。

**CRN_ELU 测试结果：SISDR: 20.52，	STOI: 0.9244， PESQ: 2.7129, 参数量6.16MB**
**distillation CRN_ELU 测试结果：SISDR: 20.57， STOI: 0.9267， PESQ: 2.7373, 参数量0.81MB**

# 3. TCN:
基于《Speech Enhancement Using Multi-Stage Self-Attentive Temporal Convolutional Networks》。

从之前实验上看，基于Attention的模型效果都不是很好，并且这个网络看起来参数量会比较大。
复现时优先按照原论文复现网络观察效果，如果效果不好，再考虑做相应修改。

# 4. GTSA:
基于《T-GSA: Transformer with Gaussian-Weighted Self-Attention for Speech Enhancement》.

效果比较差，训练起来还贼慢，甚至没有把他完整训练完的想法（所以目前的./modules里的GTSA模型参数是只训练了一会会儿的）。
我也尝试和了许多它的变体（包括现在上传的这一版），但实在是带不动.

Train:  CUDA_VISIBLE_DEVICES=1,2 python train.py GTSA ./config.yaml --user_defined_name "ANY NAME" [--resume "ANY NAME EXISTED"]

Test:   CUDA_VISIBLE_DEVICES=1,2 python predict.py ./config.yaml GTSA --user_defined_name gtsa

# 5. HiFi-GAN:
基于《HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Speech Deep Features in Adversarial Networks》。

像这种基于GAN的方法其实改的是训练策略，至于Generator其实可以是以上任意一种，目前以CRN为Generator,尝试按照原文使用MAE首先预训练两个阶段，之后再上Disciminator.
当前预训练到了第二阶段。
---------------------------------------------------------------------------------------------------------------------------------------
增加了HiFi-GAN代码，

# 6. GeneralBeamformer:
其他的网络加上MVDR Beamformer之后效果都不咋地，SISNR倒是有一定提升，但我想要提升的是PESQ!!!
GeneralBeamformer是基于NN的Beamformer，目前看来似乎参数量比较少并且效果还行，有一定希望，但是就是实时性不咋地，不知道能不能达到要求。然后就是太占显存。
