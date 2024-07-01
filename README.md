# M3FEND模型在风格改写数据集上的优化与评估

这是论文[《记忆引导的多视图多域假新闻检测》](https://ieeexplore.ieee.org/document/9802916)的官方实现，以及在新数据集上的优化和评估该论文已在 TKDE 上发表。

论文假新闻的广泛传播对个人和社会的威胁日益严重。人们为单一领域（如政治）的自动假新闻检测做出了巨大努力。然而，多个新闻领域之间通常存在相关性，因此同时检测多个领域的假新闻是有希望的。根据我们的分析，我们提出了多领域假新闻检测的两个挑战：1）领域转移，由领域在词汇、情​​感、风格等方面的差异引起。2）领域标签不完整性，源于现实世界的分类只输出一个领域标签，而不管新闻片段的主题多样性如何。在论文中，作者提出了一个记忆引导的多视图多域假新闻检测框架（M3FEND）来解决这两个挑战。我们从多视角对新闻片段进行建模，包括语义、情感和风格。具体来说，我们提出了一个领域记忆库来丰富领域信息，它可以根据看到的新闻片段和模型领域特征发现潜在的领域标签。然后，以丰富的领域信息作为输入，领域适配器可以自适应地聚合来自多个视图的判别信息，以适应各个领域的新闻。在英文和中文数据集上进行的大量离线实验证明了 M3FEND 的有效性，在线测试验证了其在实践中的优越性。

为了进一步验证M3FEND的性能，使用了新的数据集gossipcop\_v3-1\_style\_based\_fake.json。这个数据集的设计初衷是为了深入研究不同写作风格对新闻真实性判断的影响，以及探索基于风格的虚假新闻检测方法的有效性。通过对比原始新闻和风格改写后的新闻，研究者可以系统地分析语言风格的变化如何影响新闻的可信度和读者的判断。这种设计为虚假新闻检测领域提供了一个独特的研究视角，有助于开发更加健壮和有效的检测算法

## 介绍

这个github提供了M3FEND的官方实现代码，风格改写数据集新数据集的创建，M3FEND模型在新数据上进行训练和十个基线模型（BiGRU，TextCNN，RoBERTa，StyleLSTM，DualEmotion，EANN，EDDFN，MMoE，MoSE，MDFEND）的实现。请注意，在原始实验中，TextCNN和BiGRU是用word2vec作为词嵌入实现的，但是在这个github中，我们用RoBERTa嵌入实现它们。

## 环境依赖

*   Python 3.8

*   PyTorch 1.12.1

*   Pandas

*   Numpy

*   Tqdm

## 官方实现

参数配置:

*   dataset: the English or Chinese dataset, default for `ch`

*   early\_stop: default for `3`

*   domain\_num: the Chinese dataset could choose 3, 6, and 9, while the English dataset could choose 3, default for `3`

*   epoch: training epoches, default for `50`

*   gpu: the index of gpu you will use, default for `0`

*   lr: learning\_rate, default for `0.0001`

*   model\_name: model\_name within `textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend m3fend`, default for `m3fend`

你能通过如下方式进行模型训练

1.  进入M3FEND-main文件夹中

        cd M3FEND-main

2.  运行如下命令，分别在中文和英文数据集上对模型进行训练

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 3
```

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 6
```

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset ch --domain_num 9
```

```powershell
python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset en --domain_num 3
```

The best learning rate for various models are different: BiGRU (0.0009), TextCNN (0.0007), RoBERTa (7e-05), StyleLSTM(0.0007), DualEmotion(0.0009), EANN (0.0001), EDDFN (0.0007), MDFEND (7e-5), M$^3$FEND (0.0001).

***

## 模型在新数据集上训练

## step1：下载相关数据

    cd M3FEND-01

1.  下载预训练模型

    |           模型简称          |   用途   |                                            下载地址                                           |
    | :---------------------: | :----: | :---------------------------------------------------------------------------------------: |
    | `BERT-wwm-ext, Chinese` | 处理中文文本 | [pytorch版本](https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view?pli=1) |
    | FacebookAI/roberta-base | 处理英文文本 |          [hugging face](https://huggingface.co/FacebookAI/roberta-base/tree/main)         |

2.  下载数据集，[下载地址](https://github.com/ICTMCG/M3FEND/tree/main/data)

3.  下载[gossipcop\_v3-1\_style\_based\_fake.json](https://github.com/junyachen/Data-examples?tab=readme-ov-file)文件

### step2：创建新的数据集

1.  进入M3FEND-01文件夹下的create\_dataset文件夹中，依次运行如下代码，完成新数据集创建

        cd M3FEND-01/create_dataset
        python create_new_dataset.py

2.  或者直接解压en-v3-1-00.zar文件，即可获得新数据集的trian.pkl, test.pkl和val.pkl文件，其中：

    |    文件名    | 真新闻数量 | 假新闻数量 |
    | :-------: | :---: | :---: |
    | train.pkl | 3,942 | 8,516 |
    |  test.pkl |  989  | 3,090 |
    |  val.pkl  |  990  | 3,016 |

### step：运行模型

1.  直接在M3FEND-01文件路径下执行如下命令即可在新数据集上训练模型

        cd M3FEND-01
        python main.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset en --domain_num 3

# 对比实验

1.  进入M3FEND-01路径下的baseline文件夹中，运行各个模型对应的main文件即可，注意更换文件名称和model\_name属性值

2.  例如训练BIGRU基线模型就可以使用如下两种方式：

        cd M3FEDN-01
        python main.py --gpu 1 --lr 0.0001 --model_name bigru --dataset en --domain_num 3

    <!---->

        cd M3FEND-01/baseline
        python main_bigru.py --gpu 1 --lr 0.0001 --model_name bigru --dataset en --domain_num 3

# 消融实验

1.  进入M3FEND-01路径下的ablation\_study文件夹中，运行各个消融实验对应的main文件即可，注意更换文件名称，model\_name属性值均为m3fend

2.  例如训练w/o semviwe基线模型就可以直接使用如下命令

        cd M3FEND-01/ablation_study
        python main_wosem.py --gpu 1 --lr 0.0001 --model_name m3fend --dataset en --domain_num 3

# 实验结果

1.  M3FEND在两个数据集上的结果：

    ![](README_md_files/412da6a0-36eb-11ef-9da4-81e95177f5b6.jpeg?v=1\&type=image)

2.  对比实验1，各个基线模型在原始数据集上的实验结果：

    ![](README_md_files/5d8f94c0-36eb-11ef-9da4-81e95177f5b6.jpeg?v=1\&type=image)

3.  对比实验2，各个基线模型在新建数据集上的实验结果：

    ![](README_md_files/8143e6a0-36eb-11ef-9da4-81e95177f5b6.jpeg?v=1\&type=image)

4.  消融实验，M3FEND模型在原始数据集和新建数据集上的消融实验结果：

    ![](README_md_files/f73ef900-3752-11ef-b3ed-2ddda1e370d6.jpeg?v=1\&type=image)

# 预训练模型和实验结果

1.  原始模型、新数据集上训练的模型、新数据集上训练的basline模型和新数据集上消融实验模型对应的训练好的模型参数都在param\_models文件夹中

    |                                          文件名                                          |       解释      |
    | :-----------------------------------------------------------------------------------: | :-----------: |
    |                                      m3fend\_org                                      | M3FEND-main模型 |
    |                                      m3fend\_new                                      |  M3FEND-01模型  |
    |          bert、bigru、dualemotion、eann、eddfn、mdfend、mmoe、mose、stylelstm、textcnn         |  各个baseline模型 |
    | m3fend\_woada、m3fend\_woemo、m3fend\_wointer、m3fend\_womeo、m3fend\_wosem、m3fend\_wosty |    各个消融实验模型   |

2.  所有模型训练过程中的输出结果都在results文件夹中

    |                                                      文件名                                                      |          解释          |
    | :-----------------------------------------------------------------------------------------------------------: | :------------------: |
    |                                                    main.txt                                                   |  M3FEND-01模型训练过程输出结果 |
    |                                                  main(1).text                                                 | 各个baseline模型训练过程输出结果 |
    |    bert.txt、bigru.txt、dualemotion.txt、eann.txt、eddfn.txt、mdfend.txt、mmoe.txt、mose.txt、stylelstm、textcnn.txt   |   各个消融实验模型训练过程输出结果   |
    | m3fend\_woada.txt、m3fend\_woemo.txt、m3fend\_wointer.txt、m3fend\_womeo.txt、m3fend\_wosem.txt、m3fend\_wost.txty |  M3FEND-01模型训练过程输出结果 |

## Reference

    Zhu, Yongchun, et al. "Memory-Guided Multi-View Multi-Domain Fake News Detection." IEEE Transactions on Knowledge and Data Engineering (2022).

<!---->

    Nan, Qiong, et al. "MDFEND: Multi-domain fake news detection." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.

or in bibtex style:

    @article{zhu2022memory,
      title={Memory-Guided Multi-View Multi-Domain Fake News Detection},
      author={Zhu, Yongchun and Sheng, Qiang and Cao, Juan and Nan, Qiong and Shu, Kai and Wu, Minghui and Wang, Jindong and Zhuang, Fuzhen},
      journal={IEEE Transactions on Knowledge and Data Engineering},
      year={2022},
      publisher={IEEE}
    }
    @inproceedings{nan2021mdfend,
      title={MDFEND: Multi-domain fake news detection},
      author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
      booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
      pages={3343--3347},
      year={2021}
    }

