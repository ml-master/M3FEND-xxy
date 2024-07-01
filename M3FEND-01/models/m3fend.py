import os
import torch
from torch.autograd import Variable
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import logging
import math
from sklearn.cluster import KMeans
import numpy as np
from torch.nn.parameter import Parameter

def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim = 1))

def norm(x):
    length = cal_length(x).view(-1, 1)
    x = x / length
    return x

def convert_to_onehot(label, batch_size, num):
    return torch.zeros(batch_size, num).cuda().scatter_(1, label, 1)

class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num = 10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num # 每个domain的event memory矩阵存放的all_feature数量（聚类的数量）
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        self.domain_memory = dict() # 要经过M3FENDModel的save_feature()和init_memory方法获得

    def forward(self, feature, category):
        """reading operation前面计算similarity distribution v 的部分
        feature: 一个batch中所有样本的all_feature
        """
        feature = norm(feature)
        domain_label = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()
        
        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

        return domain_att

    def write(self, all_feature, category):
        """
        feature: 不是经过聚类之后的center，而是所有样本将三个视角进行拼接之后直接获得的all_feature
        """
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = [] # 每个domain一个list
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1)) # 将样本的all_feature存放在对应domain的list中
        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0) # 同一 domain 中样本的特征拼接成一个张量。
            # topic_att--sim(多个样本跟M_d中的每个m_i之间的attn矩阵); self.topic_fc(domain_fea_dict[i]--nW; self.domain_memory[i]--特定的M_d
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1).unsqueeze(2)
            # 将 domain 中样本的特征 domain_fea_dict[i] 在第二维度上重复 self.memory_num 次,得到 tmp_fea,m_i
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            # 得add:每个样本n和sim_i之间的相似度add_i,有广播机制，结果维度跟tmp_fea维度相同，（6（sample_num)，10，1024），这个6不是domain数量，表示的是当前domain有的样本数量
            new_mem = tmp_fea * topic_att
            # 在样本维度上取平均--这些样本都是来自相同domain的样本，能获得一个更加鲁棒和通用的 memory 表示,能够捕捉到不同样本之间的共性,同时减少个体样本的噪声和变异的影响。
            new_mem = new_mem.mean(dim = 0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1) # 在样本维度上求平均,并转换为列向量
            """更新M_i"""
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem

class M3FENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, LNN_dim, domain_num,dataset):
        super(M3FENDModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num # 语义
        self.emotion_num_expert = emotion_num #情绪
        self.style_num_expert = style_num #风格
        self.LNN_dim = LNN_dim
        print('semantic_num_expert:', self.semantic_num_expert, 'emotion_num_expert:', self.emotion_num_expert, 'style_num_expert:', self.style_num_expert, 'lnn_dim:', self.LNN_dim)
        self.fea_size =256
        self.emb_dim = emb_dim
        if dataset == 'ch':
            # self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
            self.bert = BertModel.from_pretrained('../chinese_wwm_ext_pytorch/').requires_grad_(False)
        elif dataset == 'en':
            # self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
            self.bert = RobertaModel.from_pretrained('../roberta_base/').requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        content_expert = []  # 语义模块模型--CNN
        for i in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        emotion_expert = [] # 情绪模块模型--MLP
        for i in range(self.emotion_num_expert):
            if dataset == 'ch':
                emotion_expert.append(MLP(47 * 5, [256, 320,], dropout, output_layer=False))
            elif dataset == 'en':
                emotion_expert.append(MLP(38 * 5, [256, 320,], dropout, output_layer=False))
        self.emotion_expert = nn.ModuleList(emotion_expert)

        style_expert = [] # 风格模块模型--MLP
        for i in range(self.style_num_expert):
            if dataset == 'ch':
                style_expert.append(MLP(48, [256, 320,], dropout, output_layer=False))
            elif dataset == 'en':
                style_expert.append(MLP(32, [256, 320,], dropout, output_layer=False))
        self.style_expert = nn.ModuleList(style_expert)

        # domain-adapter？
        self.gate = nn.Sequential(nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
                                      nn.ReLU(),
                                      nn.Linear(mlp_dims[-1], self.LNN_dim),
                                      nn.Softmax(dim = 1))

        self.attention = MaskAttention(emb_dim)

        self.weight = torch.nn.Parameter(torch.Tensor(self.LNN_dim, self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert)).unsqueeze(0).cuda()
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        # domain memory bank：domain characteristic memory（一个domain提取一个总体特征） + domain event memory（每个domain一个event memory，一个domain内所有新闻的all_feature聚类成为10个块的中心点的all_feature)
        # 实际上：domain characteristic memory是通过self.domain_embedder获取到的，MemoryNetwork中的domain_memory(字典）是domain event memory
        if dataset == 'ch':
            self.domain_memory = MemoryNetwork(input_dim = self.emb_dim + 47 * 5 + 48, emb_dim = self.emb_dim + 47 * 5 + 48, domain_num = self.domain_num, memory_num = self.memory_num)
        elif dataset == 'en':
            self.domain_memory = MemoryNetwork(input_dim = self.emb_dim + 38 * 5 + 32, emb_dim = self.emb_dim + 38 * 5 + 32, domain_num = self.domain_num, memory_num = self.memory_num)

        self.domain_embedder = nn.Embedding(num_embeddings = self.domain_num, embedding_dim = emb_dim)
        self.all_feature = {}

        # predictor
        self.classifier = MLP(320, mlp_dims, dropout)
        
    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        category = kwargs['category']
        
        content_feature = self.bert(content, attention_mask = content_masks)[0] # 预处理阶段，提取semantic feature

        gate_input_feature, _ = self.attention(content_feature, content_masks) # 经过了mask-attention之后的语义特征

        """
        完整的domain memory bank 的 reading operation，得到gate_input
        """
        # memory_att就是矩阵v（ppt），调用的是MemoryNetwork的forward()方法
        # torch.cat([gate_input_feature, emotion_feature, style_feature]就是所有新闻的all_feature
        # 计算v矩阵，表示样本特征和各个domain的domain event memory矩阵之间的相似性
        memory_att = self.domain_memory(torch.cat([gate_input_feature, emotion_feature, style_feature], dim = -1), category)
        # C={c_1,c_2...,c_domain_num}， 获取到domain character memory，所有domain的一个完整特征表示
        domain_emb_all = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda())
        # u=v_i*c_i，计算样本特征和那个domain整体特征最相似
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        # c_d 该样本原本的所属的domain整体特征
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        # enriched domain information:[c_d, u]，传入domain——adapter中的输入数据
        gate_input = torch.cat([domain_embedding, general_domain_embedding], dim = -1)

        """经过domain-adapter"""
        gate_value = self.gate(gate_input).view(content_feature.size(0), 1, self.LNN_dim)

        """得到矩阵z（没有做多头的吗？）"""
        shared_feature = []
        for i in range(self.semantic_num_expert):
            shared_feature.append(self.content_expert[i](content_feature).unsqueeze(1)) # r_sem
        for i in range(self.emotion_num_expert):
            shared_feature.append(self.emotion_expert[i](emotion_feature).unsqueeze(1)) # r_emo
        for i in range(self.style_num_expert):
            shared_feature.append(self.style_expert[i](style_feature).unsqueeze(1)) # r_sty
        shared_feature = torch.cat(shared_feature, dim=1)
        # 实现ln(r_sem),ln(r_emo),ln(r_sty)
        embed_x_abs = torch.abs(shared_feature) # 对shared_feature取绝对值
        embed_x_afn = torch.add(embed_x_abs, 1e-7) # 将embed_x_abs与一个小常数1e-7相加,用于数值稳定性。
        embed_x_log = torch.log1p(embed_x_afn) # 对embed_x_afn应用log1p函数,即log(1 + x)。
        # 实现a_sem*ln(r_sem) + a_emo*ln(r_emo) + a_sty*ln(r_sty)
        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out) # 应用expm1函数,即exp(x) - 1
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        """模型右侧输出和左侧输出相乘"""
        shared_feature = torch.bmm(gate_value, shared_feature).squeeze()

        """分类器预测新闻真假"""
        deep_logits = self.classifier(shared_feature)

        return torch.sigmoid(deep_logits.squeeze(1))

    def save_feature(self, **kwargs):
        """
        初始化domain event bank，得到还没有进行聚类的结果
        将新闻的all_feature按照各自的domain标签进行划分，
        存放在self.all_feature字典中，其中domain标签作为key，对应domain的所有新闻的all_feature组成的matrix作为对应的value
        """
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        content_feature = self.bert(content, attention_mask = content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)

        for index in range(all_feature.size(0)):
            # 将64个样本的各自all_feature按照各自的domain进行分类存储在self.all_feature字典中
            """Each domain has a Domain Event Memory matrix"""
            domain = int(category[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).cpu().detach().numpy())

    def init_memory(self):
        """对save_feature()得到的还没有进行聚类的domain evnet memeory进行聚类，得到最终的初始化结果"""
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            kmeans = KMeans(n_clusters=self.memory_num, init='k-means++').fit(all_feature)
            centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).cuda()
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']

        category = kwargs['category']

        content_feature = self.bert(content, attention_mask = content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)

class Trainer():
    def __init__(self,emb_dim,mlp_dims,use_cuda,lr,dropout,train_loader,val_loader,test_loader,category_dict,weight_decay,
                 save_param_dir,semantic_num,emotion_num,style_num,lnn_dim,dataset,early_stop = 5,epoches = 100,):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict # 新闻domain类别的字典，根据传入的参数的不同，分为3类，6类，9类三种
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num # 语义提取模块的模型层数 7
        self.emotion_num = emotion_num # 情绪提取模块的模型层数 7
        self.style_num = style_num # 风格提取模块的模型层数 2
        self.lnn_dim = lnn_dim
        self.dataset = dataset # "ch" /"en"

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = M3FENDModel(self.emb_dim, self.mlp_dims, self.dropout, self.semantic_num, self.emotion_num,
                                 self.style_num, self.lnn_dim, len(self.category_dict),self.dataset)
        if self.use_cuda:
            self.model = self.model.cuda()
        # 交叉熵损失
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.98)
        """ 一、初始化domain memory bank（其中的domain event memory部分）
        这个初始化过程的目的是为了初始化模型的记忆机制(memory mechanism)或者特征存储(feature storage)
        在某些模型架构中,特别是那些涉及到记忆机制或者特征存储的模型,在开始主要的训练循环之前,需要先在训练数据上"预热"或者初始化模型的记忆或特征存储。
        主要是初始化MemoryNetwork类实例domain_memory的domain_memory(字典）属性，其他属性已经初始化完毕
        """
        self.model.train() # 设置为train模式
        # 将训练数据加载器 self.train_loader 包装在一个 tqdm 对象中,以创建一个带有进度条的迭代器
        train_data_iter = tqdm.tqdm(self.train_loader)
        for step_n, batch in enumerate(train_data_iter):
            # 一个batch中有10个列表，分别存储不同的GT属性值
            batch_data = data2gpu(batch, self.use_cuda)
            """ A. save_feature：先将所有新闻的all_feature按照domain进行划分，存成对应的键值对"""
            label_pred = self.model.save_feature(**batch_data)
        """ B. init_memory：对刚才的划分结果进行聚类，将每个domain聚类得到的10个中心点的all_feature代替该domain所有新闻的all_feature
            聚类结果作为MemoryNetwork类实例domain_memory的domain_memory(字典）属性，完成domain memory bank的初始化
        """
        self.model.init_memory()
        print('initialization finished')

        """ 二、训练模型 """
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                """ A.模型预测 
                包含：1.完整的domain memory bank 的 reading operation，得到gate_input；
                     2.gate_input传入domain_adapter中；
                     3.模型左侧处理，得到z；
                     4.将左右两侧的输出进行矩阵相乘，传入predictor中进行分类预测                     
                """
                label_pred = self.model(**batch_data) # 模型根据上一轮的domain memory bank的内容，预测当前样本的真假
                """ B.计算损失函数，梯度下降，更新参数"""
                loss =  loss_fn(label_pred, label.float())  # 计算交叉熵损失
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    """ C. write operation,更新域记忆库中的域事件记忆单元"""
                    self.model.write(**batch_data) # 更新domain memory bank的内容
                if(scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item()) # 求所有样本的损失平均值，作为当前batch的损失值
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss.item())

            """三、每个epoch训练完之后，验证集上验证模型效果"""
            self.model.train()
            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_m3fend.pkl'))
                self.best_mem = self.model.domain_memory.domain_memory
                best_metric = results['metric']
            elif mark == 'esc': # 可以出发早退条件，提前结束epoch循环
                break
            else:
                continue

        """四、所有epoch训练完成后，测试集上测试模型效果"""
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_m3fend.pkl')))
        self.model.domain_memory.domain_memory = self.best_mem
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_m3fend.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data) # 模型预测分类结果

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)
