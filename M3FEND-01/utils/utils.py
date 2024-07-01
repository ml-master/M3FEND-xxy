from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        # 如果当前性能指标没有超过最优性能指标,就判断当前epoch的索引 self.curindex 与最优性能指标对应的epoch索引 self.maxindex 之差是否大于等于 self.early_step。
        #   如果大于等于 self.early_step,表示已经连续 early_step 个epoch没有性能提升,触发早停条件,返回 'esc' 标记。
        #   否则,返回 'continue' 标记,表示继续训练。
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        metrics_by_category[c] = {
            'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
        }

    metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred)
    
    for c, res in res_by_category.items():
        #precision, recall, fscore, support = precision_recall_fscore_support(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), zero_division=0)
        metrics_by_category[c] = {
            'precision': precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
            'recall': recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
            'fscore': f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
            'auc': metrics_by_category[c]['auc'],
            'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
        }
    return metrics_by_category

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(), # (64,170)，值为0~5500+
            'content_masks': batch[1].cuda(), # (64,170)，值为0/1
            'comments': batch[2].cuda(), # (64,170) ,值为0~5500+
            'comments_masks': batch[3].cuda(), # (64,170) ,值为0/1
            'content_emotion': batch[4].cuda(), # (64,47) 值为0.0~1.0 (64,38)
            'comments_emotion': batch[5].cuda(), # (64,94) 值为0.0~1.0 (64,76)
            'emotion_gap': batch[6].cuda(), # (64,94) 值为-1.0~1.0 (64,76)
            'style_feature': batch[7].cuda(), # (64,48) 值为0.0~1.0 (64,32)
            'label': batch[8].cuda(), # (64,) 值为0/1
            'category': batch[9].cuda() # (64,) 值为0-5 取值范围根据类别数量参数值而定
            }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'comments': batch[2],
            'comments_masks': batch[3],
            'content_emotion': batch[4],
            'comments_emotion': batch[5],
            'emotion_gap': batch[6],
            'style_feature': batch[7],
            'label': batch[8],
            'category': batch[9]
            }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v