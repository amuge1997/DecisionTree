import numpy as n



class DTree:
    # 离散决策树
    '''
    输入数据 arr_X:
            每行为一个样本,
            每个样本的同一行都表示一类特征
    输入标签 arr_L:
            每行为一个样本,每个标签向量只有一行为 1 ,其他为 0
    '''

    class NodeClass:
        def __init__(self,it_deep,arr_Label):
            self.subNode = {}
            # 特征(样本的行)
            self.it_featureRow = None
            self.it_deep = it_deep + 1
            # 这个节点使用的训练集
            self.arr_Label = arr_Label
            # 这个节点的预测
            self.probability()

        def add_subNode(self,it_featureRow,it_featureValue,inst_subNode):
            if self.it_featureRow is not None:
                if it_featureRow != self.it_featureRow:
                    raise Exception('add_subNode() 输入的参数\'特征行 featureRow\'错误!')
            self.it_featureRow = it_featureRow
            self.subNode[it_featureValue] = inst_subNode

        def probability(self):
            arr_L = self.arr_Label
            arr_nL = n.sum(arr_L, axis=1)
            arr_sumL = arr_nL.sum()
            arr_proL = (arr_nL / arr_sumL)
            # 每个标签的概率
            self.arr_proLabel = arr_proL

        def isLeaf(self):
            if len(self.subNode) == 0:
                return True
            else:
                return False

    def __init__(self,arr_X,arr_L):
        arr_X = arr_X.T
        arr_L = arr_L.T
        self.root = self.NodeClass(0,arr_L)
        self.node({'X':arr_X,'L':arr_L},self.root)

    # 终止判断
    def isStop(self,fl_gain=None,it_deep=None):
        fl_maxGin = fl_gain.max()
        # 信息增益过小, 停止
        if fl_gain is not None and fl_maxGin < 0.1:
            return True
        # 树过深, 停止
        if it_deep is not None and it_deep >= 5:
            return True
        return False

    # 节点处理
    def node(self,dc_sample,ins_node):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        fl_nowPurity = self.purity({'X': None, 'L': arr_L})         # 计算当前节点的纯度, 用于计算信息增益
        arr_everyFeaturePurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})   # 计算每一个特征分割后的纯度
        fl_gain = fl_nowPurity - arr_everyFeaturePurityExpect       # 计算信息增益

        if self.isStop(fl_gain,ins_node.it_deep):
            return

        it_maxGainFeatureRow = fl_gain.argmax()                     # 获取信息增益最大的特征(行)索引数组
        arr_maxGainArrX = arr_X[it_maxGainFeatureRow]               # 获取信息增益最大的特征(行)样本
        set_ = set(arr_maxGainArrX)
        for i in set_:
            arr_col = n.where(arr_maxGainArrX == i)[0]

            # 根据特征的每一个取值将样本分离
            arr_aNewX = arr_X[:, arr_col]
            arr_aNewL = arr_L[:, arr_col]

            inst_subNode = self.NodeClass(ins_node.it_deep,arr_aNewL)   # 在确定了选取的特征行后, 该特征的每个取值作为一个子节点
            ins_node.add_subNode(it_featureRow=it_maxGainFeatureRow, it_featureValue=i, inst_subNode=inst_subNode)  # 将子节点连接到父节点上
            self.node({'X': arr_aNewX, 'L': arr_aNewL}, inst_subNode)   # 递归创建节点

    # 计算所有特征纯度的期望
    def all_feature_purity_expect(self,dc_sample):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        it_featureNumSum = arr_X.shape[0]                   # 特征数量
        it_sampleSum = arr_X.shape[1]                       # 样本数量

        arr_allFPurityExpect = n.zeros(it_featureNumSum)    # 记录所有特征后的分割纯度

        # 计算使用某个特征进行分割后的纯度的期望
        for it_featureIndex, arr_aFX in enumerate(arr_X): # it_featureIndex特征索引,即第几个特征  arr_aFX每个样本的该索引特征取值
            set_enableValue = set(arr_aFX)                  # 一个特征的所有可能的取值
            fl_oneFPurityExpect = 0                         # 一个特征的纯度期望

            for i in set_enableValue:                       # 遍历一个特征的每个取值
                arr_colIndex = n.where(arr_aFX == i)[0]     # 拥有该特征取值的样本的索引
                arr_aFaVL = arr_L[:, arr_colIndex]          # 一个特征的一个取值的所有样本的标签集合
                fl_purity = self.purity({'X': None, 'L': arr_aFaVL})    # 一个特征的一个取值的纯度
                it_aFaVNumSum = arr_aFaVL.shape[1]          # 一个特征的一个取值的样本数量
                fl_pro = it_aFaVNumSum / it_sampleSum       # 一个特征的一个取值的样本数量 在 所有样本 中的比例
                fl_oneFPurityExpect += fl_pro * fl_purity   # 一个特征的纯度的期望

            arr_allFPurityExpect[it_featureIndex] = fl_oneFPurityExpect # 所有特征的纯度的期望
        return arr_allFPurityExpect

    # 计算纯度
    def purity(self,dc_sample):
        fl_zero = 1e-6          # 定义一个极小值代替 0

        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        arr_nL = n.sum(arr_L, axis=1)   # 样本每个类别数量
        arr_sumL = arr_nL.sum()         # 样本数量
        arr_proL = arr_nL / arr_sumL    # 样本概率 = 样本每个类别数量 / 样本数量
        arr_proL = n.where(arr_proL < fl_zero, fl_zero, arr_proL)   # 将样本概率中接近 0 的值进行替换, 否则 0*log(0) 将出现问题
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))             # 计算纯度
        return fl_purity

    # 预测
    def predict(self,arr_aX):
        arr_aX = arr_aX.T
        self.pred(self.root,arr_aX)
    def pred(self,node,arr_aX):
        if node.isLeaf():
            print('概率: {}'.format(node.arr_proLabel))
            return
        it_featureRow = node.it_featureRow
        it_featureValue = arr_aX[it_featureRow,0]
        self.pred(node= node.subNode[it_featureValue],arr_aX=arr_aX)

if __name__ == '__main__':
    # arr_X = n.array([
    #     [1, 2, 1, 1],
    #     [1, 2, 1, 3],
    #     [1, 2, 1, 1],
    #     [0, 0, 0, 0],
    # ], dtype=n.int)

    # arr_X = n.array([
    #     [1, 1, 1, 2],
    #     [1, 2, 1, 1],
    #     [1, 1, 1, 1],
    # ])
    # arr_L = n.array([
    #     [1, 0, 1, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    # ])
    # arr_aX = n.array([
    #     [1],
    #     [2],
    #     [1]
    # ])

    arr_X = n.array([
        [1,1,1],
        [1,2,1],
        [1,1,1],
        [2,1,1],
    ])

    arr_L = n.array([
        [1,0,0],
        [0,1,0],
        [1,0,0],
        [0,0,1],
    ])

    T = DTree(arr_X,arr_L)

    arr_aX = n.array([
        [1,2,1]
    ])

    T.predict(arr_aX)








