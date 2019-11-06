import numpy as n



class DDTree:
    # 离散决策树

    class NodeClass:
        def __init__(self,it_deep):
            self.subNode = {}
            self.it_featureRow = None
            self.it_deep = it_deep + 1

        def add_subNode(self,it_featureRow,it_featureValue,inst_subNode):
            self.it_featureRow = it_featureRow
            self.subNode[it_featureValue] = inst_subNode

        def isLeaf(self):
            if len(self.subNode) == 0:
                return True
            else:
                return False
    def walk(self,node):
        if node.isLeaf():
            return
        for it_featureValue,ins_subNode in node.subNode.items():
            print('进入{}节点'.format(it_featureValue))
            self.walk(ins_subNode)


    def __init__(self):
        arr_X = n.array([
            [1, 2, 1, 1],
            [1, 2, 1, 3],
            [1, 2, 1, 1],
            [0, 0, 0, 0],
        ], dtype=n.int)
        arr_L = n.array([
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        self.root = self.NodeClass(0)
        self.node({'X':arr_X,'L':arr_L},self.root)

    def node(self,dc_sample,ins_node):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        # 计算当前节点的纯度
        fl_nowPurity = self.purity({'X': None, 'L': arr_L})
        # 计算所有特征分割后的纯度
        arr_allFeaturePurityExpect = self.all_feature_purity_expect({'X': arr_X, 'L': arr_L})
        # 计算信息增益
        fl_gain = fl_nowPurity - arr_allFeaturePurityExpect

        fl_maxGin =fl_gain.max()
        if fl_maxGin<0.1:
            print(fl_maxGin)
            return
        it_maxGainRow = fl_gain.argmax()

        # 删去信息增益最大的特征
        arr_newX = n.delete(arr_X, it_maxGainRow, axis=0)

        arr_maxGainArrX = arr_X[it_maxGainRow]
        set_ = set(arr_maxGainArrX)
        for i in set_:
            arr_col = n.where(arr_maxGainArrX == i)[0]
            arr_aNewX = arr_newX[:, arr_col]
            arr_aNewL = arr_L[:, arr_col]

            inst_subNode = self.NodeClass(ins_node.it_deep)
            ins_node.add_subNode(it_featureRow=it_maxGainRow, it_featureValue=i, inst_subNode=inst_subNode)
            self.node({'X': arr_aNewX, 'L': arr_aNewL}, inst_subNode)


    def all_feature_purity_expect(self,dc_sample):
        arr_X = dc_sample['X']
        arr_L = dc_sample['L']

        it_featureNumSum = arr_X.shape[0]
        it_sampleSum = arr_X.shape[1]
        # 记录所有特征后的分割纯度
        arr_allFPurityExpect = n.zeros(it_featureNumSum)
        for it_featureNum, arr_aFX in enumerate(arr_X):
            # 计算使用某个特征进行分割后的纯度的期望
            set_ = set(arr_aFX)
            fl_aFPurityExpect = 0
            for i in set_:
                arr_col = n.where(arr_aFX == i)[0]
                # 一个特征的一个取值的所有样本的标签集合
                arr_aFaVL = arr_L[:, arr_col]
                fl_purity = self.purity({'X': None, 'L': arr_aFaVL})
                it_aFaVNumSum = arr_aFaVL.shape[1]
                # 一个特征的一个取值的所有样本的标签集合 占 所有样本 的比例
                fl_pro = it_aFaVNumSum / it_sampleSum
                # 一个特征的纯度的期望
                fl_aFPurityExpect += fl_pro * fl_purity
            # print(fl_aFPurityExpect)
            arr_allFPurityExpect[it_featureNum] = fl_aFPurityExpect
        # print(arr_allFPurityExpect)
        return arr_allFPurityExpect

    def purity(self,dc_sample):
        fl_zero = 1e-6

        arr_X = dc_sample['X']
        arr_L = dc_sample['L']
        # 统计标签占比
        arr_nL = n.sum(arr_L, axis=1)
        arr_sumL = arr_nL.sum()
        arr_proL = arr_nL / arr_sumL
        # 由于浮点数 0*log0 并非为 0 ,因此定义一个极小值代替 0
        arr_proL = n.where(arr_proL < fl_zero, fl_zero, arr_proL)
        # 计算纯度
        fl_purity = - n.sum(arr_proL * n.log(arr_proL))
        return fl_purity


if __name__ == '__main__':
    T = DDTree()
    T.walk(T.root)









