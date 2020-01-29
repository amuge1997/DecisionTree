import numpy as np
from DecisionTree.Node_R import Node_R

class CARTree_R:
    def __init__(self):
        pass

    def init(self):
        self.node_root = None

    def fit(self,arr_X,arr_Y):
        arr_X = arr_X.T
        arr_Y = arr_Y.T
        self.init()
        self.node_root = Node_R()
        self.it_feaSum = arr_X.shape[0]
        self.node(self.node_root,arr_X,arr_Y)

    def stop(self,arr_X=None,it_deep=None):
        if arr_X is not None and arr_X.shape[1]<=1:
            return True
        elif it_deep is not None and it_deep >= 5:
            return True
        return False

    def node(self,node_parent,arr_X,arr_Y):
        # 预测值
        fl_pre = np.mean(arr_Y)
        node_parent.set_pre(fl_pre)
        # print(node_parent.fl_pre)

        if self.stop(arr_X=arr_X):
            # print('样本过少')
            node_parent.set_leaf()
            return

        # 计算分割点
        # 对每个特征计算最优分割点,对每个特征的分割点求最优分割点
        it_feaIdx,fl_split = self.find_best_spilt(arr_X,arr_Y)
        if it_feaIdx is None:
            # print('结束')
            node_parent.set_leaf()
            return
        node_parent.set_fea(it_feaIdx)
        node_parent.set_split(fl_split)

        # 左分割
        arr_lessIdx = np.where(arr_X[it_feaIdx] <= fl_split)[0]
        arr_leftX = arr_X[:,arr_lessIdx]
        arr_leftY = arr_Y[:,arr_lessIdx]
        node_left = Node_R()
        node_parent.set_left(node_left)
        self.node(node_left,arr_leftX,arr_leftY)

        # 右分割
        arr_moreIdx = np.where(arr_X[it_feaIdx] > fl_split)[0]
        arr_righX = arr_X[:,arr_moreIdx]
        arr_righY = arr_Y[:,arr_moreIdx]
        node_righ = Node_R()
        node_parent.set_righ(node_righ)
        self.node(node_righ,arr_righX,arr_righY)

    def find_best_spilt(self,arr_X,arr_Y):
        # 输入 arr_X,arr_Y

        arr_minSplit = np.zeros((self.it_feaSum))
        arr_minError = np.zeros((self.it_feaSum))
        for it_feaidx in range(self.it_feaSum):
            fl_split,fl_error = self.find_best_spilt_from_fea(it_feaidx,arr_X,arr_Y)
            arr_minSplit[it_feaidx] = fl_split
            arr_minError[it_feaidx] = fl_error

        it_minErrorIdx = np.argmin(arr_minError)
        it_minFea = it_minErrorIdx
        fl_split = arr_minSplit[it_minFea]

        fl_max = np.max(arr_X[it_minFea])
        fl_min = np.min(arr_X[it_minFea])

        if fl_split == fl_max or fl_split == fl_min:
            it_minFea = None
            fl_split = None

        # 返回 it_feature,fl_split
        return it_minFea,fl_split

    def find_best_spilt_from_fea(self,it_feaidx,arr_X,arr_Y):
        arr_feaX = arr_X[it_feaidx,:]
        arr_sortX = np.sort(arr_feaX)
        arr_unqiX = np.unique(arr_sortX)

        arr_temp1 = arr_unqiX[1:]
        arr_temp2 = arr_unqiX[:-1]
        arr_temp3 = (arr_temp1 + arr_temp2)/2
        arr_temp4 = np.append([arr_unqiX[0]],arr_temp3)
        arr_split = np.append(arr_temp4,[arr_unqiX[-1]])
        arr_bestSplit = np.zeros((arr_split.shape))

        for it_idx,fl_split in enumerate(arr_split):
            arr_lessYidx = np.where(arr_feaX <= fl_split)
            arr_moreYidx = np.where(arr_feaX > fl_split)

            arr_lessY = arr_Y[0,arr_lessYidx[0]]
            arr_moreY = arr_Y[0,arr_moreYidx[0]]
            fl_error = self.cal_error(arr_lessY,arr_moreY)

            arr_bestSplit[it_idx] = fl_error
        fl_minError = np.min(arr_bestSplit)
        fl_minErrorIdx = np.argmin(arr_bestSplit)
        fl_minSplit = arr_split[fl_minErrorIdx]
        # 返回 fl_split
        return fl_minSplit,fl_minError

    def cal_error(self,arr_Y1,arr_Y2):
        if arr_Y1.shape[0] == 0:
            fl_error = np.var(arr_Y2)
        elif arr_Y2.shape[0] == 0:
            fl_error = np.var(arr_Y1)
        else:
            fl_error = np.var(arr_Y1) + np.var(arr_Y2)
        return fl_error

    def predict(self,arr_X):
        def pre(node,arr_xt):
            if not node.bl_leaf:
                it_fea = node.it_fea
                fl_split = node.fl_split
                node_left = node.tree_left
                node_righ = node.tree_righ
                if arr_xt[it_fea] <= fl_split:
                    return pre(node_left,arr_xt)
                elif arr_xt[it_fea] > fl_split:
                    return pre(node_righ,arr_xt)
            else:
                return node.fl_pre
        it_XSum = arr_X.shape[0]
        arr_Y = np.zeros((it_XSum,1))
        for it_idx in range(it_XSum):
            arr_x = arr_X[it_idx,:].reshape(-1,1)
            arr_Y[it_idx] = pre(self.node_root,arr_x)

        return arr_Y


if __name__ == '__main__':

    # arr_X = np.array([
    #     [0,0],
    #     [1,6],
    #     [2,2],
    #     [3,3],
    #     [4,4],
    #     [5,5],
    # ])
    # arr_Y = np.array([
    #     [1],
    #     [0.5],
    #     [1],
    #     [2],
    #     [2.5],
    #     [2],
    # ])
    #
    # arr_Xt = np.array([
    #     [1,6.5],
    #     [3.5,3]
    # ])

    arr_X = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ]).T
    arr_Y = np.array([
        [1, 0.5, 1, 2, 2.5, 2, 4, 4.5, 4]
    ]).T

    arr_Xt = np.linspace(-1,10,100).reshape(-1,1)

    ct = CARTree_R()
    ct.fit(arr_X=arr_X,arr_Y=arr_Y)
    arr_Yt = ct.predict(arr_Xt)
    print(arr_Yt)

    import matplotlib.pyplot as plt

    plt.scatter(arr_X,arr_Y,c='red')
    plt.plot(arr_Xt,arr_Yt,c='blue')
    plt.grid()
    plt.show()













