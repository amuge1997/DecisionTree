import numpy as np
from tkinter import *

class DTreeDraw:
    def __init__(self,ins_Tree):
        self.DTree = ins_Tree

        self.tk = Tk()                                      # 主控件
        self.tk.resizable(width=False,height=False)         # 不可缩放
        self.cv = Canvas(self.tk,background='#afafaf', width=1000, height=900)  # 画布
        self.cv.pack()

    # 决策树可视化
    def drawTree(self):
        self.drawWalk(self.DTree.root, {'x': 0, 'y': 0},{'x':0,'y':0})

    def drawWalk(self, node, node_xy, node_lxy):
        x, y = node_xy['x'], node_xy['y']

        it_wid = 170
        it_hei = 60
        it_xoffset = 100
        it_yoffset = 70
        it_drawSrcX = x*it_wid + it_xoffset         # 节点参考点坐标X
        it_drawSrcY = y*it_hei + it_yoffset         # 节点参考点坐标Y
        it_rectangleX = it_drawSrcX+it_wid-40       # 矩形右下角坐标X
        it_rectangleY = it_drawSrcY+it_hei          # 矩形右下角坐标Y

        self.cv.create_rectangle(it_drawSrcX,it_drawSrcY,it_rectangleX,it_rectangleY)           # 节点矩形
        sr_pro = '概率: {}'.format(np.round(node.arr_proLabel,2))                               # 概率
        sr_fea = '特征: {}'.format(node.it_selfFeatureRow)                                      # 特征(行)
        sr_val = '取值: {}'.format(node.it_selfFeatureVal)                                      # 特征(行)的取值
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 5, text=sr_fea, anchor='nw')
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 20, text= sr_val, anchor='nw')
        self.cv.create_text(it_drawSrcX + 10, it_drawSrcY + 35, text= sr_pro, anchor='nw')

        tp_thisxy = (it_drawSrcX,it_drawSrcY)       # 当前节点坐标
        tp_lastxy = (node_lxy['x'],node_lxy['y'])   # 父节点坐标
        self.cv.create_line([tp_lastxy,tp_thisxy],arrow=LAST)

        node_lxy = {'x':it_rectangleX,'y':it_drawSrcY}
        if node.isLeaf():
            return node_xy['y']             # 如果是叶子节点则返回当前高度
        x += 1
        for k, subNode in node.subNode.items():
            y = self.drawWalk(subNode, {'x': x, 'y': y},node_lxy)
            y += 1
        return y - 1                        # 返回当前高度

    # 打开主循环进行显示
    def show(self):
        self.tk.mainloop()























