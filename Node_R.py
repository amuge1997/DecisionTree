class Node_R:
    def __init__(self):
        self.fl_split = None
        self.it_fea = None
        self.fl_pre = None
        self.tree_left = None
        self.tree_righ = None
        self.bl_leaf = False

    def set_fea(self,it_fea):
        self.it_fea = it_fea

    def set_split(self,fl_split):
        self.fl_split = fl_split

    def set_pre(self,fl_pre):
        self.fl_pre = fl_pre

    def set_left(self,tree):
        self.tree_left = tree

    def set_righ(self,tree):
        self.tree_righ = tree

    def set_leaf(self):
        self.bl_leaf = True

















