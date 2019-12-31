# http://codecrafthouse.jp/p/2014/09/decision-tree/
import numpy as np
from sklearn import datasets

class _Node:
  def __init__(self):
    self.left = None
    self.right = None
    self.feature = None
    self.threthold = None
    self.label = None
    self.numdata = None # データ数
    self.gini_index = None

  # 木の構築を行う
  def build(self, data, target):
    self.numdata = data.shape[0]
    num_features = data.shape[1]

    # 全データが同一クラスになったら終了
    if len(np.unique(target)) == 1:
      self.label = target[0]
      return

    # 一番多いラベル
    class_cnt = {i: len(target[target == i]) for i in np.unique(target)}
    self.label = max(class_cnt.items(), key=lambda x:x[1])[0]

    best_gini_index = 0.0
    best_feature = None
    best_threshold = None

    gini = self.gini_func(target)

    for f in range(num_features):
      data_f = np.unique(data[:, f])
      points = (data_f[:-1] + data_f[1:]) / 2.0

      for p in points:
        target_l = target[data[:, f] < p]
        target_r = target[data[:, f] >= p]

        gini_l = self.gini_func(target_l)
        gini_r = self.gini_func(target_r)
        pl = float(target_l.shape[0]) / self.numdata
        pr = float(target_r.shape[0]) / self.numdata
        gini_index = gini - (pl * gini_l + pr * gini_r)

        if gini_index > best_gini_index:
          best_gini_index = gini_index
          best_feature = f
          best_threshold = p

    # 不純度が低くならなければ終了
    if best_gini_index == 0:
      return

    self.gini_index = best_gini_index
    self.feature = best_feature
    self.threthold = best_threshold

    data_l = data[data[:, self.feature] < self.threthold]
    target_l = target[data[:, self.feature] < self.threthold]
    self.left = _Node()
    self.left.build(data_l, target_l)

    data_r = data[data[:, self.feature] >= self.threthold]
    target_r = target[data[:, self.feature] >= self.threthold]
    self.right = _Node()
    self.right.build(data_r, target_r)

  # Gini関数を計算
  def gini_func(self, target):
    classes = np.unique(target)
    numdata = target.shape[0]

    gini = 1.0
    for c in classes:
      gini -= (len(target[target == c]) / numdata) ** 2

    return gini

  # 剪定する
  def prune(self, criterion, numall):
    if self.feature == None:
      return

    self.left.prune(criterion, numall)
    self.right.prune(criterion, numall)

    if self.left.feature == None and self.right.feature == None:
      result = self.gini_index * float(self.numdata) / numall
      if result < criterion:
        self.feature = None
        self.left = None
        self.right = None

  # 入力データの分類クラスを返す
  def predict(self, d):
    if self.feature == None:
      return self.label

    if d[self.feature] < self.threthold:
      return self.left.predict(d)
    else:
      return self.right.predict(d)

  def print_tree(self, depth, TF):
    head = "    " * depth + TF + " -> "

    # 節の場合
    if self.feature != None:
      print(head + str(self.feature) + " < " + str(self.threthold) + "?")
      self.left.print_tree(depth + 1, "T")
      self.right.print_tree(depth + 1, "F")

    # 葉の場合
    else:
      print(head + "{" + str(self.label) + ": " + str(self.numdata) + "}")

class DecisionTree:
  def __init__(self, criterion=0.1):
    self.root = None
    self.criterion = criterion

  def fit(self, data, target):
    self.root = _Node()
    self.root.build(data, target)
    self.root.prune(self.criterion, self.root.numdata)

  def predict(self, data):
    ans = []
    for d in data:
      ans.append(self.root.predict(d))
    return np.array(ans)

  def print_tree(self):
    self.root.print_tree(0, " ")

def main():
  iris = datasets.load_iris()
  tree = DecisionTree()
  tree.fit(iris.data, iris.target)
  tree.print_tree()

if __name__ == "__main__":
    main()
