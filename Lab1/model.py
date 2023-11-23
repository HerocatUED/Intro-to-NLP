import math
from feature import extract_feature


class LogLinear:
    def __init__(self, length: int, class_num: int):
        self.class_num = class_num  # 类别数量
        self.length = length  # 参数规模
        self.theta = [[1e-6 for _ in range(length)]
                      for _ in range(self.class_num)]  # 模型参数

    def predict(self, x: str):
        score = [0]*self.class_num
        for i in range(self.class_num):
            f = extract_feature(x)
            score[i] = my_dot(self.theta[i], f)
        return score.index(max(score))

    def cal_grad(self, x: str, label: int):
        """ 对单个样本xk求grad """
        feature = extract_feature(x)
        dots = [my_dot(self.theta[i], feature) for i in range(self.class_num)]
        exps = [math.exp(dots[i]) for i in range(self.class_num)]
        exps_sum = sum(exps)
        exps_normalized = [exps[i]/exps_sum for i in range(self.class_num)]
        grads = []
        for j in range(self.class_num):
            grad = []
            s = exps_normalized[j]
            for item in feature:
                index = item[0]
                weight = item[1]
                if j == label:
                    grad.append((index, weight*(1-s)))
                else:
                    grad.append((index, weight*(-s)))
            grads.append(grad)
        return grads

    def update(self, epoch: list, lr: float = 0.02, alpha: float = 0.001, max_loop: int = 3):
        """ 更新模型 """
        epoch_len = len(epoch)
        for _ in range(max_loop):
            # 计算这个epoch的grad
            all_grads = [self.cal_grad(epoch[i][0], epoch[i][1])
                         for i in range(epoch_len)]
            for clazz in range(self.class_num):
                indices = set()
                for i in range(epoch_len):
                    grad = all_grads[i][clazz]
                    for item in grad:
                        index = item[0]
                        indices.add(index)
                        delta = item[1]
                        self.theta[clazz][index] += lr*delta
                for index in indices:
                    self.theta[clazz][index] -= alpha*self.theta[clazz][index]


def my_dot(theta: list, feature: list):
    """ 计算稀疏点乘 """
    ans = 0
    for item in feature:
        index = item[0]
        weight = item[1]
        ans += weight*theta[index]
    return ans
