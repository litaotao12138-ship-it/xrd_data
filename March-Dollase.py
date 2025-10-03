import numpy as np
from itertools import permutations, product

class MarchDollaseCalculator:
    def __init__(self, n0=[0,0,1], r=1.0):
        """
        n0: 择优方向单位向量
        r: March-Dollase 择优参数
        """
        self.n0 = np.array(n0) / np.linalg.norm(n0)
        self.r = r

    @staticmethod
    def generate_equivalent_planes(hkl):
        hkl = np.array(hkl)
        # 唯一排列
        perms = set(permutations(hkl))
        equiv_planes = set()
        for p in perms:
            # 找非零元素的位置
            nonzero_idx = [i for i, x in enumerate(p) if x != 0]
            # 符号组合只对非零元素
            signs = list(product([-1, 1], repeat=len(nonzero_idx)))
            for s in signs:
                plane = list(p)
                for idx, sign in zip(nonzero_idx, s):
                    plane[idx] *= sign
                equiv_planes.add(tuple(plane))
        return [np.array(plane) for plane in equiv_planes]

    @staticmethod
    def march_dollase_single(hkl, n0, r):
        """
        单个晶面的 March-Dollase 因子
        """
        hkl = np.array(hkl)
        n0 = np.array(n0) / np.linalg.norm(n0)
        v = hkl / np.linalg.norm(hkl)
        cos_alpha = np.abs(np.dot(v, n0))
        cos2 = cos_alpha**2
        sin2 = 1 - cos2
        P = (r**2 * cos2 + r**(-1) * sin2) ** (-3/2)
        return P

    def march_dollase(self, hkl, use_equivalent=True):
        """
        计算晶面 March-Dollase 因子
        hkl: 晶面索引 [h,k,l]
        use_equivalent: 是否考虑等效平面集合（默认 True）
        """
        if use_equivalent:
            planes = self.generate_equivalent_planes(hkl)
            P_list = [self.march_dollase_single(plane, self.n0, self.r) for plane in planes]
            return np.mean(P_list)
        else:
            return self.march_dollase_single(hkl, self.n0, self.r)


# # 择优方向沿 z
calculator = MarchDollaseCalculator(n0=[0,0,1], r=1.5)

# # 单个晶面
P_single = calculator.march_dollase([1,1,0], use_equivalent=False)
print("Single March-Dollase P =", P_single)

# 考虑等效晶面集合
P_avg = calculator.march_dollase([1,1,0], use_equivalent=True)
print("mulit March-Dollase P =", P_avg)

# 处理多个晶面列表
hkl_list = [[1,1,0], [2,0,0], [2,1,1], [2,2,0]]
P_list = [calculator.march_dollase(hkl) for hkl in hkl_list]
print("Multi plane March-Dollase P =", P_list)



# import numpy as np

# def march_dollase(hkl, n0, r):
#     """
#     March-Dollase 择优因子函数 (独立函数)
    
#     参数：
#     -----------
#     hkl : array_like
#         晶面索引 [h, k, l]
#     n0 : array_like
#         择优方向单位向量，例如 [0,0,1]
#     r : float
#         March-Dollase 择优参数
        
#     返回：
#     -----------
#     P : float
#         晶面择优因子
#     """
#     hkl = np.array(hkl)
#     n0 = np.array(n0) / np.linalg.norm(n0)  # 确保单位向量
#     v = hkl / np.linalg.norm(hkl)           # 晶面法向量归一化

#     cos_alpha = np.abs(np.dot(v, n0))       # α 与择优方向夹角
#     cos2 = cos_alpha**2
#     sin2 = 1 - cos2

#     P = (r**2 * cos2 + r**(-1) * sin2) ** (-3/2)
#     return P

# P = march_dollase([1,1,0], [0,0,1], 1.2)
# print("March-Dollase P =", P)
