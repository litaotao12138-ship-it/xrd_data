import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 峰形函数 (修正版)
# ----------------------
def PVf(x, x0, fwhm=0.5, eta=0.5):
    """Pseudo-Voigt profile"""
    eta = min(max(eta, 0.0), 1.0)   # 保证 0 ≤ eta ≤ 1
    gamma = fwhm / 2.0              # Lorentz 半宽
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))  # Gauss σ

    L = 1 / (1 + ((x - x0)/gamma)**2)
    G = np.exp(-(x - x0)**2 / (2*sigma**2))
    return eta * L + (1 - eta) * G


# ----------------------
# Rietveld类
# ----------------------
class CalTheProfile:
    def __init__(self, a=2.87, lmda=1.54, r=1.0, n=(0,0,1)):
        self.a = a
        self.lmda = lmda
        self.r = r         # March-Dollase 因子 (r=1 表示无择优取向)
        self.n = np.array(n) / np.linalg.norm(n)  # 择优取向方向单位向量

    # 计算 d 间距
    @staticmethod
    def cal_d_hkl(index, a=2.866):
        h, k, l = index
        return a / np.sqrt(h**2 + k**2 + l**2)

    # 计算 2θ
    def cal_2theta(self, index):
        d = self.cal_d_hkl(index, self.a)
        theta = np.degrees(np.arcsin(self.lmda / (2 * d)))
        return 2 * theta

    # 结构因子 (简化版)
    def ffe(self, index):
        a = [11.7695, 7.3573, 3.5222, 2.3045]
        b = [4.7611, 0.3072, 15.3535, 76.8805]
        c = 1.0369
        d = self.cal_d_hkl(index, self.a)
        f = sum(ai * np.exp(-bi * (0.5 / d) ** 2) for ai, bi in zip(a, b)) + c
        return f

    # FWHM (Caglioti公式)
    @staticmethod
    def FWHM(theta, u=0.29, v=-0.12, w=0.3, ig=0):
        rad = np.radians(theta)
        val = u * np.tan(rad) ** 2 + v * np.tan(rad) + w + ig / (np.cos(rad) ** 2)
        return max(np.sqrt(val), 0.05)   # 保证最小宽度 0.05°

    # Lorentz-Polarization因子
    @staticmethod
    def Lp(two_theta):
        rad = np.radians(two_theta)
        return (1 + np.cos(rad)**2) / (np.sin(rad/2)**2 * np.abs(np.cos(rad)))

    # eta 参数
    @staticmethod
    def Eta(two_theta, eta_0=0.3, X=0.18):
        return min(max(eta_0 + X * np.radians(two_theta), 0.0), 1.0)

    # March–Dollase 因子
    def MarchDollase(self, hkl):
        hkl = np.array(hkl, dtype=float)
        hkl = hkl / np.linalg.norm(hkl)   # 单位化
        cosphi = np.dot(hkl, self.n)
        cos2 = cosphi**2
        r2 = self.r**2
        denom = cos2 + (1 - cos2) * r2
        return (r2 * cos2 + (1 - cos2)/r2)**(-1.5) / denom

    # 理论强度计算 (叠加峰)
    def calc_pattern(self, x, hkl_list, mult_list):
        intensities = []
        for hkl, mult in zip(hkl_list, mult_list):
            f = self.ffe(hkl)
            I0 = f**2 * mult
            two_theta = self.cal_2theta(hkl)
            width = self.FWHM(two_theta / 2)
            eta = self.Eta(two_theta)
            lp = self.Lp(two_theta)
            md = self.MarchDollase(hkl)
            peak = I0 * lp * md * PVf(x, two_theta, width, eta)
            intensities.append(peak)
        return np.sum(intensities, axis=0)

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    # 设置角度范围 (10-105°, 步长 0.01°)
    x = np.arange(10, 105.01, 0.01)

    # 定义晶格和指数
    rietveld = CalTheProfile(a=2.87, lmda=1.5406, r=0.7, n=(0,0,1))  # r<1 扁平择优取向, r>1 拉伸
    hkl_list = [[1,1,0],[2,0,0],[2,1,1],[2,2,0]]
    mult_list = [12,6,24,12]  # 多重性

    # 计算理论曲线
    y_calc = rietveld.calc_pattern(x, hkl_list, mult_list)

    # ----------------------
    # 导出数据：逐点输出，tab 分隔
    # ----------------------
    np.savetxt("fe.txt", np.column_stack((x, y_calc)),
               delimiter="\t",  comments="", fmt="%.2f")

    # 绘图 (未归一化)
    plt.figure(figsize=(10,5))
    plt.ylabel("Intensity")
    plt.legend(["Theo (MD)"])

    plt.plot(x, y_calc, label="XRD profile", lw=1.2)
    plt.xlabel("2θ (°)")
    plt.tight_layout()
    plt.show()
