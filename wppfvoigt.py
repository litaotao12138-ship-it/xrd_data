import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import voigt_profile
from scipy.optimize import curve_fit


#数据类，读取数据，importfile是读取数据
class Data:
    def __init__(self, file):
        self.file = file
        self.refinement_flags = None

    def set_refinement_flags(self, indices, flag):
        if self.refiment_flags is None:
            self.refinement_flags = [True] * len(self.x)
        for index in indices:
            self.refinement_flags[index] = flag
    
    def get_refinable_data(self):
        if self.refinement_flags == None:
            return self.x, self.y
        else:
            return self.x[self.refinement_flags], self.y[self.refinement_flags]
        
    def importfile(self):
        if self.file.split('.')[1] == 'txt':
            df = pd.read_csv(self.file, sep = r'\s+', header = None)
            x, y = np.array(df).T
            return x, y 

class SigmaFitter:
    def __init__(self, centers, sigmas, degree=2):
        self.centers = np.array(centers)
        self.sigmas = np.array(sigmas)
        self.degree = degree
        self.coeffs = np.polyfit(self.centers, self.sigmas, degree)
        self.poly = np.poly1d(self.coeffs)

    def predict(self, center):
        return self.poly(center)

    def get_polynomial(self):
        return self.poly

    def print_polynomial(self):
        print(f"Fitted polynomial (degree {self.degree}):")
        terms = [f"{coef:.6g}*x^{i}" if i > 0 else f"{coef:.6g}" 
                 for i, coef in enumerate(self.coeffs[::-1])]
        poly_str = " + ".join(terms[::-1])
        print("sigma(x) =", poly_str)

    def plot_fit(self, num_points=200):
        plt.figure(figsize=(8,5))
        # 原始数据点
        plt.scatter(self.centers, self.sigmas, color='blue', label='Data Points')
        
        # 拟合曲线
        x_min, x_max = self.centers.min(), self.centers.max()
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = self.poly(x_vals)
        plt.plot(x_vals, y_vals, 'r-', label=f'Poly fit (deg {self.degree})')
        
        plt.xlabel('Center (2θ degrees)')
        plt.ylabel('Sigma')
        plt.title('Sigma vs Center Polynomial Fit')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
class WPPF:
    def __init__(self, filename):
        self.filename = filename
        self.x, self.y = self._import_file()
        self.background = None
        self.y_corrected = None
        self.mask = np.ones_like(self.x, dtype=bool)
        self.fitted_peaks = []

    def _import_file(self):
        if self.filename.split('.')[-1] == 'txt':
            df = pd.read_csv(self.filename, sep=r'\s+', header=None)
            x, y = np.array(df).T
            return x, y
        else:
            raise ValueError("Only .txt files supported")

    def add_mask_region(self, lower, upper):
        self.mask &= ~((self.x > lower) & (self.x < upper))

    def fit_background(self, method='chebyshev', degree=4):
        x_masked = self.x[self.mask]
        y_masked = self.y[self.mask]

        if method == 'chebyshev':
            cheb_fit = Chebyshev.fit(x_masked, y_masked, degree)
            self.background = cheb_fit(self.x)
        else:
            raise ValueError("Only 'chebyshev' method is implemented")

        self.y_corrected = self.y - self.background

    def voigt_area_model(self, x, *params):
        """多个Voigt峰的叠加模型，每四个参数表示一个峰：area, center, sigma, gamma"""
        y = np.zeros_like(x)
        for i in range(0, len(params), 4):
            area, center, sigma, gamma = params[i:i+4]
            y += area * voigt_profile(x - center, sigma, gamma)
        return y

    def fit_multiple_peaks(self, peak_guesses, window=1.5):
        """
        联合拟合多个衍射峰（仅主峰）
        - peak_guesses: 初始Kα1主峰位置列表
        """
        if self.y_corrected is None:
            raise ValueError("Background must be fitted before peak fitting.")
        self.fitted_peaks.clear()

        x_fit_region = []
        y_fit_region = []
        p0 = []
        bounds_lower = []
        bounds_upper = []

        for center in peak_guesses:
            mask = (self.x > center - window) & (self.x < center + window)
            x_fit_region.append(self.x[mask])
            y_fit_region.append(self.y_corrected[mask])
            
            area_guess = np.trapezoid(self.y_corrected[mask], self.x[mask])

            p0 += [area_guess, center, 0.2, 0.2]
            bounds_lower += [0, center - 0.5, 0.001, 0.001]
            bounds_upper += [np.inf, center + 0.5, 5.0, 5.0]

        x_all = np.concatenate(x_fit_region)
        y_all = np.concatenate(y_fit_region)

        try:
            popt, _ = curve_fit(lambda x, *params: self.voigt_area_model(x, *params),
                                x_all, y_all, p0=p0, bounds=(bounds_lower, bounds_upper))

            for i in range(0, len(popt), 4):
                area, center, sigma, gamma = popt[i:i+4]
                fit_curve = self.voigt_area_model(self.x, *popt[i:i+4])
                self.fitted_peaks.append((center, fit_curve, (area, center, sigma, gamma)))

        except RuntimeError:
            print("⚠️ 联合拟合失败，请检查初始参数或窗口设置。")

    def print_peak_parameters(self):
        print("\nFitted Peaks (area-based):")
        for i, (center, _, params) in enumerate(self.fitted_peaks, start=1):
            if params is not None:
                area, center, sigma, gamma = params
                print(f"{i}: center={center:.3f}°, area={area:.3f}, sigma={sigma:.3f}, gamma={gamma:.3f}")
            else:
                print(f"{i}: center={center:.3f}° — fit failed")

    def plot_results(self):
        plt.figure(figsize=(14, 8))
        
        # 原始数据
        plt.plot(self.x, self.y, label='Raw Data', alpha=0.6)

        # 计算总Voigt曲线
        total_voigt = np.sum([fit for _, fit, _ in self.fitted_peaks], axis=0) if self.fitted_peaks else 0

        # 总拟合 = Voigt 总和 + 背景
        if self.background is not None:
            total_fit = total_voigt + self.background
            plt.plot(self.x, total_fit, label='Total Fit', color='red', linewidth=2)

        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.title("XRD Peak Fitting")
        plt.show()
    
    def plot_all_voigt(self):
        plt.figure(figsize=(14, 8))

        # 原始数据
        plt.plot(self.x, self.y, label='Raw Data', color='black', alpha=0.6)

        if self.background is not None:
            background = self.background
        else:
            background = np.zeros_like(self.x)

        # 每个 Voigt 拟合曲线 + 背景
        for i, (center, fit_curve, _) in enumerate(self.fitted_peaks, start=1):
            voigt_with_bg = fit_curve + background
            plt.plot(self.x, voigt_with_bg, label=f'Peak {i} @ {center:.2f}°', linewidth=1)

        # 总拟合曲线
        if self.fitted_peaks:
            total_voigt = np.sum([fit for _, fit, _ in self.fitted_peaks], axis=0)
            total_fit = total_voigt + background
            plt.plot(self.x, total_fit, label='Total Fit (Voigt + Background)', color='red', linewidth=2)

        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.title("XRD Fit with Individual Voigt Peaks + Background")
        plt.show()


------------以下是读取并显示XRD谱 TiO2-------------  
# 如果使用的是vscode，可以使用ctrl + / 来消除或者增加前面的#号键
# data_p = Data('tio2.txt')
# x,y  = data_p.importfile()

# plt.plot(x,y)
# plt.show()

# # ------------以下是读取并显示XRD谱 std.txt-------------
# data_sistd = Data('sistd.txt')
# x,y  = data_sistd.importfile()

# plt.plot(x,y)
# plt.show()

# # ------------------ 使用示例 std.txt,全谱拟合，未考虑晶体结构------------------
# if __name__ == "__main__":
#     analyzer = XRDAnalyzer("sistd.txt")

#     analyzer.add_mask_region(27, 30)
#     analyzer.add_mask_region(46, 49)
#     analyzer.add_mask_region(55, 58)
#     analyzer.add_mask_region(68, 71)
#     analyzer.add_mask_region(75, 78)
#     analyzer.add_mask_region(87, 90)
#     analyzer.add_mask_region(93, 95)


#     analyzer.fit_background(method='chebyshev', degree=4)

#     diffraction_peaks = [28.5, 47.4, 56.1, 69.2, 76.5, 88.1, 94.9]
#     analyzer.fit_multiple_peaks(diffraction_peaks, window=3.0)

#     analyzer.print_peak_parameters()
#     analyzer.plot_all_voigt()


# # ------------------ 使用示例 拟合硅标半峰宽分析------------------
# # 你的拟合数据, 这是利用使用示例 std.txt,全谱拟合，未考虑晶体结构此处拟合出来的数据，将第一列衍射角和第三列sigma转化为两个list
centers = [28.402, 47.268, 56.088, 69.099, 76.347, 88.000, 94.920]
sigmas =  [0.023,  0.022,  0.020,  0.022,  0.021,  0.021,  0.026]

# 实例化拟合器，选择多项式阶数
sigma_fitter = SigmaFitter(centers, sigmas, degree=2)

# 打印拟合公式
sigma_fitter.print_polynomial()

# 绘制拟合曲线和数据点
sigma_fitter.plot_fit()

# 预测一个新的 sigma 值
two_diffraction_list = [25.272, 37.836]
for two_diffraction in two_diffraction_list:
    predicted_sigma = sigma_fitter.predict(two_diffraction)
    print(f"\nPredicted sigma at center={two_diffraction:.3f}°: {predicted_sigma:.6f}")


# # # ------------------ 使用示例 ------------------
# if __name__ == "__main__":
#     analyzer = WPPF("tio2.txt")

#     analyzer.add_mask_region(23, 27)
#     analyzer.add_mask_region(35, 40)
#     analyzer.add_mask_region(46, 50)
#     analyzer.add_mask_region(51, 57)
#     analyzer.add_mask_region(60, 65)
#     analyzer.add_mask_region(66, 71)
#     analyzer.add_mask_region(73, 77)

#     analyzer.fit_background(method='chebyshev', degree=4)

#     diffraction_peaks = [25.3, 37.0, 37.7, 38.5, 48.0, 53.8, 55.0, 62.7, 68.7,70.3, 75.0, 76.0]
#     analyzer.fit_multiple_peaks(diffraction_peaks, window=3.0)

#     analyzer.print_peak_parameters()
#     analyzer.plot_all_voigt()
