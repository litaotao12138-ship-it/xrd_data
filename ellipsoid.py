# 完整 XRDAnalyzer（含椭球拟合与可视化）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import voigt_profile
from scipy.optimize import curve_fit, least_squares
from mpl_toolkits.mplot3d import Axes3D


class XRDAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.x, self.y = self._import_file()
        self.background = None
        self.y_corrected = None
        self.mask = np.ones_like(self.x, dtype=bool)
        self.fitted_peaks = []  # (center, fit_curve_on_full_x, params)
        # Ellipsoid defaults (units: user must set wavelength & axes in same length units, e.g. nm)
        self.axes = np.array([10.0, 10.0, 10.0], dtype=float)
        self.orientation = np.eye(3)
        self.lattice_params = None
        self.wavelength = None
        self.hkl_list = None

    # ---------- I/O & preprocessing ----------
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
            raise ValueError("Only 'chebyshev' is implemented")
        self.y_corrected = self.y - self.background

    # ---------- Voigt multi-peak ----------
    def voigt_area_model(self, x, *params):
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 4):
            area, center, sigma, gamma = params[i:i+4]
            y += area * voigt_profile(x - center, sigma, gamma)
        return y

    def fit_multiple_peaks(self, peak_guesses, window=1.5):
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
            if mask.sum() < 5:
                continue
            x_fit_region.append(self.x[mask])
            y_fit_region.append(self.y_corrected[mask])

            # ✅ 兼容新旧 NumPy 的 trapezoid 写法
            if hasattr(np, "trapezoid"):
                area_guess = np.trapezoid(self.y_corrected[mask], self.x[mask])
            else:
                area_guess = np.trapz(self.y_corrected[mask], self.x[mask])

            p0 += [max(area_guess, 1e-6), center, 0.2, 0.2]
            bounds_lower += [0.0, center - 0.5, 0.001, 0.001]
            bounds_upper += [np.inf, center + 0.5, 5.0, 5.0]

        if not x_fit_region:
            raise RuntimeError("No fitting regions formed — check peak_guesses/window")

        x_all = np.concatenate(x_fit_region)
        y_all = np.concatenate(y_fit_region)

        try:
            popt, _ = curve_fit(lambda x, *params: self.voigt_area_model(x, *params),
                                x_all, y_all, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=20000)

            for i in range(0, len(popt), 4):
                params_i = popt[i:i+4]
                area, center, sigma, gamma = params_i
                fit_curve = self.voigt_area_model(self.x, *params_i)
                self.fitted_peaks.append((center, fit_curve, tuple(params_i)))
        except RuntimeError as e:
            print("⚠️ 联合拟合失败：", str(e))

    def print_peak_parameters(self):
        print("\nFitted Peaks (area-based):")
        for i, (center, _, params) in enumerate(self.fitted_peaks, start=1):
            if params is not None:
                area, center, sigma, gamma = params
                print(f"{i}: center={center:.4f}°, area={area:.1f}, sigma={sigma:.4f}, gamma={gamma:.4f}")
            else:
                print(f"{i}: center={center:.4f}° — fit failed")

    def plot_all_voigt(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.x, self.y, label='Raw Data', color='black', alpha=0.6)
        background = self.background if self.background is not None else np.zeros_like(self.x)
        for i, (center, fit_curve, _) in enumerate(self.fitted_peaks, start=1):
            plt.plot(self.x, fit_curve + background, label=f'Peak {i} @ {center:.2f}°')
        if self.fitted_peaks:
            total_voigt = np.sum([fit for _, fit, _ in self.fitted_peaks], axis=0)
            plt.plot(self.x, total_voigt + background, label='Total Fit', color='red', linewidth=2)
        plt.xlabel('2θ (°)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.title("Voigt Peaks + Background")
        plt.show()

    # ---------- 辅助：椭球 & 晶格 ----------
    def set_ellipsoid_axes(self, axes):
        self.axes = np.array(axes, dtype=float)

    def set_orientation(self, R):
        R = np.array(R, dtype=float)
        if R.shape != (3,3):
            raise ValueError("orientation must be 3x3")
        self.orientation = R

    def set_lattice_params(self, lattice_params):
        self.lattice_params = lattice_params

    def set_wavelength(self, wavelength):
        """wavelength must be in same length unit as axes (e.g. nm)"""
        self.wavelength = float(wavelength)

    def set_hkl(self, hkl_list):
        self.hkl_list = hkl_list

    @staticmethod
    def fwhm_voigt_from_sigma_gamma(sigma, gamma):
        fL = 2.0 * gamma
        fG = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)

    @staticmethod
    def deconvolve_instrument(beta_meas, beta_inst, profile='gaussian'):
        beta_meas = float(beta_meas)
        beta_inst = float(beta_inst)
        if profile.lower() == 'gaussian':
            val = beta_meas**2 - beta_inst**2
            return np.sqrt(max(0.0, val))
        elif profile.lower() == 'lorentzian':
            val = beta_meas - beta_inst
            return max(0.0, val)
        else:
            raise ValueError("profile must be 'gaussian' or 'lorentzian'")

    @staticmethod
    def _deg2rad(x): return x * np.pi/180.0

    def _direct_lattice_matrix(self):
        if self.lattice_params is None:
            raise ValueError("lattice_params must be set")
        la = self.lattice_params
        a,b,c = float(la['a']), float(la['b']), float(la['c'])
        alpha = self._deg2rad(la.get('alpha',90.0))
        beta  = self._deg2rad(la.get('beta',90.0))
        gamma = self._deg2rad(la.get('gamma',90.0))
        vx = np.array([a, 0.0, 0.0])
        vy = np.array([b*np.cos(gamma), b*np.sin(gamma), 0.0])
        cz = c * np.array([np.cos(beta),
                           (np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
                           np.sqrt(max(0.0, 1 - np.cos(beta)**2 - ((np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2))])
        A = np.vstack([vx, vy, cz]).T
        return A

    def reciprocal_direction_from_hkl(self, hkl):
        A = self._direct_lattice_matrix()
        B = 2.0 * np.pi * np.linalg.inv(A).T
        h,k,l = map(float, hkl)
        G = h*B[:,0] + k*B[:,1] + l*B[:,2]
        return G / np.linalg.norm(G)

    def d_spacing_from_hkl(self, hkl):
        A = self._direct_lattice_matrix()
        B = 2.0 * np.pi * np.linalg.inv(A).T
        h,k,l = map(float, hkl)
        G = h*B[:,0] + k*B[:,1] + l*B[:,2]
        return 2.0 * np.pi / np.linalg.norm(G)

    def bragg_theta_from_d(self, d, wavelength):
        arg = wavelength / (2.0 * d)
        if abs(arg) > 1.0:
            raise ValueError("No Bragg reflection for given lambda and d.")
        return np.arcsin(arg)

    def effective_thickness_L_from_direction(self, n_crystal):
        n = np.array(n_crystal, dtype=float)
        n = n / np.linalg.norm(n)
        n_ell = self.orientation.dot(n)
        D = 1.0 / (self.axes**2)
        invL2 = np.sum((n_ell**2) * D)
        if invL2 <= 0:
            return np.inf
        return 1.0 / np.sqrt(invL2)

    def L_from_beta(self, beta_rad, theta_rad, wavelength, K=0.9):
        if beta_rad <= 0:
            return np.inf
        return (K * wavelength) / (beta_rad * np.cos(theta_rad))

    # ---------- 椭球拟合 ----------
    def fit_ellipsoid_axes_from_peaks(self, peak_mappings, wavelength=None, K=0.9, beta_inst=None, inst_profile='gaussian',
                                      initial_axes=None, bounds=(1e-6, 1e4)):
        if wavelength is None:
            if self.wavelength is None:
                raise ValueError("wavelength must be provided (or set_wavelength earlier).")
            wavelength = self.wavelength
        if not self.fitted_peaks:
            raise RuntimeError("No fitted peaks: run fit_multiple_peaks first.")
        ns = []
        betas_rad = []
        thetas = []
        valid_indices = []
        if beta_inst is None:
            beta_inst_arr = None
        elif np.isscalar(beta_inst):
            beta_inst_arr = None
            beta_inst_scalar = float(beta_inst)
        else:
            beta_inst_arr = np.array(beta_inst, dtype=float)
        for idx, ((center, _, params), mapping) in enumerate(zip(self.fitted_peaks, peak_mappings)):
            if params is None or mapping is None:
                continue
            area, center_fit, sigma, gamma = params
            fwhm_deg = self.fwhm_voigt_from_sigma_gamma(sigma, gamma)
            if beta_inst_arr is not None:
                beta_inst_deg = beta_inst_arr[idx]
            elif beta_inst is not None:
                beta_inst_deg = beta_inst_scalar
            else:
                beta_inst_deg = 0.0
            beta_corr_deg = self.deconvolve_instrument(fwhm_deg, beta_inst_deg, profile=inst_profile)
            beta_corr_rad = np.deg2rad(beta_corr_deg)
            # decide n_crystal and theta
            if self.lattice_params is not None and isinstance(mapping, (list,tuple)) and len(mapping)==3 and mapping == tuple(map(int,mapping)):
                n_crystal = self.reciprocal_direction_from_hkl(mapping)
                d = self.d_spacing_from_hkl(mapping)
                theta = self.bragg_theta_from_d(d, wavelength)
            else:
                n_crystal = np.array(mapping, dtype=float)
                n_crystal = n_crystal / np.linalg.norm(n_crystal)
                theta = np.deg2rad(center_fit / 2.0)
            ns.append(n_crystal)
            betas_rad.append(beta_corr_rad)
            thetas.append(theta)
            valid_indices.append(idx)
        if len(ns) < 3:
            raise RuntimeError(f"Need >=3 peaks with mappings to fit ellipsoid (found {len(ns)}).")
        ns = np.array(ns); betas_rad = np.array(betas_rad); thetas = np.array(thetas)
        def residuals(x):
            a,b,c = x
            axes_local = np.array([a,b,c])
            D = 1.0 / (axes_local**2)
            res = np.zeros(len(ns))
            for i in range(len(ns)):
                n_ell = self.orientation.dot(ns[i])
                invL2 = np.sum((n_ell**2) * D)
                L = 1.0 / np.sqrt(invL2) if invL2>0 else 1e12
                beta_model = (K * wavelength) / (L * np.cos(thetas[i]))
                res[i] = beta_model - betas_rad[i]
            return res
        x0 = np.array(initial_axes if initial_axes is not None else tuple(self.axes))
        lb = np.full(3, bounds[0], dtype=float)
        ub = np.full(3, bounds[1], dtype=float)
        sol = least_squares(residuals, x0, bounds=(lb,ub))
        fitted = sol.x
        self.axes = fitted.copy()
        return {'axes':fitted, 'success':sol.success, 'message':sol.message, 'cost':sol.cost, 'residuals':sol.fun, 'n_used':len(ns)}

    def compute_Ls_from_fitted_peaks(self, peak_mappings, wavelength=None, K=0.9, beta_inst=None, inst_profile='gaussian'):
        results = []
        if wavelength is None:
            if self.wavelength is None:
                raise ValueError("need wavelength")
            wavelength = self.wavelength
        if not self.fitted_peaks:
            return results
        if beta_inst is None:
            beta_inst_seq = None
        elif np.isscalar(beta_inst):
            beta_inst_seq = None
            beta_inst_scalar = float(beta_inst)
        else:
            beta_inst_seq = np.array(beta_inst, dtype=float)
        for idx, ((center, _, params), mapping) in enumerate(zip(self.fitted_peaks, peak_mappings)):
            if params is None or mapping is None:
                results.append(None); continue
            area, center_fit, sigma, gamma = params
            fwhm_deg = self.fwhm_voigt_from_sigma_gamma(sigma, gamma)
            if beta_inst_seq is not None:
                beta_inst_deg = beta_inst_seq[idx]
            elif beta_inst is not None:
                beta_inst_deg = beta_inst_scalar
            else:
                beta_inst_deg = 0.0
            beta_corr_deg = self.deconvolve_instrument(fwhm_deg, beta_inst_deg, profile=inst_profile)
            beta_corr_rad = np.deg2rad(beta_corr_deg)
            if self.lattice_params is not None and isinstance(mapping, (list,tuple)) and len(mapping)==3 and mapping == tuple(map(int,mapping)):
                d = self.d_spacing_from_hkl(mapping)
                theta = self.bragg_theta_from_d(d, wavelength)
                n_crystal = self.reciprocal_direction_from_hkl(mapping)
            else:
                n_crystal = np.array(mapping, dtype=float)
                n_crystal = n_crystal / np.linalg.norm(n_crystal)
                theta = np.deg2rad(center_fit / 2.0)
            L = self.L_from_beta(beta_corr_rad, theta, wavelength, K=K)
            results.append({'index':idx,'center_deg':center_fit,'fwhm_deg':fwhm_deg,'beta_corr_deg':beta_corr_deg,'beta_corr_rad':beta_corr_rad,'theta_rad':theta,'n_crystal':n_crystal,'L':L})
        return results

    # ---------- 打印 & 绘图辅助 ----------
    def print_ellipsoid_parameters(self):
        a,b,c = self.axes
        print("\nEllipsoid axes (same units as wavelength/axes):")
        print(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

    def plot_ellipsoid_fit(self, peak_mappings, wavelength=None):
        if wavelength is None:
            if self.wavelength is None:
                raise ValueError("wavelength required")
            wavelength = self.wavelength
        Ls = self.compute_Ls_from_fitted_peaks(peak_mappings, wavelength=wavelength)
        # collect measured L and model L for directions used
        meas = []
        model = []
        dirs = []
        for r in Ls:
            if r is None: continue
            dirs.append(r['n_crystal'])
            meas.append(r['L'])
            model.append(self.effective_thickness_L_from_direction(r['n_crystal']))
        meas = np.array(meas); model = np.array(model); dirs = np.array(dirs)
        # scatter measured vs model
        plt.figure(figsize=(6,5))
        plt.scatter(model, meas)
        mx = np.linspace(min(model.min(), meas.min())*0.8, max(model.max(), meas.max())*1.2, 50)
        plt.plot(mx, mx, '--', color='gray')
        plt.xlabel('Model L (ellipsoid) (same units)')
        plt.ylabel('Measured L (Scherrer)')
        plt.title('Ellipsoid model vs measured L')
        plt.grid(True)
        plt.show()
        # optional 3D visualization of ellipsoid surface (scaled)
        u = np.linspace(0, 2*np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        x_s = self.axes[0] * np.outer(np.cos(u), np.sin(v))
        y_s = self.axes[1] * np.outer(np.sin(u), np.sin(v))
        z_s = self.axes[2] * np.outer(np.ones_like(u), np.cos(v))
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_s, y_s, z_s, rstride=4, cstride=4, alpha=0.2)
        # plot directions scaled by model L (for visualization)
        for i, n in enumerate(dirs):
            n_ell = self.orientation.dot(n)
            Lm = model[i]
            ax.quiver(0,0,0, n_ell[0]*Lm, n_ell[1]*Lm, n_ell[2]*Lm, length=1, normalize=False, color='r')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Fitted Ellipsoid (surface) and L directions')
        plt.show()

# ---------------- 示例主流程 ----------------
if __name__ == "__main__":
    analyzer = XRDAnalyzer("tio2.txt")  # 使用你上传的数据文件（示例已上传）
    # 屏蔽区域（与您提供的一致）
    mask_regions = [(23, 27), (35, 40), (46, 50), (51, 57), (60, 65), (66, 71), (73, 77)]
    for lo, hi in mask_regions:
        analyzer.add_mask_region(lo, hi)
    analyzer.fit_background(method='chebyshev', degree=4)
    diffraction_peaks = [25.3, 37.0, 37.7, 38.5, 48.0, 53.8, 55.0, 62.7, 68.7, 70.3, 75.0, 76.0]
    analyzer.fit_multiple_peaks(diffraction_peaks, window=3.0)
    analyzer.print_peak_parameters()
    analyzer.plot_all_voigt()

    # --- 重要：保持单位一致（此处用 nm） ---
    # 将晶格常数从 Å -> nm（乘 0.1）
    lattice_params_A = {'a':3.785, 'b':3.785, 'c':9.514, 'alpha':90, 'beta':90, 'gamma':90}
    lattice_params_nm = {k: (0.1*v if k in ['a','b','c'] else v) for k,v in lattice_params_A.items()}
    analyzer.set_lattice_params(lattice_params_nm)
    analyzer.set_wavelength(0.15406)   # nm (Cu Kα)
    # Miller 指数列表（按峰的顺序）
    hkl_list = [
        (1,0,1),(1,0,3),(0,0,4),(1,0,2),(2,0,0),(1,0,5),
        (2,1,1),(2,0,4),(1,0,6),(2,2,0),(2,1,5),(3,0,1)
    ]
    analyzer.set_hkl(hkl_list)
    # 建议初始轴：如果无先验用10 nm各向同性初值；若怀疑各向异性可用 [20,10,8] nm 等
    initial_axes = [10.0, 10.0, 10.0]
    fit_res = analyzer.fit_ellipsoid_axes_from_peaks(hkl_list, wavelength=analyzer.wavelength, initial_axes=initial_axes)
    analyzer.print_ellipsoid_parameters()
    analyzer.plot_ellipsoid_fit(hkl_list, wavelength=analyzer.wavelength)
