import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
import pandas as pd
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def _detect_transmittance_col(header_line):
    header_lower = header_line.lower()
    if 't_s' in header_lower or 't_avg' in header_lower:
        return 2
    if 'transmittance' in header_lower:
        return 2
    return 2


def load_matrix_from_file(src_dir):
    wavelengths, transmittances = [], []
    all_files = os.listdir(src_dir)
    txt_files = sorted([f for f in all_files if f.endswith('.txt') and os.path.isfile(os.path.join(src_dir, f))])
    trans_col = None
    for f_name in txt_files:
        file_path = os.path.join(src_dir, f_name)
        waves, trans = [], []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    if trans_col is None:
                        trans_col = _detect_transmittance_col(line)
                    continue
                items = line.split(chr(0x09))
                wave_f = float(items[0].strip())
                if wave_f - wave_f // 1 < 0.000001 and 350 <= int(wave_f) <= 950:
                    waves.append(int(wave_f))
                    trans.append(float(items[trans_col].strip()) / 100)
        wavelengths.append(waves)
        transmittances.append(trans)
    transmittances = np.array(transmittances)
    row_sums = transmittances.sum(axis=1, keepdims=True)
    max_row_sum = np.max(row_sums)
    print("max_row_sum", max_row_sum)
    print(f"Loaded {len(txt_files)} txt files from {src_dir}")
    return np.array(wavelengths), transmittances


def gaussian_beam(x, mu, sig):
    return np.exp(-np.power(np.array(x) - mu, 2.) / (2 * np.power(sig, 2.)))


def find_fwhm_normal(in_x, in_y):
    x_dense = np.linspace(in_x[0], in_x[-1], len(in_x) * 10)
    cs = CubicSpline(in_x, in_y)
    y_points = cs(x_dense)
    peak_id = np.argmax(y_points)
    y_left = y_points[:peak_id]
    y_right = y_points[peak_id:]
    half_peak = y_points[peak_id] / 2
    lefts = np.sort(np.where(y_left < half_peak))[0]
    left_idx = (lefts[lefts.size - 1]) if lefts.size > 0 else -1
    rights = np.sort(np.where(y_right < half_peak))[0]
    right_idx = (peak_id + rights[0]) if rights.size > 0 else -1
    return x_dense[right_idx] - x_dense[left_idx], x_dense[(left_idx + right_idx) // 2], x_dense[peak_id]


def build_D2(L: int) -> np.ndarray:
    D = np.zeros((L - 2, L), dtype=np.float64)
    for i in range(L - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def plot_simple_curve(x, y, in_title):
    plt.plot(x, y, linestyle='-')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(in_title)


def plot_simple_multi_curves(x, ys, in_label):
    for idx, y in enumerate(ys):
        plt.plot(x, y, color=COLORS[idx % len(COLORS)])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity " + str(idx))
    plt.title(in_label)


DEFAULT_SAVE_BASE_DIR = r"D:\Git-仓库\spectral calculation\spectral_cal\data analysis"


def get_timestamped_output_dir(src_dir, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_SAVE_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_basename = os.path.basename(src_dir.rstrip(os.sep))
    subfolder_name = f"{timestamp}-{src_basename}"
    output_dir = os.path.join(base_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def save_spectrum_comparison_plot_with_fwhm(
    x, input_spectrum, reconstructed_spectrum, cwl, fwhm, output_path,
    wave_len_list=None, wavelength_display_range=None
):
    input_fwhm, input_cwl, input_peak = find_fwhm_normal(x, input_spectrum)
    recon_fwhm, recon_cwl, recon_peak = find_fwhm_normal(x, reconstructed_spectrum)
    if wavelength_display_range is not None:
        wl_start, wl_end = wavelength_display_range
        mask = (x >= wl_start) & (x <= wl_end)
        x_display = x[mask]
        input_display = input_spectrum[mask]
        recon_display = reconstructed_spectrum[mask]
    else:
        x_display = x
        input_display = input_spectrum
        recon_display = reconstructed_spectrum
    plt.figure(figsize=(12, 8))
    plt.plot(x_display, input_display, 'b-', linewidth=2, label='Input Spectrum')
    plt.plot(x_display, recon_display, 'r--', linewidth=2, label='Reconstructed Spectrum')
    fwhm_text = (
        f"Input Spectrum FWHM: {input_fwhm:.2f} nm\n"
        f"Input Spectrum CWL: {input_cwl:.2f} nm\n"
        f"Reconstructed FWHM: {recon_fwhm:.2f} nm\n"
        f"Reconstructed CWL: {recon_cwl:.2f} nm\n"
        f"FWHM Difference: {abs(input_fwhm - recon_fwhm):.2f} nm"
    )
    plt.text(0.98, 0.98, fwhm_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title(f'Spectrum Comparison - CWL: {cwl}nm, FWHM: {fwhm}nm', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrum comparison plot: {output_path}")
    return input_fwhm, recon_fwhm, input_cwl, recon_cwl


def lst_with_aug_reg_cwl_fwhm_list_loop(
    src_dir,
    cwl_list,
    fwhm_list,
    amplitude=0.6,
    lam=0.01,
    show_plot=True,
    save_results=False,
    output_dir=None,
    wavelength_display_range=None
):
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    wave_len_list = wave_lens[0]
    new_coef_matrix = new_coef_matrix[:len(wave_len_list), :]
    print("wave_len_list.shape", wave_lens.shape)
    print("new_coef_matrix.shape", new_coef_matrix.shape)
    print(f"wave_len_list range: {wave_len_list[0]} - {wave_len_list[-1]} nm")
    
    D2 = build_D2(len(wave_len_list))
    A_aug = np.vstack([new_coef_matrix, np.sqrt(lam) * D2])
    
    results = []
    csv_data = {
        'Input_CWL_nm': [],
        'Output_CWL_nm': [],
        'Input_FWHM_nm': [],
        'Output_FWHM_nm': [],
        'MSE': []
    }
    
    if save_results:
        if output_dir is None:
            output_dir = get_timestamped_output_dir(src_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(new_coef_matrix, aspect='auto', 
                   extent=[wave_len_list[0], wave_len_list[-1], 0, len(wave_len_list)],
                   cmap="rainbow")
        plt.colorbar(label='Transmittance')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Channel Index")
        plt.title("Response Matrix")
        resp_matrix_path = os.path.join(output_dir, "response_matrix.png")
        plt.savefig(resp_matrix_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved response matrix: {resp_matrix_path}")
        
        for fwhm in fwhm_list:
            fwhm_dir = os.path.join(output_dir, f"FWHM={fwhm}")
            os.makedirs(fwhm_dir, exist_ok=True)
            print(f"Created FWHM directory: {fwhm_dir}")
    
    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    
    total_iterations = len(cwl_list) * len(fwhm_list)
    current_iteration = 0
    
    for cwl in cwl_list:
        peak_idx = np.argmin(np.abs(wave_len_list - cwl))
        actual_cwl = wave_len_list[peak_idx]
        
        for fwhm in fwhm_list:
            current_iteration += 1
            sig = fwhm / 2.355
            
            print(f"\n{'='*50}")
            print(f"Iteration {current_iteration}/{total_iterations}")
            print(f"CWL: {actual_cwl}nm (idx: {peak_idx}), FWHM: {fwhm}nm, sig: {sig:.2f}")
            print(f"{'='*50}")
            
            arr1 = amplitude * gaussian_beam(np.arange(0, len(arr), 1.0), peak_idx, sig)
            
            if show_plot:
                plt.figure()
                plot_simple_curve(wave_len_list, arr1, f"Input Curve - CWL: {actual_cwl}nm, FWHM: {fwhm}nm")
                # plt.show()
            
            y_measured = new_coef_matrix @ arr1.T
            y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])
            
            if show_plot:
                plt.figure()
                plot_simple_curve(wave_len_list, y_measured, f"Trans Curve - CWL: {actual_cwl}nm, FWHM: {fwhm}nm")
                # plt.show()
            
            res = lsq_linear(A_aug, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
            x_rec_nnls = res.x
            
            print(f"Reconstruction shape: {x_rec_nnls.shape}")
            
            mse = np.mean((x_rec_nnls - arr1) ** 2)
            print(f"MSE: {mse:.6f}")
            
            if show_plot:
                plt.figure()
                plot_simple_multi_curves(wave_len_list, [x_rec_nnls, arr1], 
                                         f"Rebuild - CWL: {actual_cwl}nm, FWHM: {fwhm}nm")
                # plt.show()
            
            input_fwhm_calc, input_cwl_calc, _ = find_fwhm_normal(wave_len_list, arr1)
            recon_fwhm_calc, recon_cwl_calc, _ = find_fwhm_normal(wave_len_list, x_rec_nnls)
            
            if save_results:
                fwhm_dir = os.path.join(output_dir, f"FWHM={fwhm}")
                plot_filename = f"cwl_{actual_cwl}_fwhm_{fwhm}.png"
                plot_filepath = os.path.join(fwhm_dir, plot_filename)
                save_spectrum_comparison_plot_with_fwhm(
                    wave_len_list, arr1, x_rec_nnls, actual_cwl, fwhm, plot_filepath,
                    wavelength_display_range=wavelength_display_range
                )
            
            result_entry = {
                'cwl_nm': actual_cwl,
                'cwl_idx': peak_idx,
                'fwhm_nm': fwhm,
                'sig': sig,
                'amplitude': amplitude,
                'input_spectrum': arr1.copy(),
                'reconstructed_spectrum': x_rec_nnls.copy(),
                'trans_curve': y_measured.copy(),
                'mse': mse,
                'input_fwhm_calc': input_fwhm_calc,
                'recon_fwhm_calc': recon_fwhm_calc,
                'input_cwl_calc': input_cwl_calc,
                'recon_cwl_calc': recon_cwl_calc
            }
            results.append(result_entry)
            
            csv_data['Input_CWL_nm'].append(input_cwl_calc)
            csv_data['Output_CWL_nm'].append(recon_cwl_calc)
            csv_data['Input_FWHM_nm'].append(input_fwhm_calc)
            csv_data['Output_FWHM_nm'].append(recon_fwhm_calc)
            csv_data['MSE'].append(mse)
    
    print(f"\n{'='*60}")
    print(f"Loop completed. Total iterations: {len(results)}")
    print(f"{'='*60}")
    
    if save_results:
        csv_path = os.path.join(output_dir, "spectrum_results.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV results saved: {csv_path}")
        
        summary_path = os.path.join(output_dir, "rebuild_summary.npy")
        np.save(summary_path, results)
        print(f"Summary saved: {summary_path}")
    
    return results


if __name__ == "__main__":
    u500_data = r"D:\OneDrive\Jerome\04-仓库\00-工作\Coating Design Data Base\U500\202605032112-U500_Mac-angular-bow"
    
    results = lst_with_aug_reg_cwl_fwhm_list_loop(
        u500_data,
        cwl_list=[510, 530, 550, 570, 590, 610, 630, 650, 670, 690],
        fwhm_list=[1, 3, 5, 7, 9],
        amplitude=0.6,
        lam=0.01,
        show_plot=True,
        save_results=True
    )
