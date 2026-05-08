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
    items = header_line.split(chr(0x09))
    num_cols = len(items)
    if 't_s' in header_lower or 't_avg' in header_lower:
        return min(2, num_cols - 1)
    if 'transmittance' in header_lower:
        return min(2, num_cols - 1)
    return min(2, num_cols - 1)


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


def load_continuum_spectra_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    wavelengths = df.iloc[:, 0].values
    spectra = df.iloc[:, 1:15].values.T
    return wavelengths, spectra


def save_continuum_spectrum_comparison(
    x, input_spectrum, reconstructed_spectrum,
    group_name, spectrum_idx, output_path,
    mse=None, correlation=None
):
    plt.figure(figsize=(12, 8))
    plt.plot(x, input_spectrum, 'b-', linewidth=2, label='Input Spectrum')
    plt.plot(x, reconstructed_spectrum, 'r--', linewidth=2, label='Reconstructed Spectrum')
    metrics_text = f"MSE: {mse:.6f}\nCorrelation: {correlation:.4f}" if mse is not None and correlation is not None else ""
    if metrics_text:
        plt.text(0.98, 0.98, metrics_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title(f'{group_name} - Spectrum {spectrum_idx:02d}', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_continuum_summary_plot(wave_len_list, group_results, group_name, output_path):
    num_spectra = len(group_results)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    for i in range(num_spectra):
        ax = axes[i]
        result = group_results[i]
        ax.plot(wave_len_list, result['input_spectrum'], 'b-', linewidth=1.5, label='Input')
        ax.plot(wave_len_list, result['reconstructed_spectrum'], 'r--', linewidth=1.5, label='Recon')
        ax.set_title(f"Spectrum {i+1:02d}\nMSE:{result['mse']:.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    ax_summary = axes[-1]
    mses = [r['mse'] for r in group_results]
    corrs = [r['correlation'] for r in group_results]
    avg_mse = np.mean(mses)
    avg_corr = np.mean(corrs)
    std_mse = np.std(mses)
    std_corr = np.std(corrs)
    summary_text = (
        f"Summary Statistics\n"
        f"─────────────────\n"
        f"Total Spectra: {num_spectra}\n"
        f"Average MSE: {avg_mse:.6f}\n"
        f"Std MSE: {std_mse:.6f}\n"
        f"Average Corr: {avg_corr:.4f}\n"
        f"Std Corr: {std_corr:.4f}"
    )
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_summary.axis('off')
    fig.suptitle(f'{group_name} - All Spectra Summary', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def reconstruct_continuum_spectra(
    src_dir,
    continuum_data_dir,
    lam=0.01,
    save_results=True,
    output_dir=None,
    wavelength_display_range=None
):
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    wave_len_list_full = wave_lens[0]
    new_coef_matrix_full = new_coef_matrix[:len(wave_len_list_full), :]
    print(f"Response matrix shape: {new_coef_matrix_full.shape}")
    print(f"Response matrix wavelength range: {wave_len_list_full[0]} - {wave_len_list_full[-1]} nm")
    
    D2 = build_D2(len(wave_len_list_full))
    A_aug_full = np.vstack([new_coef_matrix_full, np.sqrt(lam) * D2])
    
    group_names = ['A组', 'B组', 'C组']
    csv_files = ['A组_反射率.csv', 'B组_反射率.csv', 'C组_反射率.csv']
    
    all_results = {}
    
    if save_results:
        if output_dir is None:
            output_dir = get_timestamped_output_dir(src_dir)
        continuum_dir = os.path.join(output_dir, "continuum")
        os.makedirs(continuum_dir, exist_ok=True)
        print(f"Created continuum directory: {continuum_dir}")
    
    for group_name, csv_file in zip(group_names, csv_files):
        csv_path = os.path.join(continuum_data_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {group_name}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {group_name}")
        print(f"{'='*50}")
        
        wavelengths_csv, spectra = load_continuum_spectra_from_csv(csv_path)
        print(f"Loaded {spectra.shape[0]} spectra from {csv_file}")
        print(f"CSV wavelength range: {wavelengths_csv[0]} - {wavelengths_csv[-1]} nm")
        
        wl_min = max(wave_len_list_full[0], wavelengths_csv[0])
        wl_max = min(wave_len_list_full[-1], wavelengths_csv[-1])
        wave_len_list = wave_len_list_full[(wave_len_list_full >= wl_min) & (wave_len_list_full <= wl_max)]
        print(f"Using wavelength range: {wave_len_list[0]} - {wave_len_list[-1]} nm ({len(wave_len_list)} points)")
        
        cs = CubicSpline(wavelengths_csv, spectra.T, axis=0)
        spectra_interp_full = cs(wave_len_list_full).T
        spectra_interp_full = np.clip(spectra_interp_full, 0, None)
        print(f"Interpolated spectra (full) shape: {spectra_interp_full.shape}")
        print(f"Interpolated spectra range: min={np.min(spectra_interp_full):.4f}, max={np.max(spectra_interp_full):.4f}")
        
        wl_mask = (wave_len_list_full >= wl_min) & (wave_len_list_full <= wl_max)
        
        group_results = []
        
        if save_results:
            group_dir = os.path.join(continuum_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
        
        for i in range(spectra_interp_full.shape[0]):
            input_spectrum_full = spectra_interp_full[i]
            
            y_measured = new_coef_matrix_full @ input_spectrum_full.T
            y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])
            
            res = lsq_linear(A_aug_full, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
            reconstructed_spectrum_full = res.x
            
            input_spectrum = input_spectrum_full[wl_mask]
            reconstructed_spectrum = reconstructed_spectrum_full[wl_mask]
            
            mse = np.mean((reconstructed_spectrum - input_spectrum) ** 2)
            correlation = np.corrcoef(input_spectrum, reconstructed_spectrum)[0, 1]
            
            print(f"  Spectrum {i+1:02d}: MSE={mse:.6f}, Corr={correlation:.4f}")
            
            result_entry = {
                'spectrum_idx': i + 1,
                'input_spectrum': input_spectrum.copy(),
                'reconstructed_spectrum': reconstructed_spectrum.copy(),
                'trans_curve': y_measured.copy(),
                'mse': mse,
                'correlation': correlation
            }
            group_results.append(result_entry)
            
            if save_results:
                plot_filename = f"spectrum_{i+1:02d}.png"
                plot_filepath = os.path.join(group_dir, plot_filename)
                save_continuum_spectrum_comparison(
                    wave_len_list, input_spectrum, reconstructed_spectrum,
                    group_name, i + 1, plot_filepath,
                    mse=mse, correlation=correlation
                )
        
        if save_results:
            summary_path = os.path.join(group_dir, "summary_all_spectra.png")
            save_continuum_summary_plot(wave_len_list, group_results, group_name, summary_path)
            print(f"Saved summary plot: {summary_path}")
        
        all_results[group_name] = group_results
    
    print(f"\n{'='*60}")
    print(f"Continuum reconstruction completed.")
    print(f"{'='*60}")
    
    return all_results


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


DEFAULT_SAVE_BASE_DIR = r"F:\05-Jerome Studios\Coating Design\Simulate_Output\spectral-calculation"


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
                plt.close()
            
            y_measured = new_coef_matrix @ arr1.T
            y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])
            
            if show_plot:
                plt.figure()
                plot_simple_curve(wave_len_list, y_measured, f"Trans Curve - CWL: {actual_cwl}nm, FWHM: {fwhm}nm")
                plt.close()
            
            res = lsq_linear(A_aug, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
            x_rec_nnls = res.x
            
            print(f"Reconstruction shape: {x_rec_nnls.shape}")
            
            mse = np.mean((x_rec_nnls - arr1) ** 2)
            print(f"MSE: {mse:.6f}")
            
            if show_plot:
                plt.figure()
                plot_simple_multi_curves(wave_len_list, [x_rec_nnls, arr1], 
                                         f"Rebuild - CWL: {actual_cwl}nm, FWHM: {fwhm}nm")
                plt.close()
            
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
    u500_data = r"E:\AA_repository\OneDrive - Unispectral Qingdao Microelectronics Co. LTD\01_研发\01-开发相关\07_MEMS\03_Coating\02-Macleod-仿真数据库\analysis_output\FPI-Performance\20260507_213546_202604171031-U450-MEMS-Metal-Coating-Ag-J08\BPF-data"
    
    continuum_data_dir = r"E:\AA_repository\OneDrive - Unispectral Qingdao Microelectronics Co. LTD\01_研发\00_应用场景\15-LED检测\A01-需求输入\20260415-色板测试\03-分析数据\20260422-完整分析结果"
    
    continuum_results = reconstruct_continuum_spectra(
        src_dir=u500_data,
        continuum_data_dir=continuum_data_dir,
        lam=0.01,
        save_results=True
    )
