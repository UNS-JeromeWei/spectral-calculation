import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from multiprocessing import Pool, cpu_count
import multiprocessing

# Matplotlib font configuration (avoid display issues)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf"  # Cyan
]


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_numeric(s):
    """Check if string can be parsed as a number (int or float)"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_wavelength(value_str):
    """Parse wavelength value, supporting both integer and float formats
    
    Returns integer wavelength value, or None if parsing fails
    """
    value_str = value_str.strip()
    if not value_str:
        return None
    
    try:
        # Try integer first
        return int(value_str)
    except ValueError:
        try:
            # Try float and round to integer
            return int(float(value_str))
        except ValueError:
            return None


def detect_delimiter(line):
    """Auto-detect delimiter type from a line of text
    
    Supported delimiters: tab, comma, whitespace
    Returns the most likely delimiter character or string
    """
    tab_count = line.count('\t')
    comma_count = line.count(',')
    
    if tab_count >= 2:
        return '\t'
    elif comma_count >= 2:
        return ','
    else:
        # Default to whitespace splitting (handles multiple spaces)
        return None  # None means use split() without arguments


def find_transmittance_column(headers):
    """Find the column index for transmittance data from header row
    
    Looks for keywords: 'transmittance', 'trans', '透射', '透射率'
    Returns column index, or 2 as default (0-indexed)
    """
    keywords = ['transmittance', 'trans', '透射', '透射率']
    for idx, h in enumerate(headers):
        h_lower = h.lower().strip()
        for kw in keywords:
            if kw in h_lower:
                return idx
    return 2  # Default: 3rd column (index 2)


def find_wavelength_column(headers):
    """Find the column index for wavelength data from header row
    
    Looks for keywords: 'wavelength', 'wave', '波长', 'nm'
    Returns column index, or 0 as default (0-indexed)
    """
    keywords = ['wavelength', 'wave', '波长', 'nm']
    for idx, h in enumerate(headers):
        h_lower = h.lower().strip()
        for kw in keywords:
            if kw in h_lower:
                return idx
    return 0  # Default: 1st column (index 0)


def parse_numeric_value(value_str):
    """Parse a string to float, handling various formats
    
    Handles: standard float, scientific notation, values starting with '.'
    Returns float value or None if parsing fails
    """
    value_str = value_str.strip()
    if not value_str:
        return None
    
    # Handle values starting with '.' (e.g., '.6192' -> '0.6192')
    if value_str.startswith('.'):
        value_str = '0' + value_str
    
    try:
        return float(value_str)
    except ValueError:
        return None


def load_matrix_from_file(src_dir):
    """Load all airgap files and return wavelengths and transmittance matrix
    
    Enhanced version with auto-detection for:
    - Delimiter type (tab, comma, whitespace)
    - Column positions (wavelength and transmittance)
    - Numeric formats (standard, scientific, leading dot)
    
    Returns:
        wavelengths: 2D array (same wavelength array for each file)
        transmittances: 2D array, each row is one airgap's transmittance curve
    """
    wavelengths = []
    transmittances = []

    files = sorted(os.listdir(src_dir))

    for f_name in files:
        file_path = os.path.join(src_dir, f_name)
        waves, trans = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: Empty file {f_name}")
            continue
        
        # Detect delimiter from first line
        delimiter = detect_delimiter(lines[0])
        
        # Try to parse header row
        first_line = lines[0].strip()
        if delimiter:
            first_items = [item.strip() for item in first_line.split(delimiter)]
        else:
            first_items = first_line.split()
        
        # Check if first line is header (non-numeric first column)
        is_header = not is_numeric(first_items[0]) if first_items else False
        
        # Determine column indices
        if is_header:
            wave_col = find_wavelength_column(first_items)
            trans_col = find_transmittance_column(first_items)
            start_idx = 1  # Skip header
        else:
            wave_col = 0
            trans_col = 2  # Default: wavelength, reflectance, transmittance
            start_idx = 0
        
        # Parse data lines
        for idx in range(start_idx, len(lines)):
            line = lines[idx].strip()
            if not line:
                continue
            
            if delimiter:
                items = [item.strip() for item in line.split(delimiter)]
            else:
                items = line.split()
            
            # Skip lines with insufficient columns
            if len(items) <= max(wave_col, trans_col):
                continue
            
            # Parse wavelength (support both integer and float formats)
            # Keep as float to preserve decimal wavelengths (e.g., 400.2, 400.4)
            wave_val = parse_numeric_value(items[wave_col])
            if wave_val is None:
                continue
            waves.append(wave_val)
            
            # Parse transmittance (float, divide by 100 to convert from %)
            trans_val = parse_numeric_value(items[trans_col])
            if trans_val is not None:
                trans.append(trans_val / 100.0)
            else:
                # If parsing fails, use 0 as fallback
                trans.append(0.0)
        
        wavelengths.append(waves)
        transmittances.append(trans)
        print(f"Loaded {f_name}: {len(waves)} points, delimiter={'tab' if delimiter==chr(0x09) else 'comma' if delimiter==',' else 'whitespace'}")

    transmittances = np.array(transmittances)
    row_sums = transmittances.sum(axis=1, keepdims=True)
    max_row_sum = np.max(row_sums)
    print("max_row_sum", max_row_sum)

    return np.array(wavelengths), transmittances


def gaussian_beam(x, mu, sig):
    return np.exp(-np.power(np.array(x) - mu, 2.) / (2 * np.power(sig, 2.)))


def build_D2(L: int) -> np.ndarray:
    """(L-2) x L second-difference matrix"""
    D = np.zeros((L - 2, L), dtype=np.float64)
    for i in range(L - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def find_fwhm_normal(in_x, in_y):
    """Get FWHM, CWL, and peak wavelength from a curve

    Uses linear interpolation to precisely locate the half-maximum positions.
    FWHM = right_half_wavelength - left_half_wavelength (at 50% peak value)

    :param in_x: wavelength values
    :param in_y: intensity values
    :return: fwhm, cwl, peak_wavelength
    """
    # Interpolate to get dense points
    x_dense = np.linspace(in_x[0], in_x[-1], len(in_x) * 10)
    cs = CubicSpline(in_x, in_y)
    y_points = cs(x_dense)

    # Find peak using reconstructed spectrum
    peak_id = np.argmax(y_points)
    peak_value = y_points[peak_id]
    half_peak = peak_value / 2

    # Find left half-maximum position (from peak going left)
    left_idx = None
    for i in range(peak_id, 0, -1):
        if y_points[i] <= half_peak:
            left_idx = i
            break
    
    # Find right half-maximum position (from peak going right)
    right_idx = None
    for i in range(peak_id, len(y_points)):
        if y_points[i] <= half_peak:
            right_idx = i
            break

    # Calculate precise wavelengths using linear interpolation
    if left_idx is not None and right_idx is not None:
        # Left side: interpolate between left_idx and left_idx + 1
        x1, x2 = x_dense[left_idx], x_dense[left_idx + 1]
        y1, y2 = y_points[left_idx], y_points[left_idx + 1]
        # Linear interpolation: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        # Solve for x when y = half_peak
        left_wavelength = x1 + (half_peak - y1) * (x2 - x1) / (y2 - y1)
        
        # Right side: interpolate between right_idx - 1 and right_idx
        x1, x2 = x_dense[right_idx - 1], x_dense[right_idx]
        y1, y2 = y_points[right_idx - 1], y_points[right_idx]
        right_wavelength = x1 + (half_peak - y1) * (x2 - x1) / (y2 - y1)
        
        fwhm = right_wavelength - left_wavelength
        cwl = (left_wavelength + right_wavelength) / 2
    else:
        # Fallback if half-maximum not found
        fwhm = 0.0
        cwl = x_dense[peak_id]
    
    return fwhm, cwl, x_dense[peak_id]


def calc_spectral_residual_score(in_y, threshold_pct=0.03):
    """Calculate Spectral Residual Score: Coverage x log10(Attenuation)

    :param in_y: reconstructed curve y values
    :param threshold_pct: threshold percentage (default 0.03)
    :return: coverage, attenuation, score
    """
    peak_val = np.max(in_y)
    threshold = peak_val * threshold_pct
    stopband_mask = in_y < threshold
    
    coverage = np.sum(stopband_mask) / len(in_y)
    
    if coverage == 0:
        return 0.0, 1.0, 0.0
    
    stopband_mean = np.mean(in_y[stopband_mask])
    attenuation = peak_val / stopband_mean if stopband_mean > 0 else 1000.0
    score = coverage * np.log10(attenuation)
    
    return coverage, attenuation, score


def plot_simple_multi_curves(x, ys, in_title, save_path=None, cwl=None, fwhm=None,
                              input_fwhm=None, score=None, fwhm_diff_pct=None,
                              wave_idx=None):
    plt.figure(figsize=(12, 6))
    for idx, y in enumerate(ys):
        plt.plot(x, y, color=COLORS[idx])
        plt.xlabel("Wavelength (nm)", fontsize=12, fontweight='bold')
        plt.ylabel("Intensity", fontsize=12, fontweight='bold')

    plt.title(in_title, fontsize=11)
    plt.grid(True)

    if cwl is not None and fwhm is not None:
        peak_val = np.max(ys[0])
        half_val = peak_val / 2
        threshold_val = peak_val * 0.03

        annot_text = f"CWL: {cwl:.2f}nm\nRecon FWHM: {fwhm:.2f}nm"
        if input_fwhm is not None:
            annot_text += f"\nInput FWHM: {input_fwhm:.1f}nm"
            if fwhm_diff_pct is not None:
                annot_text += f"\ndFWHM: {fwhm_diff_pct:+.2f}%"
        if wave_idx is not None:
            annot_text += f"\nWave Index: {wave_idx}"
        if score is not None:
            annot_text += f"\nSpectral Score: {score:.2f}"

        plt.annotate(annot_text,
                     xy=(cwl, peak_val), xytext=(cwl + 15, peak_val * 0.70),
                     fontsize=7, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red', alpha=0.8))
        plt.axhline(y=half_val, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        plt.axhline(y=threshold_val, color='purple', linestyle='--', linewidth=0.8, alpha=0.6, label='3% Threshold')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")


def create_time_folder(src_dir):
    """Create a folder named with current timestamp and source directory name"""
    current_time = time.strftime("%Y%m%d%H%M%S")
    src_name = os.path.basename(os.path.normpath(src_dir))
    folder_name = f"{current_time}-{src_name}"
    parent_dir = os.path.dirname(os.getcwd())
    save_folder = os.path.join(parent_dir, "data analysis", folder_name)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def process_single_wavelength(args):
    """Process a single wavelength point - designed for parallel execution
    
    Args:
        args: tuple containing all necessary parameters
            (peak_idx, input_fwhm, sigma, A_aug, new_coef_matrix, D2, 
             wave_lens_0, start_wavelength, wavelength_interval, fwhm_save_folder)
    
    Returns:
        dict with results: peak_idx, recon_fwhm, recon_cwl, actual_center_wavelength,
                          fwhm_diff_pct, coverage_val, attenuation_val, score_val,
                          x_rec_nnls, arr1
    """
    (peak_idx, input_fwhm, sigma, A_aug, new_coef_matrix, D2, 
     wave_lens_0, start_wavelength, wavelength_interval, num_wavelength_points) = args
    
    # Create Gaussian curve
    arr1 = 0.6 * gaussian_beam(np.arange(0, num_wavelength_points, 1.0), peak_idx, sigma)
    
    # Calculate measurement curve
    y_measured = new_coef_matrix @ arr1.T
    
    # Build augmented measurement vector
    y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])
    
    # Reconstruct using lsq_linear
    res = lsq_linear(A_aug, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
    x_rec_nnls = res.x
    
    # Normalize reconstructed spectrum
    input_peak_val = 0.6
    rec_peak_val = np.max(x_rec_nnls)
    if rec_peak_val > 0:
        x_rec_normalized = x_rec_nnls * (input_peak_val / rec_peak_val)
    else:
        x_rec_normalized = x_rec_nnls
    
    # Calculate CWL and FWHM
    recon_fwhm, recon_cwl, peak_wavelength = find_fwhm_normal(wave_lens_0, x_rec_normalized)
    fwhm_diff_pct = (recon_fwhm - input_fwhm) / input_fwhm * 100.0
    
    # Calculate spectral score
    coverage_val, attenuation_val, score_val = calc_spectral_residual_score(x_rec_nnls)
    
    # Calculate actual center wavelength
    actual_center_wavelength = start_wavelength + peak_idx * wavelength_interval
    
    return {
        'peak_idx': peak_idx,
        'recon_fwhm': recon_fwhm,
        'recon_cwl': recon_cwl,
        'actual_center_wavelength': actual_center_wavelength,
        'fwhm_diff_pct': fwhm_diff_pct,
        'coverage_val': coverage_val,
        'attenuation_val': attenuation_val,
        'score_val': score_val,
        'x_rec_nnls': x_rec_nnls,
        'arr1': arr1,
        'input_fwhm': input_fwhm
    }


def lst_with_aug_reg_400_700_wave_index_loop(src_dir):
    """Loop through wavelength indices and FWHM values to reconstruct curves
    
    Core calculation follows: rebuild_curves_2_peak_260227_for_new_uc500.py
    function: lst_with_aug_reg_new_uc500_with_wave_loop
    
    Extensions:
    - FWHM loop: 5, 10, 15 nm
    - Summary outputs: fwhm_diff_summary.csv, spectral_score_summary.csv
    """
    # Create time-named root folder
    save_root = create_time_folder(src_dir)
    print(f"Created save root folder: {save_root}")

    # Load all airgap data (reference method)
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    
    # Get wavelength interval from data (assumes uniform spacing)
    wavelength_interval = wave_lens[0][1] - wave_lens[0][0]
    print(f"Wavelength interval: {wavelength_interval:.4f} nm")
    
    # Data structure: each file is one airgap (row), each column is one wavelength point
    # wave_lens shape: (num_airgaps, num_wavelength_points)
    # new_coef_matrix shape: (num_airgaps, num_wavelength_points)
    
    start_wavelength = wave_lens[0][0]
    num_airgaps = wave_lens.shape[0]
    num_wavelength_points = wave_lens.shape[1]
    
    print(f"Original data: {num_airgaps} airgaps x {num_wavelength_points} wavelength points")
    
    # Calculate wavelength indices for 400~700nm range
    target_start_wavelength = 400.0
    target_end_wavelength = 970.0
    
    # Calculate start and end indices for wavelength range
    wavelength_start_idx = int(round((target_start_wavelength - start_wavelength) / wavelength_interval))
    wavelength_end_idx = int(round((target_end_wavelength - start_wavelength) / wavelength_interval)) + 1
    
    # Ensure indices are within bounds
    wavelength_start_idx = max(0, wavelength_start_idx)
    wavelength_end_idx = min(num_wavelength_points, wavelength_end_idx)
    
    # Number of wavelength points in target range
    num_wavelengths_in_range = wavelength_end_idx - wavelength_start_idx
    
    print(f"Target wavelength range: {target_start_wavelength}nm ~ {target_end_wavelength}nm")
    print(f"Wavelength indices: {wavelength_start_idx} ~ {wavelength_end_idx} ({num_wavelengths_in_range} points)")
    
    # Cut wavelength columns to 400~700nm range
    # Keep all airgap rows (they represent different FPI cavity lengths)
    wave_lens = wave_lens[:, wavelength_start_idx:wavelength_end_idx]
    new_coef_matrix = new_coef_matrix[:, wavelength_start_idx:wavelength_end_idx]
    
    print("wave_lens.shape", wave_lens.shape)
    print("new_coef_matrix.shape", new_coef_matrix.shape)
    print(f"Wavelength range: {wave_lens[0][0]}nm ~ {wave_lens[0][-1]}nm")
    
    # Update start_wavelength to the actual start of our range
    start_wavelength = wave_lens[0][0]

    # Fixed regularization parameter (reference: lam = 0.01)
    lam = 0.01
    
    # D2 regularization matrix acts on wavelength dimension
    # Build D2 based on number of wavelength points
    num_wavelength_points = wave_lens.shape[1]
    num_airgaps = wave_lens.shape[0]
    D2 = build_D2(num_wavelength_points)
    
    # For spectral reconstruction with augmented regularization:
    # We solve: [T; sqrt(lam)*D2] @ x = [y_measured; 0]
    # where T is transmittance matrix (airgap x wavelength), x is spectrum (wavelength), y is measurement (airgap)
    # 
    # new_coef_matrix: (num_airgaps, num_wavelengths) = (701, 1501)
    # D2: (num_wavelengths-2, num_wavelengths) = (1499, 1501)
    # 
    # These can be stacked vertically since they have the same number of columns (wavelengths)
    # A_aug: (num_airgaps + num_wavelengths-2, num_wavelengths) = (701+1499, 1501) = (2200, 1501)
    # This is an overdetermined system (more equations than unknowns)
    
    A_aug = np.vstack([new_coef_matrix, np.sqrt(lam) * D2])
    print(f"A_aug shape: {A_aug.shape}")

    arr = np.zeros(num_wavelength_points, dtype=float)
    
    # Input CWL: 420~960nm, interval 20nm
    input_cwl_list = list(range(420, 961, 20))
    
    # Convert CWL to wave_idx correctly, considering wavelength interval
    # If wavelength starts at 400nm, index = (cwl - 400) / interval
    check_wave = [round((cwl - start_wavelength) / wavelength_interval) for cwl in input_cwl_list]
    print(f"Input CWL list (420~960nm, interval 20nm): {input_cwl_list}")
    print(f"Check wave array (wave_idx): {check_wave}")
    
    # Input FWHM list: 5, 10, 15 nm
    input_fwhm_list = list(range(5, 16, 5))
    print(f"Input FWHM list: {input_fwhm_list}")
    
    # Summary data storage
    summary_input_wavelength = {}
    summary_recon_fwhm = {}
    summary_fwhm_diff_pct = {}
    summary_score = {}
    
    total_tasks = len(input_fwhm_list) * len(check_wave)
    completed_tasks = 0

    # ========== Sequential Processing (Safe Mode) ==========
    # Using sequential processing to avoid memory overload
    # Parallel processing caused system crash due to large matrix copying
    print(f"\nUsing sequential processing (safe mode to prevent memory overload)")
    
    start_time_total = time.time()
    
    for input_fwhm in input_fwhm_list:
        # Convert FWHM to sigma (FWHM = 2.355 * sigma)
        # sigma is in nm units, need to convert to index units
        sigma_nm = input_fwhm / 2.355
        sigma = sigma_nm / wavelength_interval  # Convert to index units
        
        # Create subfolder for each FWHM
        fwhm_folder_name = f"FWHM_{input_fwhm:02d}"
        fwhm_save_folder = os.path.join(save_root, fwhm_folder_name)
        os.makedirs(fwhm_save_folder, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Processing Input FWHM = {input_fwhm}nm, sigma = {sigma:.4f}")
        print(f"Save folder: {fwhm_save_folder}")
        
        # Store results for this FWHM
        cwl_list = []
        fwhm_list = []
        fwhm_diff_pct_list = []
        coverage_list = []
        attenuation_list = []
        score_list = []
        input_wl_list = []
        
        start_time_fwhm = time.time()
        
        for peak_idx in check_wave:
            completed_tasks += 1
            
            # Create Gaussian curve (reference method, modified for variable FWHM)
            # arr1 is a spectrum vector (wavelength dimension), length = num_wavelength_points
            arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), peak_idx, sigma)
            
            # Calculate measurement curve (reference method)
            # new_coef_matrix: (num_airgaps, num_wavelengths)
            # arr1: (num_wavelengths,)
            # y_measured: (num_airgaps,) - simulated measurement at each airgap
            y_measured = new_coef_matrix @ arr1.T
            
            # Build augmented measurement vector for lsq_linear
            y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])
            
            # Reconstruct using lsq_linear (reference method)
            res = lsq_linear(A_aug, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
            x_rec_nnls = res.x
            
            # Normalize reconstructed spectrum to match input peak value (0.6)
            # This ensures FWHM comparison is based on same peak baseline
            input_peak_val = 0.6  # Same as arr1 peak value
            rec_peak_val = np.max(x_rec_nnls)
            if rec_peak_val > 0:
                x_rec_normalized = x_rec_nnls * (input_peak_val / rec_peak_val)
            else:
                x_rec_normalized = x_rec_nnls
            
            # Calculate CWL and FWHM using normalized spectrum
            recon_fwhm, recon_cwl, peak_wavelength = find_fwhm_normal(wave_lens[0], x_rec_normalized)
            fwhm_diff_pct = (recon_fwhm - input_fwhm) / input_fwhm * 100.0
            
            # Calculate spectral score
            coverage_val, attenuation_val, score_val = calc_spectral_residual_score(x_rec_nnls)
            
            # Calculate actual center wavelength from peak_idx
            actual_center_wavelength = start_wavelength + peak_idx * wavelength_interval
            
            cwl_list.append(recon_cwl)
            fwhm_list.append(recon_fwhm)
            fwhm_diff_pct_list.append(fwhm_diff_pct)
            coverage_list.append(coverage_val)
            attenuation_list.append(attenuation_val)
            score_list.append(score_val)
            input_wl_list.append(actual_center_wavelength)  # Use actual wavelength
            
            # Generate title and save path (use CWL for filename)
            title = (f"Wave Index: {peak_idx} | Input WL: {actual_center_wavelength:.1f}nm | "
                     f"Input FWHM: {input_fwhm}nm | Recon FWHM: {recon_fwhm:.2f}nm | "
                     f"dFWHM: {fwhm_diff_pct:+.2f}% | Score: {score_val:.2f}")
            save_path = os.path.join(fwhm_save_folder, f"CWL_{actual_center_wavelength:.0f}nm.png")
            
            # Plot and save
            plot_simple_multi_curves(wave_lens[0], [x_rec_nnls, arr1], title, save_path,
                                     cwl=recon_cwl, fwhm=recon_fwhm, input_fwhm=input_fwhm,
                                     score=score_val, fwhm_diff_pct=fwhm_diff_pct,
                                     wave_idx=peak_idx)
            plt.close()
            
            print(f"  [{completed_tasks}/{total_tasks}] WaveIdx: {peak_idx}, "
                  f"InputWL: {actual_center_wavelength:.1f}nm, CWL: {recon_cwl:.1f}nm, "
                  f"Recon FWHM: {recon_fwhm:.2f}nm, dFWHM: {fwhm_diff_pct:+.2f}%")
        
        elapsed_time_fwhm = time.time() - start_time_fwhm
        print(f"  FWHM={input_fwhm}nm completed in {elapsed_time_fwhm:.2f} seconds")
        
        # Store summary data
        summary_input_wavelength[input_fwhm] = input_wl_list
        summary_recon_fwhm[input_fwhm] = fwhm_list
        summary_fwhm_diff_pct[input_fwhm] = fwhm_diff_pct_list
        summary_score[input_fwhm] = score_list
        
        # Plot distribution for this FWHM
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sub-plot 1: CWL vs Wave Index
        axes[0, 0].scatter(check_wave, cwl_list, color='blue', s=20)
        axes[0, 0].set_xlabel('Wave Index', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('CWL (nm)', fontsize=10, fontweight='bold')
        axes[0, 0].set_title(f'CWL vs Wave Index (Input FWHM={input_fwhm}nm)', fontsize=10)
        axes[0, 0].grid(True)

        # Sub-plot 2: Reconstructed FWHM vs Input Wavelength
        axes[0, 1].scatter(input_wl_list, fwhm_list, color='green', s=20)
        axes[0, 1].axhline(y=input_fwhm, color='red', linestyle='--', linewidth=1, label=f'Input FWHM={input_fwhm}nm')
        axes[0, 1].set_xlabel('Input Wavelength (nm)', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('Reconstructed FWHM (nm)', fontsize=10, fontweight='bold')
        axes[0, 1].set_title(f'Reconstructed FWHM vs Input Wavelength', fontsize=10)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True)

        # Sub-plot 3: dFWHM (%) vs Input Wavelength
        colors_diff = ['red' if d > 0 else 'blue' for d in fwhm_diff_pct_list]
        axes[1, 0].scatter(input_wl_list, fwhm_diff_pct_list, c=colors_diff, s=20)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_xlabel('Input Wavelength (nm)', fontsize=10, fontweight='bold')
        axes[1, 0].set_ylabel('dFWHM (%)', fontsize=10, fontweight='bold')
        axes[1, 0].set_title(f'FWHM Difference vs Input Wavelength', fontsize=10)
        axes[1, 0].grid(True)

        # Sub-plot 4: Spectral Score vs Input Wavelength
        axes[1, 1].scatter(input_wl_list, score_list, color='purple', s=20)
        axes[1, 1].set_xlabel('Input Wavelength (nm)', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('Spectral Score', fontsize=10, fontweight='bold')
        axes[1, 1].set_title(f'Spectral Score vs Input Wavelength', fontsize=10)
        axes[1, 1].grid(True)

        plt.tight_layout()
        dist_save_path = os.path.join(fwhm_save_folder, "cwl_fwhm_distribution.png")
        plt.savefig(dist_save_path)
        print(f"Saved distribution plot to: {dist_save_path}")
        plt.close()

        # Save CSV data
        data = np.column_stack((check_wave, input_wl_list, cwl_list, fwhm_list, fwhm_diff_pct_list, 
                                coverage_list, attenuation_list, score_list))
        csv_save_path = os.path.join(fwhm_save_folder, "cwl_fwhm_data.csv")
        np.savetxt(csv_save_path, data, delimiter=',',
                   header='Wave_Index,Input_Wavelength(nm),CWL(nm),Recon_FWHM(nm),Delta_FWHM(%),Coverage,Attenuation,Score',
                   fmt='%d,%.1f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f')
        print(f"Saved CSV data to: {csv_save_path}")

    # ========== Plot Summary Comparison (FWHM only) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.get_cmap('viridis')
    num_fwhm = len(input_fwhm_list)

    # Sub-plot 1: Reconstructed FWHM vs Input Wavelength
    for i, input_fwhm in enumerate(input_fwhm_list):
        color = cmap(i / max(num_fwhm - 1, 1))
        axes[0].plot(summary_input_wavelength[input_fwhm], summary_recon_fwhm[input_fwhm], '-o',
                     color=color, markersize=3, label=f'Input FWHM={input_fwhm}nm')
    axes[0].set_xlabel('Input Wavelength (nm)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Reconstructed FWHM (nm)', fontsize=11, fontweight='bold')
    axes[0].set_title('Reconstructed FWHM vs Input Wavelength', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=7, ncol=2, loc='upper left')
    axes[0].grid(True)

    # Sub-plot 2: dFWHM (%) vs Input Wavelength (log scale y-axis)
    for i, input_fwhm in enumerate(input_fwhm_list):
        color = cmap(i / max(num_fwhm - 1, 1))
        axes[1].semilogy(summary_input_wavelength[input_fwhm], 
                         np.abs(summary_fwhm_diff_pct[input_fwhm]), '-o',
                         color=color, markersize=3, label=f'Input FWHM={input_fwhm}nm')
    axes[1].set_xlabel('Input Wavelength (nm)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('|dFWHM| (%) - log scale', fontsize=11, fontweight='bold')
    axes[1].set_title('FWHM Difference (%) vs Input Wavelength (log scale)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=7, ncol=2, loc='upper left')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    summary_save_path = os.path.join(save_root, "fwhm_diff_summary.png")
    plt.savefig(summary_save_path)
    print(f"\nSaved summary plot to: {summary_save_path}")
    plt.close()

    # ========== Save Summary CSV ==========
    # fwhm_diff_summary.csv format
    fwhm_diff_summary_data = []
    spectral_score_summary_data = []
    for wl_idx, peak_idx in enumerate(check_wave):
        # Calculate actual wavelength from peak_idx
        input_wl = start_wavelength + peak_idx * wavelength_interval
        row_diff = [input_wl]
        row_score = [input_wl]
        for input_fwhm in input_fwhm_list:
            row_diff.append(summary_fwhm_diff_pct[input_fwhm][wl_idx])
            row_score.append(summary_score[input_fwhm][wl_idx])
        fwhm_diff_summary_data.append(row_diff)
        spectral_score_summary_data.append(row_score)
    
    # Save fwhm_diff_summary.csv
    fwhm_diff_header = 'Input_Wavelength(nm),' + ','.join([f'Delta_FWHM_pct_FWHM{f}nm' for f in input_fwhm_list])
    fwhm_diff_csv_path = os.path.join(save_root, "fwhm_diff_summary.csv")
    np.savetxt(fwhm_diff_csv_path, np.array(fwhm_diff_summary_data), delimiter=',',
               header=fwhm_diff_header, fmt='%.1f,' + ','.join(['%.2f' for _ in input_fwhm_list]))
    print(f"Saved FWHM diff summary CSV to: {fwhm_diff_csv_path}")
    
    # Save spectral_score_summary.csv
    score_header = 'Input_Wavelength(nm),' + ','.join([f'Spectral_Score_FWHM{f}nm' for f in input_fwhm_list])
    score_csv_path = os.path.join(save_root, "spectral_score_summary.csv")
    np.savetxt(score_csv_path, np.array(spectral_score_summary_data), delimiter=',',
               header=score_header, fmt='%.1f,' + ','.join(['%.4f' for _ in input_fwhm_list]))
    print(f"Saved spectral score summary CSV to: {score_csv_path}")

    # ========== Plot spectral_score_summary.png ==========
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, input_fwhm in enumerate(input_fwhm_list):
        color = cmap(i / max(num_fwhm - 1, 1))
        ax.plot(summary_input_wavelength[input_fwhm], summary_score[input_fwhm], '-o',
                color=color, markersize=4, label=f'Input FWHM={input_fwhm}nm')
    
    ax.set_xlabel('Input Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spectral Score', fontsize=12, fontweight='bold')
    ax.set_title('Spectral Score vs Input Wavelength for Different FWHM', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    spectral_score_png_path = os.path.join(save_root, "spectral_score_summary.png")
    plt.savefig(spectral_score_png_path)
    print(f"Saved spectral score summary plot to: {spectral_score_png_path}")
    plt.close()

    print(f"\n{'='*60}")
    print(f"All processing completed. Results saved in: {save_root}")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')

    file_path = r'F:\05-Jerome Studios\Coating Design\Coating_data\UDW450'

    name_list = os.listdir(file_path)

    # for name in name_list:
    #     new_uc450 = os.path.join(file_path, name)
    #     # Call main function
    #     lst_with_aug_reg_400_700_wave_index_loop(new_uc450)

    name = r"202604160938-U500_MEMS_Drift_450_J13"
    new_uc450 = os.path.join(file_path, name)
    lst_with_aug_reg_400_700_wave_index_loop(new_uc450)