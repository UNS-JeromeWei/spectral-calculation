import os
import glob
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import spectral
# from cubes_estimate import load_lab_sphere_list
# from cubes_estimate import get_file_by_extension
# from cubes_estimate import cube_to_curve
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def make_sum_regions():
    total_num = 650
    region_num = 70
    first_end = 56
    step_len = 8
    region_list = [[0, first_end + 1]]
    for i in range(region_num - 2):
        region_list.append([first_end + 1 + i * step_len, first_end + 1 + (i + 1) * step_len])

    region_list.append([first_end + 1 + (region_num - 3 + 1) * step_len, total_num])
    return region_list


def draw_curves(x, ys, title=""):
    x = np.array(x)
    x_dense = np.linspace(x[0], x[-1], len(x) * 10)
    waves = np.arange(713, 920 + 1, 23)

    for c_id, y in enumerate(ys):
        y = np.array(y)
        # plt.plot(x, y, label=' ', color='green')
        # plt.plot(x, y, label=str(waves[c_id % len(waves)]), color=COLORS[c_id % len(COLORS)])
        plt.plot(x, y, color=COLORS[c_id % len(COLORS)])
        cs = CubicSpline(x, y)
        y_dense = cs(x_dense)
        # plt.plot(x_dense, y_dense, label=' ', linestyle='--', color='gold')
        # plt.plot(x_dense, y_dense, label=' ', linestyle='--', color=COLORS[c_id % len(COLORS)])
        plt.plot(x_dense, y_dense, linestyle='--', color=COLORS[c_id % len(COLORS)])
        # print(find_cwl_in_curve(x_dense, y_dense))

    # Add labels and title
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(title)

    # Show legend
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Adds grid lines for better readability
    plt.show()


def draw_curves_no_smooth(x, ys):
    x = np.array(x)
    x_dense = np.linspace(x[0], x[-1], len(x) * 10)
    waves = np.arange(713, 920 + 1, 3)

    for c_id, y in enumerate(ys):
        y = np.array(y)
        # plt.plot(x, y, label=' ', color='green')
        plt.plot(x, y, label=str(waves[c_id]), ls='--', color=COLORS[c_id % len(COLORS)])
        cs = CubicSpline(x, y)
        y_dense = cs(x_dense)
        # plt.plot(x_dense, y_dense, label=' ', linestyle='--', color='gold')
        # plt.plot(x_dense, y_dense, label=' ', linestyle='--', color=COLORS[c_id % len(COLORS)])
        # plt.plot(x_dense, y_dense, linestyle='--', color=COLORS[c_id % len(COLORS)])
        # print(find_cwl_in_curve(x_dense, y_dense))

    # Add labels and title
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Multiple Curves on One Graph')

    # Show legend
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Adds grid lines for better readability
    plt.show()


def mix_curves_2(x_1, y_1, x_2, y_2):
    # x = np.array(x)
    x_left = max(x_1[0], x_2[0])
    x_right = min(x_1[-1], x_2[-1])
    # x_dense_1 = np.linspace(x_1[0], x_1[-1], len(x_1) * 10)
    # x_dense_2 = np.linspace(x_2[0], x_2[-1], len(x_2) * 10)

    x_dense = np.linspace(x_left, x_right, (len(x_1) + len(x_2)) * 5)
    # combined_x = np.concatenate((x_dense_1, x_dense_2))
    # sorted_x = np.sort(combined_x)

    # cs_1 = CubicSpline(x_1, y_1)
    cs_1 = interp1d(x_1, y_1, kind='linear')
    y_dense_1 = cs_1(x_dense)

    # cs_2 = CubicSpline(x_2, y_2)
    cs_2 = interp1d(x_2, y_2, kind='linear')
    y_dense_2 = cs_2(x_dense)

    plt.plot(x_dense, y_dense_1, label=str(), ls='-', color=COLORS[1 % len(COLORS)])
    plt.plot(x_dense, y_dense_2, label=str(), ls='-', color=COLORS[2 % len(COLORS)])
    #
    #
    # waves = np.arange(713, 920 + 1, 3)
    #
    # for c_id, y in enumerate(ys):
    #     y = np.array(y)
    #     # plt.plot(x, y, label=' ', color='green')
    #     plt.plot(x, y, label=str(waves[c_id]), ls='--', color=COLORS[c_id % len(COLORS)])
    #     cs = CubicSpline(x, y)
    #     y_dense = cs(x_dense)
    #     # plt.plot(x_dense, y_dense, label=' ', linestyle='--', color='gold')
    #     # plt.plot(x_dense, y_dense, label=' ', linestyle='--', color=COLORS[c_id % len(COLORS)])
    #     # plt.plot(x_dense, y_dense, linestyle='--', color=COLORS[c_id % len(COLORS)])
    #     # print(find_cwl_in_curve(x_dense, y_dense))

    # Add labels and title
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Multiple Curves on One Graph')

    # Show legend
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Adds grid lines for better readability
    plt.show()


def draw_map_of_2d_angle(data, axis):
    # print(data)
    print("csv image shape: ", data.shape)

    plt.imshow(data, aspect='auto', extent=axis, cmap="rainbow")
    plt.colorbar()
    plt.xlabel("X axis (e.g. wavelength)")
    plt.ylabel("Y axis (e.g. sample index)")
    plt.title("2D Matrix as Image with X Range")
    plt.show()


def get_wave_lens_from_folder_name(cubes_dir):
    pattern = "cube*"
    wave_lens = [int(os.path.basename(d).replace("cube_", "")) // 10 for d in glob.glob(os.path.join(cubes_dir, pattern)) if
                        os.path.isdir(d)]
    return wave_lens


def get_part_sum_list(in_curve, part_list):
    total_energy = np.sum(in_curve)
    sum_list = []
    for part in part_list:
        # print(in_curve[part[0]:part[1]])
        sum_list.append(np.sum(in_curve[part[0]:part[1]]) / total_energy)

    return sum_list


def out_x_plan(x_list, part_list):
    x_plan = []
    for part in part_list:
        # x_plan.append([x_list[part[0]], x_list[part[1] - 1]])
        x_plan.append((x_list[part[0]] + x_list[part[1] - 1]) / 2)

    print("x_plan", len(x_plan), x_plan)
    return x_plan


def reconstruct_curves(x, ys):
    x_dense = np.linspace(x[0], x[-1], len(x) * 10)
    print("len(x_dense)", len(x_dense))
    sum_plan = make_sum_regions()
    print("sum_plan", sum_plan)
    coefficient_matrix = []
    print("x_dense[sum_plan[1][0]:sum_plan[1][1]]]", x_dense[sum_plan[1][0]:sum_plan[1][1]])
    for y in ys:
        cs = CubicSpline(x, y)
        y_dense = cs(x_dense)
        # print("np.sum(curve)", np.sum(y_dense))

        sum_list = get_part_sum_list(y_dense, sum_plan)
        # print("sum part", sum_list)
        # print("sum sum list", np.sum(sum_list))
        coefficient_matrix.append(sum_list)

    x_plan = out_x_plan(x_dense, sum_plan)
    coefficient_matrix = np.array(coefficient_matrix)
    # coefficient_matrix[coefficient_matrix < 0.01] = 0
    print("coefficient_matrix.shape", coefficient_matrix.shape)
    # for i in range(70):
    #     coefficient_matrix[i, i] = 0.0001

    return coefficient_matrix, x_plan


def rebuild_test():
    cubes_path = r"D:\Unispectral\python_code\computational_spectral_imaging\cubes_1A0"
    reconstructed_curves = []

    curves_array = np.load('curvearray.npy')
    wave_lens = get_wave_lens_from_folder_name(cubes_path)
    print(curves_array.shape)
    coef_matrix, x_plan = reconstruct_curves(wave_lens, curves_array)
    loop_array = np.transpose(curves_array)
    print("np.linalg.det(coef_matrix)", np.linalg.det(coef_matrix))
    print("np.linalg.matrix_rank(coef_matrix)", np.linalg.matrix_rank(coef_matrix))
    for curve in loop_array:
        # print("curve.shape", curve.shape)
        x = np.linalg.solve(coef_matrix, curve)
        reconstructed_curves.append(x)
        # print("x.shape", x.shape)

    # draw_curves(wave_lens, curves_array)

    print("np.linalg.det(coef_matrix)", np.linalg.det(coef_matrix))
    print("np.linalg.matrix_rank(coef_matrix)", np.linalg.matrix_rank(coef_matrix))

    # draw_curves(np.arange(70), reconstructed_curves[:])

    draw_curves_no_smooth(x_plan, coef_matrix)


def load_one_cube():
    sub_dir = r"D:\Unispectral\python_code\computational_spectral_imaging\single_cube\cube_8000"
    sub_dir = r"D:\Unispectral\python_code\computational_spectral_imaging\valid_cubes\cube_9100"
    sub_dir = r"D:\Unispectral\python_code\computational_spectral_imaging\valid_cubes_5\cube_8020"
    sub_dir = r"D:\Unispectral\python_code\computational_spectral_imaging\valid_cubes_5\cube_7920"
    # sr_files_path = r"D:\Unispectral\python_code\computational_spectral_imaging\LabSphere_Grating2_slit1200_ver23"
    # cubes_path = r"D:\Unispectral\python_code\computational_spectral_imaging\single_cube"
    #
    #
    # lab_sphere_list = load_lab_sphere_list(sr_files_path)
    # lab_sphere_list = sorted(lab_sphere_list, key=lambda lab_sphere: lab_sphere.mono_wave)
    #
    # modes = [lls.mono_wave for lls in lab_sphere_list]
    # print("modes", len(modes), modes)
    #
    # norm_factor = [lls.irradiance_integ for lls in lab_sphere_list]
    # print("norm_factor", len(norm_factor), norm_factor)
    #
    # f_linear = interp1d(modes, norm_factor)
    # # print("f_linear(700)", f_linear(700))

    wave_len = int(os.path.basename(sub_dir).replace("cube_", "")) // 10
    hdr_file = get_file_by_extension(sub_dir, "hdr")
    raw_file = get_file_by_extension(sub_dir, "raw")

    init_cube = spectral.envi.open(hdr_file, raw_file).load(dtype=np.float32).asarray().copy()
    cube = init_cube.transpose(2, 0, 1)
    cube = (cube - 64)  #/ f_linear(wave_len)
    print(cube.shape)

    return cube


def find_fwhm_normal(in_x, in_y):
    """ get fwhm

        :param : wavelengths: x value list, y_points: y value list
        :type : [], []
        :return: left_idx, right_idx, half_peak
        :rtype: int, int, float
    """
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
    # return left_idx, right_idx, half_peak


def smoothness_loss_1d(y):
    diff = y[1:] - y[:-1]
    return torch.mean(diff ** 2)


def smoothness_loss_2nd(y):
    diff2 = y[2:] - 2*y[1:-1] + y[:-2]
    return torch.mean(diff2 ** 2)


def pytorch_optimizor(in_matrix, in_vector):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    n = in_matrix.shape[0]

    # Define C, D, and initialize x
    # C = torch.rand(n, requires_grad=False)  # Target 1D vector of size 70
    # D = torch.rand(n, n, requires_grad=False)  # Matrix of size 70x70
    C = torch.from_numpy(in_vector).to(dtype=torch.double).to(device)
    A = torch.from_numpy(in_matrix).to(dtype=torch.double).to(device)
    x = torch.rand(n, requires_grad=True).to(dtype=torch.double).to(device) # Initial vector of size 70
    x = x.detach()
    x.requires_grad = True

    # Regularization parameter
    # lambda_reg = 0.1
    lambda_reg = 0.00001
    penalty_weight = 0.0001
    # penalty_weight = 0
    # learning_rate = 0.1
    learning_rate = 0.01

    # Define the objective function
    def objective(x):
        diff = C - A @ x  # Difference between C and D * x
        term1 = torch.norm(diff, p=2) ** 2  # L2 norm of the difference
        term2 = lambda_reg * torch.norm(x, p=2) ** 2  # Regularization term
        penalty = penalty_weight * torch.sum(torch.relu(-x))
        return term1 + term2 + penalty + 0.1 * (smoothness_loss_1d(x) + smoothness_loss_2nd(x))

    # Use an optimizer to minimize the objective
    optimizer = optim.Adam([x], lr=learning_rate)

    # Training loop
    num_iterations = 20000
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = objective(x)
        loss.backward()
        optimizer.step()

        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

    # Final optimized vector
    print("\nOptimized vector x:\n", x.cpu().detach().numpy())

    return x.cpu().detach().numpy()


def plot_simple_curve(x, y, in_title):
    # x = [0, 1, 2, 3, 4]
    # y = [0, 1, 4, 9, 16]

    plt.plot(x, y,  linestyle='-',)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(in_title)
    plt.show()


def plot_simple_multi_curves(x, ys, in_label):
    # x = [0, 1, 2, 3, 4]
    # y = [0, 1, 4, 9, 16]

    for idx, y in enumerate(ys):
        plt.plot(x, y, color=COLORS[idx])
        plt.xlabel("x")
        plt.ylabel("y " + str(idx))


    plt.title(in_label)
    plt.show()


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def load_matrix_from_file(src_dir):
    wavelengths, transmittances = [], []

    files = sorted(os.listdir(src_dir))
    # print("files", files)

    for f_name in files:
        file_path = os.path.join(src_dir, f_name)
        waves, trans = [], []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= 1:
                    items = line.split(chr(0x09))
                    # if is_integer(items[0].strip()):
                    wave_f = float(items[0].strip())
                    if wave_f - wave_f // 1 < 0.000001 and 350 <= int(wave_f) <= 800:
                        waves.append(int(wave_f))
                        trans.append(float(items[2].strip()) / 100)
        wavelengths.append(waves)
        transmittances.append(trans)

    transmittances = np.array(transmittances)
    row_sums = transmittances.sum(axis=1, keepdims=True)
    # transmittances = transmittances / row_sums
    max_row_sum = np.max(row_sums)
    print("max_row_sum", max_row_sum)
    # transmittances = transmittances / max_row_sum

    return np.array(wavelengths), transmittances


def gaussian_beam(x, mu, sig):
    return np.exp(-np.power(np.array(x) - mu, 2.) / (2 * np.power(sig, 2.)))


def cubic_spline_sim_1(wavelengths):
    fix_num = 8
    step_wide = int(len(wavelengths) / fix_num)
    x = np.array([wavelengths[step_wide * i] for i in range(fix_num + 1)])

    # rng = np.random.default_rng(seed=42)
    rng = np.random.default_rng()
    y = rng.uniform(0, 0.8, size=fix_num + 1)

    cs = CubicSpline(x, y, bc_type='natural')
    y_new = cs(wavelengths)

    return y_new


def random_peak_sum_with_noise(wavelengths):
    x = wavelengths.astype(np.float64, copy=False)
    y = np.zeros_like(x)

    for _ in range(5):
        center = np.random.uniform(wavelengths[0], wavelengths[-1] - 1)
        width = np.random.uniform(30, 100)
        amp = np.random.uniform(0.1, 0.9)
        y += amp * np.exp(-(x - center) ** 2 / (2 * width ** 2))

    y += 0.05 * np.random.randn(len(x))
    return y


def gaussian_smooth_curve(wavelengths):
    y = np.random.randn(len(wavelengths)) + 0.5
    y_smooth = gaussian_filter1d(y, sigma=20)

    return y_smooth


def fourier_curve(wavelengths):
    x = wavelengths.astype(np.float64, copy=False)
    y = np.zeros_like(x)

    for k in range(1, 10):
        a = np.random.randn() * 0.1
        b = np.random.randn() * 0.1
        y += a * np.sin(k / 100.0 * x) + b * np.cos(k / 100.0 * x)

    return y + 0.8


def savgol_filter_curve(wavelengths):
    y = np.random.randn(len(wavelengths))
    y_smooth = savgol_filter(y, window_length=101, polyorder=3) + 0.5

    return y_smooth


def rebuild_test_one_curve(src_dir):
    # cubes_path = r"D:\Unispectral\python_code\computational_spectral_imaging\cubes_1A0"
    # reconstructed_curves = []
    #
    # curves_array = np.load('curvearray.npy')
    # # wave_lens = get_wave_lens_from_folder_name(cubes_path)
    # wave_lens = np.arange(660, 980 + 1, 5)
    # print(curves_array.shape)
    # coef_matrix, x_plan = reconstruct_curves(wave_lens, curves_array)
    # print("wave_lens:", wave_lens)
    # loop_array = np.transpose(curves_array)
    # print("np.linalg.det(coef_matrix)", np.linalg.det(coef_matrix))
    # print("np.linalg.matrix_rank(coef_matrix)", np.linalg.matrix_rank(coef_matrix))
    #
    # print("coef_matrix[0]", coef_matrix[0], np.sum(coef_matrix[0]), np.sum(coef_matrix[1]))
    #
    # cube = load_one_cube()
    # curve = cube_to_curve(cube)

    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)

    print("new_coef_matrix.shape", new_coef_matrix.shape)

    # new_coef_matrix = new_coef_matrix[0: 350]
    selected_index = 80

    # print("new_coef_matrix[100]", new_coef_matrix[100])

    # print("wave_lens", wave_lens.shape)
    # print("new_coef_matrix[0].shape", new_coef_matrix[0].shape)
    plt.imshow(new_coef_matrix, aspect='auto', extent=[400, 1100, 0, 700])
    plt.colorbar()
    plt.xlabel("X axis (e.g. wavelength)")
    plt.ylabel("Y axis (e.g. sample index)")
    plt.title("2D Matrix as Image with X Range")
    plt.show()

    # plt.colorbar(label='Transmittance')

    # plt.imshow(new_coef_matrix, aspect='auto', extent=[wave_lens[0], wave_lens[-1], len(wave_lens) - 1, 0], cmap='jet')

    plot_simple_curve(wave_lens[0], new_coef_matrix[300], "Resp Matrix Section")

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1
    # arr1 = 1.0 * gaussian_beam(np.arange(0, len(arr), 1.0), 450, 15 / 2.355)
    # arr1 = 0.8 * gaussian_beam(np.arange(0, len(arr), 1.0), 50, 10)
    arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 180, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 400, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 600, 14)
    # arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))

    # arr1 = random_peak_sum_with_noise(wave_lens[0])
    # arr1 = gaussian_smooth_curve(wave_lens[0])
    # arr1 = fourier_curve(wave_lens[0])
    # arr1 = savgol_filter_curve(wave_lens[0])

    # arr1[200] = 0.5
    # arr1[300] = 0.3
    print("arr1.T", arr1.T.shape)
    plot_simple_curve(wave_lens[0], arr1, "Org Curve")

    C = new_coef_matrix @ arr1.T

    curve = new_coef_matrix @ arr1.T
    curve += np.random.rand(len(arr1)) * 0.01 * curve

    # window = 10
    # curve = gaussian_filter1d(curve, sigma=window)
    # curve2 = np.convolve(curve, np.ones(window) / window, mode='same')
    # curve2[:window // 2], curve2[-window // 2:] = curve[:window // 2], curve[-window // 2:]
    # curve = curve2

    my_x_axis = np.arange(1, len(C) + 1)
    plot_simple_curve(my_x_axis, curve, "Trans Curve With Noise")

    print("new_coef_matrix", new_coef_matrix)

    print("curve.shape", curve.shape)
    x = pytorch_optimizor(new_coef_matrix, curve)

    print("x.shape", x.shape)

    print("wave_len", wave_lens[0][selected_index])
    # plot_simple_curve(my_x_axis, x)

    plot_simple_multi_curves(wave_lens[0], [x, arr1], "Rebuild Curve. Blue new; Orange org")

    window = 8
    # np.convolve(x, np.ones(window) / window, mode='same')
    # avg_x = np.convolve(x, np.ones(window) / window, mode='same')
    # avg_x[:window // 2], avg_x[-window // 2:] = x[:window // 2], x[-window // 2:]
    # avg_x = savgol_filter(x, window_length=window, polyorder=3)
    avg_x = gaussian_filter1d(x, sigma=window)
    plot_simple_multi_curves(wave_lens[0], [avg_x, arr1], "Smoothed Curve")

    # # x = np.linalg.solve(coef_matrix, curve)
    # print("fwhm, cwl org", find_fwhm_normal(np.arange(713, 920 + 1, 3), curve))
    # fitted_x = np.linspace(690, 958, 68)
    # print("fwhm, cwl new", find_fwhm_normal(fitted_x, x[1:len(x) - 1]))
    #
    # draw_curves(fitted_x, [x[1:len(x) - 1]], "new curve")
    # draw_curves(np.arange(713, 920 + 1, 3), [curve], "origin curve")
    # # mix_curves_2(np.arange(713, 920 + 1, 3), curve, fitted_x, x[1:len(x) - 1])


def rebuild_test_one_curve_4_new_uc500(src_dir):
    # cubes_path = r"D:\Unispectral\python_code\computational_spectral_imaging\cubes_1A0"
    # reconstructed_curves = []
    #
    # curves_array = np.load('curvearray.npy')
    # # wave_lens = get_wave_lens_from_folder_name(cubes_path)
    # wave_lens = np.arange(660, 980 + 1, 5)
    # print(curves_array.shape)
    # coef_matrix, x_plan = reconstruct_curves(wave_lens, curves_array)
    # print("wave_lens:", wave_lens)
    # loop_array = np.transpose(curves_array)
    # print("np.linalg.det(coef_matrix)", np.linalg.det(coef_matrix))
    # print("np.linalg.matrix_rank(coef_matrix)", np.linalg.matrix_rank(coef_matrix))
    #
    # print("coef_matrix[0]", coef_matrix[0], np.sum(coef_matrix[0]), np.sum(coef_matrix[1]))
    #
    # cube = load_one_cube()
    # curve = cube_to_curve(cube)

    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    cut_index = 501
    wave_lens = wave_lens[:cut_index, :cut_index]
    new_coef_matrix = new_coef_matrix[:cut_index, :cut_index]
    # wave_lens = wave_lens[len(wave_lens) - cut_index:, :cut_index]
    # new_coef_matrix = new_coef_matrix[len(new_coef_matrix) - cut_index:, :cut_index]
    print("wave_lens.shape", wave_lens.shape)
    print("new_coef_matrix.shape", new_coef_matrix.shape)


    # new_coef_matrix = new_coef_matrix[0: 350]
    selected_index = 80

    # print("new_coef_matrix[100]", new_coef_matrix[100])

    # print("wave_lens", wave_lens.shape)
    # print("new_coef_matrix[0].shape", new_coef_matrix[0].shape)
    plt.imshow(new_coef_matrix, aspect='auto', extent=[400, 400 + cut_index - 1, len(wave_lens) - cut_index, len(wave_lens)])
    plt.colorbar()
    plt.xlabel("X axis (e.g. wavelength)")
    plt.ylabel("Y axis (e.g. sample index)")
    plt.title("2D Matrix as Image with X Range")
    plt.show()

    # plt.colorbar(label='Transmittance')

    # plt.imshow(new_coef_matrix, aspect='auto', extent=[wave_lens[0], wave_lens[-1], len(wave_lens) - 1, 0], cmap='jet')

    plot_simple_curve(wave_lens[0], new_coef_matrix[300], "Resp Matrix Section")

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1
    # arr1 = 1.0 * gaussian_beam(np.arange(0, len(arr), 1.0), 450, 15 / 2.355)
    # arr1 = 0.8 * gaussian_beam(np.arange(0, len(arr), 1.0), 50, 10)
    arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 100, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 200, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 350, 14)
    # arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))

    # arr1 = random_peak_sum_with_noise(wave_lens[0])
    # arr1 = gaussian_smooth_curve(wave_lens[0])
    # arr1 = fourier_curve(wave_lens[0])
    # arr1 = savgol_filter_curve(wave_lens[0])

    # arr1[200] = 0.5
    # arr1[300] = 0.3
    print("arr1.T", arr1.T.shape)
    plot_simple_curve(wave_lens[0], arr1, "Org Curve")

    C = new_coef_matrix @ arr1.T

    curve = new_coef_matrix @ arr1.T
    # curve += np.random.rand(len(arr1)) * 0.01 * curve

    # window = 10
    # curve = gaussian_filter1d(curve, sigma=window)
    # curve2 = np.convolve(curve, np.ones(window) / window, mode='same')
    # curve2[:window // 2], curve2[-window // 2:] = curve[:window // 2], curve[-window // 2:]
    # curve = curve2

    # my_x_axis = np.arange(1, len(C) + 1)
    # plot_simple_curve(my_x_axis, curve, "Trans Curve With Noise")

    # print("new_coef_matrix", new_coef_matrix)

    # print("curve.shape", curve.shape)
    x = pytorch_optimizor(new_coef_matrix, curve)

    print("x.shape", x.shape)

    print("wave_len", wave_lens[0][selected_index])
    # plot_simple_curve(my_x_axis, x)

    plot_simple_multi_curves(wave_lens[0], [x, arr1], "Rebuild Curve. Blue new; Orange org")

    window = 8
    # np.convolve(x, np.ones(window) / window, mode='same')
    # avg_x = np.convolve(x, np.ones(window) / window, mode='same')
    # avg_x[:window // 2], avg_x[-window // 2:] = x[:window // 2], x[-window // 2:]
    # avg_x = savgol_filter(x, window_length=window, polyorder=3)
    avg_x = gaussian_filter1d(x, sigma=window)
    plot_simple_multi_curves(wave_lens[0], [avg_x, arr1], "Smoothed Curve")

    # # x = np.linalg.solve(coef_matrix, curve)
    # print("fwhm, cwl org", find_fwhm_normal(np.arange(713, 920 + 1, 3), curve))
    # fitted_x = np.linspace(690, 958, 68)
    # print("fwhm, cwl new", find_fwhm_normal(fitted_x, x[1:len(x) - 1]))
    #
    # draw_curves(fitted_x, [x[1:len(x) - 1]], "new curve")
    # draw_curves(np.arange(713, 920 + 1, 3), [curve], "origin curve")
    # # mix_curves_2(np.arange(713, 920 + 1, 3), curve, fitted_x, x[1:len(x) - 1])


def lst_no_reg(src_dir):
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)

    plt.imshow(new_coef_matrix, aspect='auto', extent=[400, 1100, 0, 700])
    plt.colorbar()
    plt.xlabel("X axis (e.g. wavelength)")
    plt.ylabel("Y axis (e.g. sample index)")
    plt.title("2D Matrix as Image with X Range")
    plt.show()
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1

    arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 180, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 400, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 600, 14)
    arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6)
    # arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))
    plot_simple_curve(wave_lens[0], arr1, "in curve")
    C = new_coef_matrix @ arr1.T

    y_measured = new_coef_matrix @ arr1.T

    # y_measured += np.random.rand(len(arr1)) * 0.001 * y_measured
    plot_simple_curve(wave_lens[0], y_measured, "trans curve")

    res = lsq_linear(new_coef_matrix, y_measured, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    x_rec_nnls = res.x
    print(x_rec_nnls.shape)
    plot_simple_multi_curves(wave_lens[0], [x_rec_nnls, arr1], "out curve")


def lst_no_reg_new_uc500(src_dir):
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    cut_index = 501
    wave_lens = wave_lens[:cut_index, :cut_index]
    new_coef_matrix = new_coef_matrix[:cut_index, :cut_index]
    print("wave_lens.shape", wave_lens.shape)
    print("new_coef_matrix.shape", new_coef_matrix.shape)

    plt.imshow(new_coef_matrix, aspect='auto',
               extent=[400, 400 + cut_index - 1, len(wave_lens) - cut_index, len(wave_lens)])
    plt.colorbar()
    plt.xlabel("X axis (e.g. wavelength)")
    plt.ylabel("Y axis (e.g. sample index)")
    plt.title("2D Matrix as Image with X Range")
    plt.show()

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1

    # arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 120, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 300, 14)
    # arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))
    arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6)
    plot_simple_curve(wave_lens[0], arr1, "in curve")
    C = new_coef_matrix @ arr1.T

    y_measured = new_coef_matrix @ arr1.T

    # y_measured += np.random.rand(len(arr1)) * 0.001 * y_measured
    plot_simple_curve(wave_lens[0], y_measured, "trans curve")

    res = lsq_linear(new_coef_matrix, y_measured, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    x_rec_nnls = res.x
    print(x_rec_nnls.shape)
    plot_simple_multi_curves(wave_lens[0], [x_rec_nnls, arr1], "out curve")


def build_D2(L: int) -> np.ndarray:
    # (L-2) x L second-difference matrix
    D = np.zeros((L - 2, L), dtype=np.float64)
    for i in range(L - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def lst_with_aug_reg(src_dir):
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    wave_len_list = wave_lens[0]
    # cut_index = 700
    # wave_lens = wave_lens[:cut_index, :cut_index]
    # new_coef_matrix = new_coef_matrix[:cut_index, :cut_index]
    new_coef_matrix = new_coef_matrix[:len(wave_len_list), :]
    print("wave_lens.shape", wave_lens.shape)
    print("new_coef_matrix.shape", new_coef_matrix.shape)

    draw_map_of_2d_angle(new_coef_matrix, [wave_len_list[0], wave_len_list[-1], len(wave_len_list), 0])

    # plt.imshow(new_coef_matrix, aspect='auto',
    #            extent=[400, 400 - 1, len(wave_lens), len(wave_lens)])
    # plt.colorbar()
    # plt.xlabel("X axis (e.g. wavelength)")
    # plt.ylabel("Y axis (e.g. sample index)")
    # plt.title("2D Matrix as Image with X Range")
    # plt.show()

    lam = 200
    D2 = build_D2(len(wave_len_list))
    A_aug = np.vstack([new_coef_matrix, np.sqrt(lam) * D2])

    # plt.imshow(new_coef_matrix, aspect='auto',
    #            extent=[400, 400 + cut_index - 1, len(wave_lens) - cut_index, len(wave_lens)])

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1

    arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 202, 6)
    # arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 120, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6)
    # arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 120, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 300, 14)
    # arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))
    # arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6)

    plot_simple_curve(wave_len_list, arr1, "in curve")
    # C = new_coef_matrix @ arr1.T

    y_measured = new_coef_matrix @ arr1.T
    y_measured += np.random.rand(len(arr1)) * 0.01 * y_measured

    y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])

    # y_measured += np.random.rand(len(arr1)) * 0.001 * y_measured
    plot_simple_curve(wave_len_list, y_measured, "trans curve")

    # res = lsq_linear(new_coef_matrix, y_measured, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    res = lsq_linear(A_aug, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
    x_rec_nnls = res.x
    print(x_rec_nnls.shape)
    plot_simple_multi_curves(wave_len_list, [x_rec_nnls, arr1], "")


def lst_with_aug_reg_with_loop(src_dir):
    # wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    # cut_index = 501
    # wave_lens = wave_lens[:cut_index, :cut_index]
    # new_coef_matrix = new_coef_matrix[:cut_index, :cut_index]
    # print("wave_lens.shape", wave_lens.shape)
    # print("new_coef_matrix.shape", new_coef_matrix.shape)
    wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    wave_len_list = wave_lens[0]
    new_coef_matrix = new_coef_matrix[:len(wave_len_list), :]
    print("wave_len_list.shape", wave_lens.shape)
    print("new_coef_matrix.shape", new_coef_matrix.shape)

    lam = 0.01
    D2 = build_D2(len(wave_len_list))
    A_aug = np.vstack([new_coef_matrix, np.sqrt(lam) * D2])

    # plt.imshow(new_coef_matrix, aspect='auto', extent=[400, 400 + cut_index - 1, len(wave_lens) - cut_index, len(wave_lens)])
    # plt.colorbar()
    # plt.xlabel("X axis (e.g. wavelength)")
    # plt.ylabel("Y axis (e.g. sample index)")
    # plt.title("2D Matrix as Image with X Range")
    # plt.show()

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1

    for peak_idx in list(range(40, 460, 40)):

        arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), peak_idx, 6)
        # arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 120, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6)
        # arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 120, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 300, 14)
        # arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))
        # arr1 = 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 220, 6)
        
        plot_simple_curve(wave_len_list, arr1, "in curve")
        # C = new_coef_matrix @ arr1.T
        
        y_measured = new_coef_matrix @ arr1.T
        
        y_aug = np.concatenate([y_measured, np.zeros(D2.shape[0], dtype=np.float64)])
        
        # y_measured += np.random.rand(len(arr1)) * 0.001 * y_measured
        plot_simple_curve(wave_len_list, y_measured, "trans curve")
        
        # res = lsq_linear(new_coef_matrix, y_measured, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
        res = lsq_linear(A_aug, y_aug, bounds=(-0.0, np.inf), lsmr_tol='auto', verbose=0)
        x_rec_nnls = res.x
        print(x_rec_nnls.shape)
        plot_simple_multi_curves(wave_len_list, [x_rec_nnls, arr1], str(peak_idx + 400))


def rebuild_test_one_curve_from_file(npy_file):
    # cubes_path = r"D:\Unispectral\python_code\computational_spectral_imaging\cubes_1A0"
    # reconstructed_curves = []
    #
    # curves_array = np.load('curvearray.npy')
    # # wave_lens = get_wave_lens_from_folder_name(cubes_path)
    # wave_lens = np.arange(660, 980 + 1, 5)
    # print(curves_array.shape)
    # coef_matrix, x_plan = reconstruct_curves(wave_lens, curves_array)
    # print("wave_lens:", wave_lens)
    # loop_array = np.transpose(curves_array)
    # print("np.linalg.det(coef_matrix)", np.linalg.det(coef_matrix))
    # print("np.linalg.matrix_rank(coef_matrix)", np.linalg.matrix_rank(coef_matrix))
    #
    # print("coef_matrix[0]", coef_matrix[0], np.sum(coef_matrix[0]), np.sum(coef_matrix[1]))
    #
    # cube = load_one_cube()
    # curve = cube  _to_curve(cube)

    new_coef_matrix = np.load(npy_file)
    row_sum_max = np.max(np.sum(new_coef_matrix, axis=1))
    print("row_sum_max", row_sum_max)
    new_coef_matrix = new_coef_matrix / row_sum_max

    wave_lens = [np.arange(400, 1100 + 1)]

    print("new_coef_matrix.shape", new_coef_matrix.shape)
    #
    # wave_lens, new_coef_matrix = load_matrix_from_file(src_dir)
    # new_coef_matrix = new_coef_matrix[0: 350]
    selected_index = 80

    # print("new_coef_matrix[100]", new_coef_matrix[100])

    # print("wave_lens", wave_lens.shape)
    # print("new_coef_matrix[0].shape", new_coef_matrix[0].shape)
    plt.imshow(new_coef_matrix, aspect='auto', extent=[400, 1100, 0, 700])
    plt.colorbar()
    plt.xlabel("X axis (e.g. wavelength)")
    plt.ylabel("Y axis (e.g. sample index)")
    plt.title("2D Matrix as Image with X Range")
    plt.show()

    # plt.colorbar(label='Transmittance')

    # plt.imshow(new_coef_matrix, aspect='auto', extent=[wave_lens[0], wave_lens[-1], len(wave_lens) - 1, 0], cmap='jet')

    plot_simple_curve(wave_lens[0], new_coef_matrix[300], "Resp Matrix Section")

    arr = np.zeros(new_coef_matrix.shape[0], dtype=float)
    # for i in range(350):
    # arr1 = arr.copy()
    # arr1[selected_index] = 1
    # arr1 = 1.0 * gaussian_beam(np.arange(0, len(arr), 1.0), 450, 15 / 2.355)
    # arr1 = 0.8 * gaussian_beam(np.arange(0, len(arr), 1.0), 50, 10)
    # arr1 = gaussian_beam(np.arange(0, len(arr), 1.0), 180, 10) + 0.6 * gaussian_beam(np.arange(0, len(arr), 1.0), 400, 6) + 1.2 * gaussian_beam(np.arange(0, len(arr), 1.0), 600, 14)
    arr1 = cubic_spline_sim_1(np.arange(0, len(arr), 1.0))

    # arr1 = random_peak_sum_with_noise(wave_lens[0])
    # arr1 = gaussian_smooth_curve(wave_lens[0])
    # arr1 = fourier_curve(wave_lens[0])
    # arr1 = savgol_filter_curve(wave_lens[0])

    # arr1[200] = 0.5
    # arr1[300] = 0.3
    print("arr1.T", arr1.T.shape)
    plot_simple_curve(wave_lens[0], arr1, "Org Curve")

    C = new_coef_matrix @ arr1.T

    curve = new_coef_matrix @ arr1.T
    curve += np.random.rand(len(arr1)) * 0.01 * curve

    # window = 10
    # curve = gaussian_filter1d(curve, sigma=window)
    # curve2 = np.convolve(curve, np.ones(window) / window, mode='same')
    # curve2[:window // 2], curve2[-window // 2:] = curve[:window // 2], curve[-window // 2:]
    # curve = curve2

    my_x_axis = np.arange(1, len(C) + 1)
    plot_simple_curve(my_x_axis, curve, "Trans Curve With Noise")

    print("new_coef_matrix", new_coef_matrix)

    print("curve.shape", curve.shape)
    x = pytorch_optimizor(new_coef_matrix, curve)

    print("x.shape", x.shape)

    print("wave_len", wave_lens[0][selected_index])
    # plot_simple_curve(my_x_axis, x)

    plot_simple_multi_curves(wave_lens[0], [x, arr1], "Rebuild Curve. Blue new; Orange org")

    window = 8
    # np.convolve(x, np.ones(window) / window, mode='same')
    # avg_x = np.convolve(x, np.ones(window) / window, mode='same')
    # avg_x[:window // 2], avg_x[-window // 2:] = x[:window // 2], x[-window // 2:]
    # avg_x = savgol_filter(x, window_length=window, polyorder=3)
    avg_x = gaussian_filter1d(x, sigma=window)
    plot_simple_multi_curves(wave_lens[0], [avg_x, arr1], "Smoothed Curve")

    # # x = np.linalg.solve(coef_matrix, curve)
    # print("fwhm, cwl org", find_fwhm_normal(np.arange(713, 920 + 1, 3), curve))
    # fitted_x = np.linspace(690, 958, 68)
    # print("fwhm, cwl new", find_fwhm_normal(fitted_x, x[1:len(x) - 1]))
    #
    # draw_curves(fitted_x, [x[1:len(x) - 1]], "new curve")
    # draw_curves(np.arange(713, 920 + 1, 3), [curve], "origin curve")
    # # mix_curves_2(np.arange(713, 920 + 1, 3), curve, fitted_x, x[1:len(x) - 1])


if __name__ == "__main__":
    # rebuild_test()
    vis_500 = r"D:\Unispectral\python_code\computational_spectral_imaging\Coating_Simulation\2025122901\data"
    nir_700 = r"D:\Unispectral\python_code\computational_spectral_imaging\Coating_Simulation\2025122902\data"
    fpi_450 = r"D:\Unispectral\python_code\computational_spectral_imaging\Coating_Simulation\FPI-J03"
    new_uc500 = r"D:\Unispectral\doc\新UC500设计\data_new_uc500_sim"
    new_uc450 = r"D:\Unispectral\doc\202604171031-U450-MEMS-Metal-Coating-Ag-J08"

    # rebuild_test_one_curve(fpi_450)
    # lst_no_reg(fpi_450)

    # rebuild_test_one_curve_4_new_uc500(new_uc500)
    # lst_no_reg_new_uc500(new_uc500)
    # lst_with_aug_reg_new_uc500(fpi_450)
    # lst_with_aug_reg_new_uc500(new_uc450)
    lst_with_aug_reg(new_uc450)

    # rebuild_test_one_curve_from_file(r"E:\fp_cavity_sim_260105\08.npy")

