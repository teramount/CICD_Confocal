import copy
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
from scipy import ndimage
from src.confocal_analysis_codes.ellipsoid_fit_confocal import run_on_images
from scipy.ndimage import gaussian_filter
from tkinter import filedialog, simpledialog, messagebox, ttk
import tkinter as tk
from scipy.ndimage import label
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
# import il_calc
import src.confocal_analysis_codes.scan_analysis_tools as utils
from external.Theoretical_calculations.reflection_model import setup_visual

MID_STAGE_BUFFER = 10
TO_TARGET = 63.5
RECT_BUFFER = 15
EXTERNAL_FIX = 0
XPANDER_S = 2000#1700#2000
XPANDER_Y = 2700#2300#3000#2700
RES = 0.276
INIT_LOC = 30
STOP_HEIGHT = 17
DELTA = 1
START_FIT_INDEX = 35 #10
FIND_PIVOT_X_OFFSET = 200
FIND_PIVOT_Y_OFFSET = 600
FIBER_OFFSET = 0.175
PIVOT_PLANE_BUFFER = 0 * 2
LOW_SECTION_LENGTH = 13
TOP_TO_BOT_DIST = 75
BOT_LEN = 40
CHANNEL_PITCH = 450
CHANNEL_BUFFER = 80
SHORT_PIVOT_LEN = 15
SHORT_PIVOT_LEN_MIN_X = 0
SHORT_PIVOT_LEN_MAX_X = 0
SHORT_PIVOT_LEN_MIN_Y = 0
SHORT_PIVOT_LEN_MAX_Y = 0
ROI_START =1800#1950 #was 2050
ROI_END = 3400# 3400#3000
FIT_RAD = 180
XPANDER_BOX_LENGTH = 450
XPANDER_FALLBACK_LENGTH = 450
FIT_RADIUS_FOCAL = 165
FIT_RADIUS_RES = 219
XPANDER_FINDING_OFFSET = 300##75#50 #1000#p1 was 650 #350
XPANDER_FINDING_LIMIT =1100#900#850#750# 1550 # 1700# p1 was 1700#1500
TARGET_HEIGHT_PHD = 18.2
TARGET_ANGLE = 40.89
MAIN_CSV_PATH = r"G:\Shared drives\Design\Reports\Merged Plug and BoC csv data\Merged BoC data.csv"
STOOL_CSV_PATH = r"G:\Shared drives\Design\Reports\Merged Plug and BoC csv data\merged_boc_stool_data.csv"


COLUMN_LIST = ['component_id', 'pivot_bottom', 'pivot_top', 'PHD', 'pivot_axial_focal', 'pivot_trans_focal',
               'pivot_angle_SI_reference', 'pivot_angle_pivot_top_reference', 'axial_focal_length',
               'trans_focal_length', 'PXD_geo', 'PXTO_opt', 'TO_geo', 'XTO_opt', 'PXD_opt',
               'stool_height', 'xpander_mid_x', 'xpander_mid_y', 'pivot_RMSE', 'xpander_RMSE', 'variation_id',
               'column', 'row', 'channel', 'variation_comment', 'master', 'stamp', 'rep', 'master-stamp',
               'wafer_id', 'column-row', 'column-row-channel', 'scan_configuration', 'DUT_state',
               'dicing_vendor', 'xpander_height', 'PXTO_geo', 'XTO_geo', 'scan_index', 'leveling_angle_x',
               'leveling_angle_y', 'effective_channel_height', 'angle_from_phd', 'pivot_height', 'npy_path', 'PXD_opt_uncompensated',
               'effective_channel_height_uncompensated', 'pivot_angle_SI_reference_uncompensated', 'lower_cross_to_xpander_opt_center',
               'lower_cross_to_xpander_geo_center', 'upper_cross_to_xpander_opt_center',
               'upper_cross_to_xpander_geo_center', 'first_pivot_square_to_xpander_opt_center',
               'first_pivot_square_to_xpander_geo_center', 'second_pivot_square_to_xpander_opt_center',
               'second_pivot_square_to_xpander_geo_center', 'third_pivot_square_to_xpander_opt_center',
               'third_pivot_square_to_xpander_geo_center', 'fourth_pivot_square_to_xpander_opt_center',
               'fourth_pivot_square_to_xpander_geo_center', 'mean_pivot_height', 'pxd_geo_p_opt_x',
              'xpander_height_at_opt_center', 'xpander_height_at_geo_center']

STOOL_COLUMN_LIST = ['component_id', 'stool_height', 'mech_or_newton', 'master', 'stamp', 'rep', 'col', 'row', 'site',
                     'wafer_id', 'master_stamp']


def apply_leveling(raw_data, theta_x, theta_y):
    x = np.linspace(0, raw_data.shape[1] - 1, raw_data.shape[1]) * RES
    y = np.linspace(0, raw_data.shape[0] - 1, raw_data.shape[0]) * RES
    xx, yy = np.meshgrid(x, y)
    rotated_matrix = np.sin(theta_x) * xx + np.cos(theta_x) * raw_data
    rotated_matrix = np.sin(theta_y) * yy + np.cos(theta_y) * rotated_matrix
    return rotated_matrix


def rotate_around_y(matrix_x, matrix_y, matrix_z, angle):
    angle_rad = np.radians(-angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    coordinates_3d = np.array([matrix_x.flatten(), matrix_y.flatten(), matrix_z.flatten()])
    rotated_coordinates = np.dot(rotation_matrix, coordinates_3d)
    rotated_x, rotated_y, rotated_z = np.split(rotated_coordinates, 3)
    rotated_x = rotated_x.reshape(matrix_x.shape)
    rotated_y = rotated_y.reshape(matrix_y.shape)
    rotated_z = rotated_z.reshape(matrix_z.shape)
    return rotated_x, rotated_y, rotated_z


def apply_rotation_to_pivot(unrotated_pivot, resolution, angle):
    x_max = (unrotated_pivot.shape[1] - 1) * resolution
    y_max = (unrotated_pivot.shape[0] - 1) * resolution
    x = np.linspace(0, x_max, unrotated_pivot.shape[1])
    y = np.linspace(0, y_max, unrotated_pivot.shape[0])
    matrix_x, matrix_y = np.meshgrid(x, y)
    rotated_x, rotated_y, rotated_z = rotate_around_y(matrix_x, matrix_y, unrotated_pivot, angle)
    new_cols = int(rotated_z.shape[1] * 1.32)
    new_indices = np.linspace(0, rotated_z.shape[1] - 1, new_cols)
    interpolated_data = np.array(
        [np.interp(new_indices, np.arange(rotated_z.shape[1]), rotated_z[i, :]) for i in range(rotated_z.shape[0])])
    return interpolated_data


def extract_pivot_focal_length(optic_section, fit_radius, debug=False, file_name=''):
    try:
        pivot_focal, original, fit = run_on_images(optic_section, fit_radius, RES, file_name, '', '', di=False, save_residue=False)
        trans = pivot_focal['focal_length_x']
        axial = pivot_focal['focal_length_y']
        utils.print_success('Extracted Pivot Focal successfully')
        return axial, trans
    except:
        utils.print_error('Failed pivot focal')
        return np.nan, np.nan


def apply_mask_high_pass(raw_data, threshold):
    data_for_analysis = copy.deepcopy(raw_data)
    data_for_analysis[data_for_analysis < threshold] = np.nan
    return data_for_analysis


def apply_gaussian_blur(data, kernel_size=3, turn_to_grayscale=False):
    if turn_to_grayscale:
        normalized_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        data = (normalized_data * 255).astype(np.uint8)
    return cv2.GaussianBlur(data, (kernel_size, kernel_size), 0)


def apply_median_filter(data, kernel_size=3, turn_to_grayscale=False, debug=False):
    plot_for_debug = debug
    if turn_to_grayscale:
        normalized_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        data = (normalized_data * 255).astype(np.uint8)
    median_filtered = cv2.medianBlur(data, kernel_size)
    if plot_for_debug:
        plt.figure()
        plt.imshow(median_filtered)
        plt.show()
    return median_filtered


def increase_contrast(image, matrix_size, clip_limit, debug=False, blur=True, turn_to_grayscale=True):
    plot_for_debug = debug
    apply_blurring = blur
    if turn_to_grayscale:
        normalized_data = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
        grayscale_data = (normalized_data * 255).astype(np.uint8)
    else:
        grayscale_data = image
    if apply_blurring:
        grayscale_data = apply_gaussian_blur(grayscale_data, 5)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(matrix_size, matrix_size))
    cl1 = clahe.apply(grayscale_data)
    if plot_for_debug:
        plt.figure()
        plt.imshow(cl1)
        plt.show()
    return cl1


def apply_laplacian_enhancing(data, debug=False):
    plot_for_debug = debug
    laplacian_enhanced = cv2.Laplacian(data, cv2.CV_64F)
    laplacian_enhanced = cv2.convertScaleAbs(laplacian_enhanced)
    if plot_for_debug:
        plt.figure()
        plt.imshow(laplacian_enhanced)
        plt.show()
    return laplacian_enhanced


def apply_edge_detection(data, threshold_1, threshold_2, debug=False, blurr=True):
    plot_for_debug = debug
    apply_blurring = blurr
    if apply_blurring:
        data = cv2.GaussianBlur(data, (3, 3), sigmaX=0)
    edges = cv2.Canny(data, threshold_1, threshold_2)
    if plot_for_debug:
        plt.figure()
        plt.imshow(edges)
        plt.show()
    return edges


def apply_edge_smoothing_and_closing(edges, kernel_shape=2, debug=False, num_iter=1):
    plot_for_debug = debug
    kernel = np.ones((kernel_shape, kernel_shape), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=num_iter)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.erode(edges, kernel, iterations=num_iter)
    if plot_for_debug:
        plt.figure()
        plt.imshow(edges)
        plt.show()
    return edges


def rect_finder_by_contours(edges, min_width=150, max_width=260, min_height=150, max_height=185,
                            debug=False, square=False, epsilon_0=0.1):  # width was max 280, max_height
    plot_for_debug = debug
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        epsilon = epsilon_0 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if 4 <= len(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if square and abs(w - h) > 3:
                continue  # Skip if the difference between width and height exceeds 2
            if min_width <= w <= max_width and min_height <= h <= max_height:
                rectangles.append(approx)
    if plot_for_debug:
        output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for rect in rectangles:
            cv2.drawContours(output_image, [rect], -1, (0, 255, 0), 2)
        plt.figure()
        plt.imshow(output_image)
        plt.show()
    return rectangles


def extract_pivot_from_rect(pivot_surounding_rect, raw_data, x_offset, y_offset):
    external_fix = EXTERNAL_FIX
    for rect in pivot_surounding_rect:
        x, y, w, h = cv2.boundingRect(rect)
        only_pivot = raw_data[y_offset + y: y_offset + y + h + external_fix, x + x_offset:x + x_offset + w]
        pivot_coordinates = [y_offset + y, y_offset + y + h + external_fix, x + x_offset, x + x_offset + w]
        return pivot_coordinates, only_pivot


def find_pivot(raw_data, background_height, y_start, x_start, x_end, debug=False, atp=False):
    if atp:
        step_size = 2.5
    else:
        step_size = 1
    try:
        y_start += 30
        quantized_data = quantize_data(raw_data[y_start:, x_start:x_end], step_size=step_size, debug=debug, min_group_size=1500)
        for num in set(np.unique(quantized_data)):
            masked_matrix = np.where(quantized_data == num, num, 0)
            normalized_data = (masked_matrix * 255).astype(np.uint8)
            edges = apply_edge_detection(normalized_data, 70, 140, debug=debug, blurr=False)
            smoothed_and_closed = apply_edge_smoothing_and_closing(edges)
            if atp:
                pivot_surounding_rect = rect_finder_by_contours(smoothed_and_closed, debug=debug,
                                                                min_width=160, max_width=280,
                                                                min_height=110, max_height=125)
            else:
                pivot_surounding_rect = rect_finder_by_contours(smoothed_and_closed, debug=debug)
            if len(pivot_surounding_rect) > 0:
                pivot_coordinates, pivot_only = extract_pivot_from_rect(pivot_surounding_rect, raw_data, x_start,
                                                                        y_start)
                if debug:
                    plt.figure()
                    plt.imshow(pivot_only)
                    plt.show()
                utils.print_success('Pivot found')
                return pivot_coordinates, pivot_only
    except:
        utils.print_error('Failed finding pivot')
        return np.nan, np.nan


def apply_basic_filter(pivot_section, lower_bound, upper_bound, value_to_use=np.nan):
    rough_filter = np.where((pivot_section >= lower_bound) & (pivot_section <= upper_bound), pivot_section,
                            value_to_use)
    return rough_filter


def get_plane_avg_height(pivot_plane, buffer=5, top=False, debug=False):
    if top:
        buffered_pivot = pivot_plane[buffer:-buffer, buffer:-buffer]
    else:
        buffered_pivot = pivot_plane[buffer:-buffer, buffer:-buffer]
    rough_avg = np.nanmean(buffered_pivot)
    rough_std = np.nanstd(buffered_pivot)
    rough_filter = apply_basic_filter(pivot_plane, rough_avg - 1 * rough_std, rough_avg + 1 * rough_std)
    if debug:
        plt.figure()
        plt.imshow(rough_filter)
        plt.show()
    pivot_plane_height = np.nanmean(rough_filter)
    return pivot_plane_height


def slice_dimensions(slc):
    return (slc[0].stop - slc[0].start, slc[1].stop - slc[1].start)


def filter_cross_section(non_filtered_cross_section):
    basic_mask = np.zeros((non_filtered_cross_section.shape[0], non_filtered_cross_section.shape[1]))
    mid = non_filtered_cross_section.shape[1] // 2
    binary_mask = np.where(np.isnan(non_filtered_cross_section[:, mid - 15:mid + 15]), 0, 1)
    basic_mask[:, mid - 15:mid + 15] = binary_mask
    data_labels, num_labels = ndimage.label(basic_mask)
    slices = ndimage.find_objects(data_labels)
    longest = 0
    correct_label = 0
    for lbl, slc in enumerate(slices, start=1):
        height, width = slice_dimensions(slc)
        if height > longest:
            longest = height
            correct_label = lbl
    mask = np.where(data_labels == correct_label, 1, 0).astype(np.uint8)
    main_area = non_filtered_cross_section * mask
    main_area = np.nan_to_num(main_area, nan=0)
    return main_area


def get_cross_center(pivot_section, element_height, debug=False):
    rough_filter = apply_basic_filter(pivot_section, element_height - 2, element_height - 0.5)
    cross_section = filter_cross_section(rough_filter)
    cm = ndimage.center_of_mass(cross_section)
    cm_y = cm[0]
    cm_x = cm[1]
    if debug:
        plt.figure()
        plt.imshow(rough_filter)
        plt.figure()
        plt.imshow(cross_section)
        plt.figure()
        plt.imshow(pivot_section)
        plt.scatter(cm_x, cm_y)
        plt.show()
    return cm_x, cm_y


def clear_small_groups(raw_mask):
    labeled_regions, num_features = label(raw_mask)
    region_sizes = np.bincount(labeled_regions.ravel())
    largest_region_label = np.argmax(region_sizes[1:]) + 1
    indices = np.where(labeled_regions == largest_region_label)
    min_row, max_row = indices[0].min(), indices[0].max()
    min_col, max_col = indices[1].min(), indices[1].max()
    return (min_row, max_row, min_col, max_col), (labeled_regions == largest_region_label)


def quantize_data(section, debug=False, step_size=1, min_group_size = 500):
    min_val, max_val = np.nanmin(section), np.nanmax(section)
    bins = np.arange(min_val, max_val + step_size, step_size)
    quantized_matrix = np.digitize(section, bins) - 1
    hist, bin_edges = np.histogram(quantized_matrix, bins=np.arange(quantized_matrix.max() + 2))
    small_values = np.where(hist < min_group_size)[0]
    processed_matrix = quantized_matrix.astype(float)
    processed_matrix[np.isin(quantized_matrix, small_values)] = np.nan
    if debug:
        plt.figure()
        plt.imshow(processed_matrix)
        plt.show()
    return processed_matrix


def apply_sigma_band_pass(section, num_sigma=2):
    mean = np.mean(section)
    std = np.std(section)
    mask = np.abs(section - mean) > num_sigma * std
    section = section.astype(float)  # Ensure it's a float array to support NaNs
    section[mask] = np.nan
    return section


def bounding_rect_of_non_nan(matrix):
    """Finds the bounding rectangle (min_row, max_row, min_col, max_col) of non-NaN values in a matrix."""
    non_nan_indices = np.argwhere(~np.isnan(matrix))
    if non_nan_indices.size == 0:
        return None  # No non-NaN values found
    min_row, min_col = non_nan_indices.min(axis=0)
    max_row, max_col = non_nan_indices.max(axis=0)
    return min_row, max_row, min_col, max_col


def get_high_section(quantized_pivot, pivot_only, debug=False):
    unique_values, counts = np.unique(quantized_pivot[~np.isnan(quantized_pivot)], return_counts=True)
    if len(unique_values) < 2:
        threshold = unique_values[0]  # If only one unique value, use it
    else:
        largest_value = unique_values[-1]
        second_largest_value = unique_values[-2]
        largest_count = counts[-1]
        second_largest_count = counts[-2]
        threshold = second_largest_value if second_largest_count > largest_count else largest_value
    top_section = quantized_pivot > threshold - 0.1
    top_mask, binary_mask = clear_small_groups(top_section)
    masked_original = np.where(binary_mask, pivot_only, np.nan)
    high_section = masked_original[top_mask[0] + PIVOT_PLANE_BUFFER:top_mask[1] - PIVOT_PLANE_BUFFER,
                   top_mask[2] + PIVOT_PLANE_BUFFER:top_mask[3] - PIVOT_PLANE_BUFFER]
    high_section = apply_mask_high_pass(high_section, np.nanmean(high_section) - 0.15)
    top_mask_new = bounding_rect_of_non_nan(high_section)
    final_mask = (top_mask[0] + top_mask_new[0], top_mask[0] + top_mask_new[1], top_mask[2] + top_mask_new[2],
                  top_mask[2] + top_mask_new[3])
    if debug:
        plt.figure()
        plt.imshow(high_section)
        plt.show()
    return high_section, np.nanmean(apply_sigma_band_pass(high_section)), final_mask


def get_low_section(pivot_only, top_mask, debug=False, just_section=False, atp=False):
    if atp:
        dist = SHORT_PIVOT_LEN
    else:
        dist = BOT_LEN
    y_coord = 0  # min_index[0] + 5
    bot_start = top_mask[0] - TOP_TO_BOT_DIST
    if bot_start - dist < 0:
        start_point = 0
    else:
        start_point = bot_start - dist
    bot_section = pivot_only[start_point:bot_start, top_mask[2]:top_mask[3]]
    if just_section:
        return bot_section, y_coord - 5  # y_coord - 10
    filtered_bot = apply_mask_high_pass(bot_section, np.nanmean(bot_section) - 0.2)
    bot_sec = apply_sigma_band_pass(filtered_bot)
    if debug:
        plt.figure()
        plt.imshow(bot_sec)
        plt.show()
    return bot_sec, np.nanmean(bot_sec)


def find_pivot_geo_center(pivot_only, top_mask, high_section_height, low_section_height, debug=False, atp=False):
    if atp:
        dist = SHORT_PIVOT_LEN
    else:
        dist = BOT_LEN
    high_section = pivot_only[top_mask[0] + PIVOT_PLANE_BUFFER:top_mask[1] - PIVOT_PLANE_BUFFER,
                   top_mask[2] + PIVOT_PLANE_BUFFER:top_mask[3] - PIVOT_PLANE_BUFFER]
    low_section, bot_start = get_low_section(pivot_only, top_mask,
                                             just_section=True, atp=atp)  # pivot_only[bot_start-BOT_LEN:bot_start, top_mask[2]:top_mask[3]]
    bot_start = top_mask[0] - TOP_TO_BOT_DIST
    upper_cross_mid_x, upper_cross_mid_y = get_cross_center(high_section, high_section_height, debug=debug)
    lower_cross_mid_x, lower_cross_mid_y = get_cross_center(low_section, low_section_height, debug=debug)
    if debug:
        plt.figure()
        plt.imshow(high_section)
        plt.figure()
        plt.imshow(low_section)
        plt.figure()
        plt.imshow(pivot_only)
        plt.scatter(PIVOT_PLANE_BUFFER + top_mask[2] + upper_cross_mid_x,
                    PIVOT_PLANE_BUFFER + top_mask[0] + upper_cross_mid_y)
        plt.scatter(top_mask[2] + lower_cross_mid_x, lower_cross_mid_y+ bot_start - dist)#+ bot_start - dist)
        plt.show()
    mid_y = (lower_cross_mid_y + bot_start - dist + PIVOT_PLANE_BUFFER + top_mask[0] + upper_cross_mid_y) / 2
    mid_x = (PIVOT_PLANE_BUFFER + top_mask[2] + upper_cross_mid_x + top_mask[2] + lower_cross_mid_x) / 2
    upper_cross_y = PIVOT_PLANE_BUFFER + top_mask[0] + upper_cross_mid_y
    lower_cross_y = lower_cross_mid_y + bot_start - dist
    return mid_x, mid_y, upper_cross_y, lower_cross_y


def extract_pivot_height_and_mid_point_params(pivot_only, pivot_coordinates, debug=False, atp=False, stool_offset=0.0, etched=False):
    try:
        if etched:
            offset = -(0.175 + stool_offset)
        else:
            offset = -(0.275 + stool_offset)
        quantized_pivot = quantize_data(pivot_only, debug=debug, step_size=2)
        pivot_high_plane, high_section_height, top_mask = get_high_section(quantized_pivot, pivot_only, debug=debug)
        pivot_low_plane, low_section_height = get_low_section(pivot_only, top_mask, debug=debug, atp=atp)
        geo_mid_x_alternative, geo_mid_y, upper_cross_y, lower_cross_y = find_pivot_geo_center(pivot_only, top_mask, high_section_height, low_section_height,
                                                     debug=debug, atp=atp)
        try:
            geo_mid_x, left_side_squares, right_side_squares = find_pivot_rects_mid_point(pivot_only, debug=debug)
        except:
            geo_mid_x = geo_mid_x_alternative
        # sloped_pivot_section = pivot_only[:top_mask[1], top_mask[2]:top_mask[3]]
        try:
            suspected_sloped_section = pivot_only[top_mask[0]:top_mask[1], top_mask[2] - 5:top_mask[3] + 5]
            pivot_only_high_cont = increase_contrast(suspected_sloped_section, 5, 15, debug=debug)
            pivot_only_edges = apply_edge_detection(pivot_only_high_cont, 25, 50, debug=debug)
            detected_x_lines = apply_hough_lines(pivot_only_edges, 1, np.pi / 360, 15, 0.25, debug=debug)
            pair_x, _ = closest_number_pair(detected_x_lines, 125, 10)
            pair_x = sorted(pair_x[1])
            sloped_pivot_section = pivot_only[:top_mask[1], top_mask[2]-5 + pair_x[0]:top_mask[2]-5 + pair_x[1]]
        except:
            sloped_pivot_section = pivot_only[:top_mask[1], top_mask[2] + 5:top_mask[3] - 5]
        # plt.figure()
        # plt.imshow(pivot_only)
        # plt.figure()
        # plt.imshow(sloped_pivot_section)
        # plt.show()
        pivot_cross_section = sloped_pivot_section[:, int(geo_mid_x-top_mask[2])]
        index_closest_height = np.argmin(np.abs(pivot_cross_section - offset))
        index_closest_height_uncompensated = np.argmin(np.abs(pivot_cross_section - offset + stool_offset))
        phd = high_section_height - low_section_height
        angle_from_phd = np.degrees(np.arctan(phd/(TARGET_HEIGHT_PHD/np.tan(np.radians(TARGET_ANGLE)))))
        pivot_optical_mid = index_closest_height + pivot_coordinates[0]
        pivot_optical_mid_uncompensated = index_closest_height_uncompensated + pivot_coordinates[0]
        trans_mid = pivot_coordinates[2] + geo_mid_x
        axial_mid = pivot_coordinates[0] + geo_mid_y
        real_upper_cross_y_coor = upper_cross_y + pivot_coordinates[0]
        real_lower_cross_y_coor = lower_cross_y + pivot_coordinates[0]
        mid_x_point = pivot_only.shape[1] // 2
        buffer = mid_x_point + RECT_BUFFER
        try:
            left_side_squares_real = [[a + pivot_coordinates[2], b + pivot_coordinates[0]] for (a, b) in left_side_squares]
            right_side_squares_real = [[a + pivot_coordinates[2] + buffer, b + pivot_coordinates[0]] for (a, b) in right_side_squares]
        except:
            left_side_squares_real, right_side_squares_real = np.nan, np.nan
        try:
            pivot_shape_rmse = fit_elliptical_paraboloid(pivot_only[index_closest_height - 20:index_closest_height + 20, int(geo_mid_x)-30:int(geo_mid_x)+30], debug=False)
        except:
            print('FAIL PIVOT FULL RMSE')
        if debug:
            plt.figure()
            plt.imshow(pivot_only)
            plt.scatter(geo_mid_x, index_closest_height)
            plt.scatter(geo_mid_x, geo_mid_y)
            plt.show()
        utils.print_success('Extracted Pivot parameters successfully')
        return (high_section_height, low_section_height, phd, trans_mid, axial_mid,
                pivot_optical_mid, angle_from_phd, pivot_optical_mid_uncompensated, sloped_pivot_section,
                real_upper_cross_y_coor, real_lower_cross_y_coor, left_side_squares_real, right_side_squares_real)
    except:
        utils.print_error('Failed extracting pivot params')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def generate_fit(dz_list, debug=False):
    height_vec = np.arange(0, RES * len(dz_list), RES)
    fit_params = np.polyfit(height_vec[START_FIT_INDEX:], dz_list[START_FIT_INDEX:], 2)
    fit_data = (fit_params[0] * height_vec[START_FIT_INDEX:] ** 2 + fit_params[1] * height_vec[START_FIT_INDEX:]
                + fit_params[2])
    residue = dz_list[START_FIT_INDEX:] - fit_data
    dz_array = np.array(dz_list[START_FIT_INDEX:])

    # plt.figure(123456)
    # plt.plot(-np.array(dz_list[START_FIT_INDEX:-3]), label='Pivot shape')
    # plt.plot(-(dz_array - np.nanmin(dz_array)), label='Pivot fit')
    if debug:
        plt.figure(9999999)
        plt.plot(residue)
        plt.title('Lext 5100 Pivot residual')
        plt.xlabel('pixel')
        plt.ylabel('magnitude [um]')
        plt.figure()
        plt.title('Pivot shape and fit')
        plt.xlabel('pixel')
        plt.ylabel('magnitude [um]')
        plt.plot(-np.array(dz_list[START_FIT_INDEX:-3]), label='Pivot shape')
        plt.plot(-fit_data, label='Pivot fit')
        plt.show()
    return height_vec, fit_params, fit_data, residue


def compute_average_center(centers):
    if not centers:
        return None  # Handle case with no rectangles
    avg_x = np.mean([c[0] for c in centers])
    avg_y = np.mean([c[1] for c in centers])
    return avg_x, avg_y


def find_rects_from_section(section, debug=False):
    quantized_section = quantize_data(section, step_size=5, debug=debug)
    min_val = quantized_section.min()
    mask = quantized_section == min_val
    filtered_raw_matrix = np.where(mask, section, np.nan)
    # plt.figure()
    # plt.imshow(filtered_raw_matrix)
    # plt.show()
    higher_contrast = increase_contrast(filtered_raw_matrix, 5, 3, debug)
    edges = apply_edge_detection(higher_contrast, 50, 100, debug)
    left_rects = rect_finder_by_contours(edges, 14, 25, 14, 25, debug)
    centers = find_rect_center(left_rects)
    return centers


def find_pivot_rects_mid_point(pivot_only, rotated=False, debug=False):
    if rotated:
        mid_y_point = pivot_only.shape[0] // 2
        left_side = pivot_only[:mid_y_point-RECT_BUFFER, :]
        right_side = pivot_only[mid_y_point+RECT_BUFFER:, :]
        buffer = mid_y_point+RECT_BUFFER
    else:
        mid_x_point = pivot_only.shape[1] // 2
        left_side = pivot_only[:, :mid_x_point-RECT_BUFFER,]
        right_side = pivot_only[:, mid_x_point+RECT_BUFFER:]
        buffer = mid_x_point + RECT_BUFFER
    left_rect_center = find_rects_from_section(left_side, debug=debug)
    right_rects_center = find_rects_from_section(right_side, debug=debug)
    if rotated:
        _, left_avg_center = compute_average_center(left_rect_center)
        _, right_avg_center = compute_average_center(right_rects_center)
    else:
        left_avg_center, _ = compute_average_center(left_rect_center)
        right_avg_center, _ = compute_average_center(right_rects_center)
    return (left_avg_center + right_avg_center + buffer) // 2, left_rect_center, right_rects_center


def get_pivot_cross_section_values(pivot_only, pivot_init_height):
    num_of_iter = 0
    rotated_pivot = np.rot90(pivot_only, 1)
    try:
        mid_y_point = int(find_pivot_rects_mid_point(rotated_pivot, True))
    except:
        mid_y_point = rotated_pivot.shape[0] // 2
    window_size = 2
    epsilon = 0.05
    # mid_y_point = rotated_pivot.shape[0] // 2
    loc = INIT_LOC
    still_searching = True
    dz_list_above_si, dz_list_above_pt = [], []
    while still_searching and num_of_iter < 1000:
        avg_height = np.nanmean(rotated_pivot[mid_y_point - window_size:mid_y_point + window_size, loc:loc + DELTA])
        dz_list_above_pt.append(avg_height - pivot_init_height)
        dz_list_above_si.append(avg_height)
        if (STOP_HEIGHT - (avg_height - pivot_init_height)) < epsilon:
            still_searching = False
        else:
            loc += 1
        num_of_iter += 1
    return dz_list_above_si, dz_list_above_pt


def find_pivot_target_values(height_and_angle_data, target_height, debug=False, etch=True):
    interp_function = interp1d(height_and_angle_data['Pivot height'], height_and_angle_data['Angle'],
                               kind='linear', fill_value="extrapolate")
    over_sampled_pivot_height = np.linspace(min(height_and_angle_data['Pivot height']),
                                            max(height_and_angle_data['Pivot height']), 5000)
    over_sampled_angle = interp_function(over_sampled_pivot_height)
    if debug:
        plt.figure()
        plt.title('Pivot angle as a function of height')
        plt.plot(over_sampled_pivot_height, over_sampled_angle, label='Pivot angle vs height')
        if etch:
            plt.axvline(x=0, color='r', linestyle='--', label='zero pure')
        else:
            plt.axvline(x=0, color='b', linestyle='--', label='zero gold')
            plt.axvline(x=-0.1, color='r', linestyle='--', label='zero pure')
        plt.axvline(x=target_height, color='g', linestyle='--', label='WG')
        plt.xlabel('Pivot height')
        plt.ylabel('Pivot angle')
        plt.legend()
        plt.show()
    index_closest_height = np.argmin(np.abs(over_sampled_pivot_height - target_height))
    pivot_angle_at_target_height = over_sampled_angle[index_closest_height]
    return pivot_angle_at_target_height


def calc_angle_using_fit(fitted_data, target_height, debug=False, etch=True):
    """

    :param fitted_data:
    :param target_height:
    :return:
    """
    fit_angle_list = []
    for i in range(1, len(fitted_data)):
        angle = np.rad2deg(np.arctan((fitted_data[i] - fitted_data[i - 1]) / RES))
        fit_angle_list.append(angle)
    height_and_angle_df = pd.DataFrame({'Pivot height': fitted_data[:-1], 'Angle': fit_angle_list})
    angle_at_target_height = find_pivot_target_values(height_and_angle_df, target_height, debug=debug, etch=etch)
    return angle_at_target_height


def extract_pivot_angle(pivot_only, pivot_init_height,etched, stool_height=0, debug=False, artificial_offset=0):
    """
    calculate the pivot angle
    :param pivot_only: isolated pivot
    :param pivot_init_height: pivot top height
    :param stool_height: relevant stool height - when given will calculate the angle with reference to the stool height
    :return: angel at 23.41, angle at channel height - when stool height is known
    """
    # plt.imshow(pivot_only)
    # plt.show()
    if etched:
        offset = -(0.175+artificial_offset)
    else:
        offset = -(0.275+artificial_offset)
    try:
        z_list_si_ref, z_list_pt_ref = get_pivot_cross_section_values(pivot_only, pivot_init_height)
        height_vec_si, fitted_params_si, fitted_data_si, residue_si = generate_fit(z_list_si_ref, debug=debug)
        angle_at_fiber = calc_angle_using_fit(fitted_data_si, offset, debug=debug, etch=etched) ##### HERE #####
        if stool_height > 0:
            relevant_channel_height = stool_height - pivot_init_height - FIBER_OFFSET
            height_vec_pt, fitted_params_pt, fitted_data_pt, residue_pt = generate_fit(z_list_pt_ref)
            angle_at_channel_height = calc_angle_using_fit(fitted_data_pt, relevant_channel_height, etch=etched)
        else:
            angle_at_channel_height = np.nan
        utils.print_success('Extracted Pivot angle successfully')
        return angle_at_fiber, angle_at_channel_height, np.sqrt(np.mean(residue_si ** 2))
    except:
        utils.print_error('Failed extracting pivot angle')
        return np.nan, np.nan, np.nan


def most_frequent_nonzero_nonmax2(matrix, min_count=1000):
    """
    Returns a boolean mask for the most frequent value in the matrix,
    excluding 0, NaNs, and the two highest unique values. Only considers values
    that appear at least `min_count` times.
    """
    valid = matrix[~np.isnan(matrix)]
    unique, counts = np.unique(valid, return_counts=True)

    # Exclude 0 and the two highest unique values
    nonzero_unique = unique[unique != 0]
    if len(nonzero_unique) < 3:
        target_val = np.nanmin(nonzero_unique)
        return matrix == target_val
        # raise ValueError("Not enough distinct non-zero values to exclude the two highest.")

    top_two_vals = nonzero_unique[-2:]  # Assumes np.unique returns sorted values
    filtered = [(val, count) for val, count in zip(unique, counts)
                if val != 0 and val not in top_two_vals and count > min_count]

    if not filtered:
        raise ValueError("No valid values meet the criteria.")

    # Choose the most frequent value among filtered
    target_val = max(filtered, key=lambda x: x[1])[0]

    return matrix == target_val

def closest_number_pair(numbers, target, tolerance=15):
    """
    Function that returns a pari of coordinates
    """
    closest_pair = None
    min_diff = float('inf')
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            dist = abs(numbers[i] - numbers[j])
            diff = abs(dist - target)
            if diff <= tolerance and diff < min_diff:
                min_diff = diff
                closest_pair = ((i, j), (numbers[i], numbers[j]))
    return closest_pair, min_diff if closest_pair else (None, None)


def find_xpander(leveled_raw_data, debug=False, step_size=3, atp=False):
    """
    Function that isolates the Xpander - used in the initial labeling process
    """
    x_base_offset = 15
    if atp:
        y_threshold = 225
        correction = 0
        y_offset = XPANDER_FINDING_OFFSET
        y_limit = XPANDER_FINDING_LIMIT
    else:
        y_threshold = 175
        correction = 0
        y_offset = XPANDER_FINDING_OFFSET + 310
        y_limit = XPANDER_FINDING_LIMIT + 150

    i=1
    try:
        temp_data = quantize_data(leveled_raw_data[y_offset:y_limit, x_base_offset:-x_base_offset], debug=debug, step_size=step_size, min_group_size=60000)
        temp_data[temp_data == 0] = np.nan
        # plt.figure()
        # plt.imshow(temp_data)
        # plt.show()
        f = most_frequent_nonzero_nonmax2(temp_data)
        masked_only = np.where(f, leveled_raw_data[y_offset:y_limit, x_base_offset:-x_base_offset], np.nan)
        # plt.imshow(masked_only)
        # plt.show()
        high_contrast_image = increase_contrast(masked_only,
                                                35, 15, debug=debug)  # 1600
        # high_contrast_image = increase_contrast(leveled_raw_data[XPANDER_FINDING_OFFSET:XPANDER_FINDING_LIMIT, :],
        #                                         35, 25, debug=debug)  # 1600
        edges_data = apply_edge_detection(high_contrast_image, 60, 120, debug=debug)
        laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
        x_lines = sorted(apply_hough_lines(laplace_filter, 1, np.pi / 360, 250, 0.25,
                                           debug=debug))
        while len(x_lines) < 2:
            edges_data = apply_edge_detection(high_contrast_image, 120-i, 240-(2*i), debug=debug)
            laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
            x_lines = sorted(apply_hough_lines(laplace_filter, 1, np.pi / 360, 250-i, 0.25,
                                               debug=debug))
            i += 1
        y_lines = sorted(apply_hough_lines(laplace_filter[:, :], 1, np.pi / 360, y_threshold,
                                           0.25, debug=debug, y_axis=True))
        while len(y_lines) < 2:
            edges_data = apply_edge_detection(high_contrast_image, 120-i, 240-(2*i), debug=debug)
            laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
            y_lines = sorted(apply_hough_lines(laplace_filter[:, :], 1, np.pi / 360, y_threshold-i,
                                               0.25, debug=debug, y_axis=True))
            i += 5
        pair_x, _ = closest_number_pair(x_lines, XPANDER_BOX_LENGTH)
        x_start, x_end = x_base_offset + pair_x[1][0], x_base_offset+ pair_x[1][1]
        pair_y, _ = closest_number_pair(y_lines, XPANDER_BOX_LENGTH-correction)
        if pair_y is None:
            if y_lines[0] > laplace_filter.shape[1] // 2:
                y_start, y_end = y_offset + y_lines[0]-XPANDER_FALLBACK_LENGTH, y_offset + y_lines[0]
            else:
                y_start, y_end = y_offset + y_lines[0], y_offset + y_lines[0]+XPANDER_FALLBACK_LENGTH
        else:
            y_start, y_end = y_offset + pair_y[1][0], y_offset + pair_y[1][1]
        if debug:
            plt.figure()
            plt.imshow(leveled_raw_data[y_start:y_end, x_start:x_end])
            plt.show()
        utils.print_success('found xpander')
        return leveled_raw_data[y_start:y_end, x_start:x_end], [y_start, y_end, x_start, x_end]
    except:
        utils.print_error('Failed finding the xpander')
        return np.nan, np.nan


def extract_xpander_focal_length_and_mid_point(xpander_only, xpander_coordinates, file, outpath, si_part, fit_radius,
                                               debug=False):
    """
    calculate xpander focal length and middle point
    :param xpander_only: xpander data as a matrix
    :param xpander_coordinates: xpander coordinates
    :param fit_radius: radius to be used for fitting (in pixels)
    :param debug: when true will plot figures for debugging and troubleshooting
    :return: xpander middle point (x and y) focal length (axial and trans)
    """
    try:
        result, original, fit = run_on_images(xpander_only, FIT_RADIUS_FOCAL, RES, file, outpath, '', True,
                               False, False, symmetric=False, plug=False, save_residue=False)
        # try:
        #     run_on_images(xpander_only, FIT_RADIUS_RES, RES, file, outpath, '', True,
        #                                           False, False, symmetric=True, plug=False, save_residue=True)
        # except:
        #     pass
        xpander_mid_x_coordinate = xpander_coordinates[2] + result['center_x']
        xpander_mid_y_coordinate = xpander_coordinates[0] + result['center_y']
        trans_focal = result['focal_length_x']
        axial_focal = result['focal_length_y']
        rmse = result['RMSE']
        mid_y_geo = xpander_coordinates[0] + xpander_only.shape[0] // 2
        apex_height_at_opt_center, apex_height_geo_center = utils.get_xpander_apex(xpander_only, (int(result['center_y']), int(result['center_x'])), si_part)
        if debug:
            plt.figure()
            plt.imshow(xpander_only)
            plt.scatter(result['center_x'], result['center_y'])
            plt.show()
        utils.print_success('Extracted Xpander Focal successfully')
        return (xpander_mid_x_coordinate, xpander_mid_y_coordinate, axial_focal, trans_focal, rmse, mid_y_geo,
                apex_height_at_opt_center, apex_height_geo_center)
    except:
        utils.print_error('Failed Xpander analysis')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def extract_channel_parameters(xpander_x, xpander_y, pivot_x, pivot_y, alignment_mid, pivot_optical_mid,
                               uncompensated_optical_mid, upper_cross_y, lower_cross_y, pivot_squares_avg,
                               xpander_geo_axial_mid):
    """
    calculate cahnnel related parameters
    :param xpander_x: xpander mid x coordinate
    :param xpander_y: xpander mid y coordinate
    :param pivot_x: pivot mid x coordinate
    :param pivot_y: pivot mid y coordinate
    :param alignment_mid: alignment marks middle point
    :return: pivot xpander distance, pivot xpander trans offset, trans offset
    """
    (first_pivot_square_to_xpander_opt_center, first_pivot_square_to_xpander_geo_center,
     second_pivot_square_to_xpander_opt_center, second_pivot_square_to_xpander_geo_center,
     third_pivot_square_to_xpander_opt_center, third_pivot_square_to_xpander_geo_center,
     fourth_pivot_square_to_xpander_opt_center, fourth_pivot_square_to_xpander_geo_center) = (np.nan, np.nan, np.nan,
                                                                                              np.nan, np.nan, np.nan,
                                                                                              np.nan, np.nan)
    try:
        pxd_optical_center = (pivot_optical_mid - xpander_y) * RES
        pxd_optical_center_uncompensated = (uncompensated_optical_mid - xpander_y) * RES
        pxd = (pivot_y - xpander_geo_axial_mid) * RES
        pxto = (pivot_x - xpander_x) * RES
        to = (pivot_x - alignment_mid) * RES
        xto = (xpander_x - alignment_mid) * RES
        upper_cross_to_xpander_opt = (upper_cross_y - xpander_y) * RES
        upper_cross_to_xpander_geo = (upper_cross_y - xpander_geo_axial_mid) * RES
        lower_cross_to_xpander_opt = (lower_cross_y - xpander_y) * RES
        lower_cross_to_xpander_geo = (lower_cross_y - xpander_geo_axial_mid) * RES
        pxd_geo_pivot_opt_xpander = (pivot_y - xpander_y) * RES
        for i in range(len(pivot_squares_avg)):
            if i == 0:
                first_pivot_square_to_xpander_opt_center = (pivot_squares_avg[0][1] - xpander_y) * RES
                first_pivot_square_to_xpander_geo_center = (pivot_squares_avg[0][1] - xpander_geo_axial_mid) * RES
            elif i == 1:
                second_pivot_square_to_xpander_opt_center = (pivot_squares_avg[1][1] - xpander_y) * RES
                second_pivot_square_to_xpander_geo_center = (pivot_squares_avg[1][1] - xpander_geo_axial_mid) * RES
            elif i == 2:
                third_pivot_square_to_xpander_opt_center = (pivot_squares_avg[2][1] - xpander_y) * RES
                third_pivot_square_to_xpander_geo_center = (pivot_squares_avg[2][1] - xpander_geo_axial_mid) * RES
            elif i == 3:
                fourth_pivot_square_to_xpander_opt_center = (pivot_squares_avg[3][1] - xpander_y) * RES
                fourth_pivot_square_to_xpander_geo_center = (pivot_squares_avg[3][1] - xpander_geo_axial_mid) * RES
        # first_pivot_square_to_xpander_opt_center = (pivot_squares_avg[0][1] - xpander_y) * RES
        # first_pivot_square_to_xpander_geo_center = (pivot_squares_avg[0][1] - xpadner_geo_axial_mid) * RES
        # second_pivot_square_to_xpander_opt_center = (pivot_squares_avg[1][1] - xpander_y) * RES
        # second_pivot_square_to_xpander_geo_center = (pivot_squares_avg[1][1] - xpadner_geo_axial_mid) * RES
        # third_pivot_square_to_xpander_opt_center = (pivot_squares_avg[2][1] - xpander_y) * RES
        # third_pivot_square_to_xpander_geo_center = (pivot_squares_avg[2][1] - xpadner_geo_axial_mid) * RES
        # fourth_pivot_square_to_xpander_opt_center = (pivot_squares_avg[3][1] - xpander_y) * RES
        # fourth_pivot_square_to_xpander_geo_center = (pivot_squares_avg[3][1] - xpadner_geo_axial_mid) * RES
        utils.print_success('Extracted channel parameters successfully')
        return (pxd, pxto, to, xto, pxd_optical_center, pxd_optical_center_uncompensated, upper_cross_to_xpander_opt,
                upper_cross_to_xpander_geo, lower_cross_to_xpander_opt, lower_cross_to_xpander_geo,
                first_pivot_square_to_xpander_opt_center, first_pivot_square_to_xpander_geo_center,
                second_pivot_square_to_xpander_opt_center, second_pivot_square_to_xpander_geo_center,
                third_pivot_square_to_xpander_opt_center, third_pivot_square_to_xpander_geo_center,
                fourth_pivot_square_to_xpander_opt_center, fourth_pivot_square_to_xpander_geo_center,
                pxd_geo_pivot_opt_xpander)
    except:
        utils.print_error('Failed channel analysis')
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def apply_mask_low_pass(raw_data, threshold, value_to_transform=np.nan):
    """
    filter all data above a given threshold
    :param value_to_transform:
    :param raw_data: numpy matrix to filter
    :param threshold: cutoff value
    :return: filtered matrix
    """
    data_for_analysis = copy.deepcopy(raw_data)
    data_for_analysis[data_for_analysis > threshold] = value_to_transform
    return data_for_analysis


def find_rect_center(rect_list, offset_x=0, offset_y=0):
    """
    find the center of all the rectangles in the input list
    :param rect_list: coordinates list of the rectangles
    :return: list with rectangle centers
    """
    centers = []
    try:
        for rect in rect_list:
            x, y, w, h = cv2.boundingRect(rect)  # Get bounding box for the rectangle
            center_x = x + w / 2 + offset_x  # Calculate the x-coordinate of the center
            center_y = y + h / 2 + offset_y # Calculate the y-coordinate of the center
            centers.append((center_x, center_y))  # Store the center coordinates
        return centers
    except:
        return False


def find_stool(leveled_stool_section, background_height, debug):
    try:
        if background_height == 0:
            background_height = np.nanmean(leveled_stool_section[:20, 400:440])
        normalized_data = leveled_stool_section - background_height
        no_background = apply_mask_high_pass(normalized_data, np.nanmean(normalized_data) + 15)  # was 11
        blurred_image = apply_gaussian_blur(no_background)
        high_contrast = increase_contrast(blurred_image, 35, 7, debug=debug)  # clip was 7, matrix was 50
        edges = apply_edge_detection(high_contrast, 30, 60, debug=debug)  # was 40 and 80
        smoothed_and_closed_edges = apply_edge_smoothing_and_closing(edges, debug=debug)
        stool_only_rect = rect_finder_by_contours(smoothed_and_closed_edges, 200, 300, 200, 300, debug)
        stool_coordinates, stool_only = extract_pivot_from_rect(stool_only_rect, normalized_data, 0, 0)
        stool_only = apply_mask_high_pass(stool_only, np.nanmean(stool_only) - 5)
        return stool_only
    except:
        return 0


def filter_plus_from_stool(stool_only, debug=False):
    avg_stool_with_plus = np.nanmean(stool_only)
    basic_filter = apply_mask_high_pass(stool_only, avg_stool_with_plus - 0.1)
    smoothed_data = gaussian_filter(basic_filter, sigma=0.5)
    if debug:
        plt.figure()
        plt.imshow(smoothed_data)
        plt.show()
    return smoothed_data


def calculate_percentage_in_range(stool_no_plus, average_stool_height, range_for_calculation=0.1):
    lower_bound = average_stool_height - range_for_calculation
    upper_bound = average_stool_height + range_for_calculation
    within_range = (stool_no_plus >= lower_bound) & (stool_no_plus <= upper_bound)
    percentage = np.sum(within_range) / stool_no_plus.size * 100
    return percentage


def calculate_leveling_angles_nan(heights, res=RES):
    rows, cols = heights.shape
    x, y = np.meshgrid(np.arange(cols) * res, np.arange(rows) * res)
    X = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))  # [x, y, 1] for the plane equation
    z = heights.ravel()
    valid_mask = ~np.isnan(z)  # Mask for non-NaN values
    X_valid = X[valid_mask]
    z_valid = z[valid_mask]
    coeffs, _, _, _ = np.linalg.lstsq(X_valid, z_valid, rcond=None)  # Solves for [a, b, c]
    a, b, c = coeffs
    theta_x = np.arctan(a)
    theta_y = np.arctan(b)
    return theta_x, theta_y


def calculate_stool_average_height(stool_only, debug=False):
    stool_no_plus = filter_plus_from_stool(stool_only, debug)
    average_stool_height = np.nanmean(stool_no_plus)
    percent_in_range = calculate_percentage_in_range(stool_no_plus, average_stool_height)
    return average_stool_height, percent_in_range


def find_stools_coordinates(raw_data, rects_list, offset_y, offset_x):
    rect_coor_list = []
    for rect in rects_list:
        x, y, w, h = cv2.boundingRect(rect)
        bounding_rect = raw_data[y:y + h, x + offset_x:x + offset_x + w]
        height, _ = calculate_stool_average_height(bounding_rect[3:-3, 3:-3], debug=False)
        rect_coordinates = [offset_y + y, offset_y + y + h, x + offset_x, x + offset_x + w]
        rect_coor_list.append([rect_coordinates, height])
    return rect_coor_list


def filter_duplicate_rectangles(rects, min_dist=10):
    """Removes rectangles whose centers are closer than `min_dist` pixels. Keeps the larger one."""
    filtered = []
    centers = []
    # x, y, w, h = cv2.boundingRect(approx)
    for rect in rects:
        x, y, w, h = cv2.boundingRect(rect)
        cx, cy = x + w / 2, y + h / 2
        area = w * h
        keep = True

        for i, (fcx, fcy, farea, _) in enumerate(centers):
            dist = np.sqrt((cx - fcx) ** 2 + (cy - fcy) ** 2)
            if dist < min_dist:
                # Decide which one to keep
                if area > farea:
                    centers[i] = (cx, cy, area, rect)
                keep = False
                break

        if keep:
            centers.append((cx, cy, area, rect))

    return [entry[-1] for entry in centers]

def extract_stools_from_roi(roi, offset_y, debug=False):
    try:
        filtered_by_avg_height = apply_mask_high_pass(roi, np.nanmean(roi) + 5)
        higher_contrast = increase_contrast(filtered_by_avg_height, 25, 10, debug)
        edges = apply_edge_detection(higher_contrast, 50, 100, debug)
        smoothed_and_closed = apply_edge_smoothing_and_closing(edges)
        stools_rect_list = rect_finder_by_contours(smoothed_and_closed, 200, 300, 200, 300, debug, epsilon_0=0.05)
        # Filter duplicates based on center proximity
        filtered_rects = filter_duplicate_rectangles(stools_rect_list, min_dist=10)
        stools_coordinates = find_stools_coordinates(roi, filtered_rects, offset_y, 0)
        return stools_coordinates
    except:
        return np.nan


def extract_stools(raw_data, debug=False):
    upper_stools = sorted(extract_stools_from_roi(raw_data[:1500, :], 0, debug=debug), key=lambda x: x[0][2])
    lower_stools = sorted(extract_stools_from_roi(raw_data[-1500:, :], raw_data.shape[0] - 1 - 1500, debug=debug),
                          key=lambda x: x[0][2])
    return upper_stools, lower_stools




def get_si_section_from_chip(raw_data, si_section, debug=False):
    temp = copy.deepcopy(raw_data)
    limits = np.array(si_section, dtype=np.float64)
    if not np.isnan(limits[0]) and not np.isnan(limits[1]):
        temp[:int(limits[0] + 200), :] = np.nan
        temp[int(limits[0] + 700):int(limits[1] - 700), :] = np.nan
        temp[int(limits[1] - 200):, :] = np.nan
    elif np.isnan(limits[0]) and not np.isnan(limits[1]):
        temp[600:int(limits[1] - 700), :] = np.nan
        temp[int(limits[1] - 200):, :] = np.nan
    elif not np.isnan(limits[0]) and np.isnan(limits[1]):
        temp[:int(limits[0] + 200), :] = np.nan
        temp[int(limits[0] + 700):-600, :] = np.nan
    else:
        temp[0:1000, :] = np.nan
        temp[1500:3200, :] = np.nan
        temp[3800:, :] = np.nan
    if debug:
        plt.figure()
        plt.imshow(temp)
        plt.show()
    return temp



def level_by_si(raw_data, si_section, debug=False):
    si_part_of_chip = get_si_section_from_chip(raw_data, si_section, debug=debug)
    quantized_section = quantize_data(si_part_of_chip, step_size=4, debug=debug, min_group_size=15000)
    unique_values, counts = np.unique(quantized_section[~np.isnan(quantized_section)], return_counts=True)
    sorted_indices = np.argsort(-counts)
    # Get the second most frequent value
    second_mask_val = unique_values[sorted_indices[1]]
    mask = quantized_section == second_mask_val
    # Mask everything not equal to the second most frequent
    si_part_of_chip = np.where(mask, si_part_of_chip, np.nan)
    # temp[temp != second_mask_val] = np.nan
    theta_x, theta_y = calculate_leveling_angles_nan(si_part_of_chip)
    leveled_data = apply_leveling(raw_data, -theta_x, -theta_y)
    if debug:
        plt.figure()
        plt.imshow(si_part_of_chip)
        plt.figure()
        plt.imshow(raw_data)
        plt.figure()
        plt.imshow(leveled_data)
        plt.figure()
        plt.imshow(leveled_data-raw_data)
        plt.figure()
        plt.plot(leveled_data[2750,:])
        plt.show()
    return leveled_data, theta_x, theta_y


def mask_except_regions(matrix, regions, fill_value=np.nan):
    # Start with a mask that's True everywhere (to overwrite)
    mask = np.ones_like(matrix, dtype=bool)

    # For each region, set the corresponding area to False (to keep)
    for y_s, y_e, x_s, x_e in regions:
        mask[y_s+4:y_e-4, x_s+4:x_e-4] = False

    # Apply the mask
    result = matrix.copy()
    result[mask] = fill_value
    return result


def find_stool_limit(stool_list, upper=True):
    if len(stool_list) == 0:
        return np.nan
    if upper:
        second_elements = [item[0][1] for item in stool_list]
        return max(second_elements)
    else:
        first_elements = [item[0][0] for item in stool_list]
        return min(first_elements)


def find_si_sections(upper_stool_limit, lower_stool_limit):
    si_points = []
    if np.isnan(upper_stool_limit):
        si_points.append(np.nan)
    else:
        si_points.append(upper_stool_limit)
    if np.isnan(lower_stool_limit):
        si_points.append(np.nan)
    else:
        si_points.append(lower_stool_limit)
    return si_points


def level_by_stools(raw_data, debug=False):
    leveled_data = copy.deepcopy(raw_data)
    upper_stools, lower_stools = extract_stools(raw_data, debug=debug)
    upper_stool_limit = find_stool_limit(upper_stools)
    lower_stool_limit = find_stool_limit(lower_stools, False)
    si_section = find_si_sections(upper_stool_limit, lower_stool_limit)
    regions = []
    try:
        if len(lower_stools) < 3:
            return level_by_si(raw_data,si_section, debug=debug)
        elif len(upper_stools) < 3 and upper_stools[0][0][2] > 1000:
            regions.append(upper_stools[0][0])
        elif len(upper_stools) < 3 and upper_stools[0][0][2] < 1000:
            regions.append(upper_stools[0][0])
        else:
            regions.extend([upper_stools[0][0], upper_stools[2][0]])
        regions.extend([lower_stools[0][0], lower_stools[2][0]])
        baba = mask_except_regions(raw_data, regions)
        # plt.figure()
        # plt.imshow(baba)
        # plt.show()
        calculate_leveling_angles_nan(baba)
        theta_x, theta_y = calculate_leveling_angles_nan(baba, RES)
        leveled_data = apply_leveling(leveled_data, -theta_x, -theta_y)
        if debug:
            plt.figure(1)
            plt.imshow(raw_data)
            plt.figure(2)
            plt.imshow(leveled_data)
            plt.figure(3)
            plt.imshow(raw_data - leveled_data)
            plt.figure(4)
            plt.plot(raw_data[530, :])
            plt.plot(leveled_data[530, :])
            plt.figure(5)
            plt.plot(raw_data[4300, :])
            plt.plot(leveled_data[4300, :])
            plt.show()
        return leveled_data, theta_x, theta_y
    except:
        return level_by_si(raw_data, si_section, debug=debug)



def apply_hough_lines(image_as_edges, rho, theta, threshold, angle_limit, debug=False, y_axis=False):
    lines = cv2.HoughLines(image_as_edges, rho=rho, theta=theta, threshold=threshold)
    all_lines_image = cv2.cvtColor(image_as_edges, cv2.COLOR_GRAY2BGR)
    vertical_lines = []
    coordinates = []
    img_width = image_as_edges.shape[1]
    img_height = image_as_edges.shape[0]
    if y_axis:
        val = 90
    else:
        val = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle_deg = np.degrees(theta)  # Convert angle to degrees
            if val - angle_limit < angle_deg < val + angle_limit:  # Nearly vertical lines
                vertical_lines.append((rho, theta, angle_deg))
                # Draw all detected vertical lines in **blue**
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                if y_axis:
                    coordinates.append((int(y0)))
                else:
                    coordinates.append(int(x0))
                x1, y1 = int(x0 + img_width * (-b)), int(y0 + img_height * (a))
                x2, y2 = int(x0 - img_width * (-b)), int(y0 - img_height * (a))
                cv2.line(all_lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines
    if debug:
        plt.imshow(all_lines_image)
        plt.show()
    return coordinates


def split_to_channels(normalized_data, upper_lim, lower_lim, debug=False):
    offset = 0*80
    # plt.figure()
    # plt.imshow(normalized_data)
    # plt.show()
    relevant_area = normalized_data[upper_lim+1350:upper_lim+1900, offset:]
    high_cont = increase_contrast(relevant_area, 80, 25, debug=debug)
    edges = apply_edge_detection(high_cont, 50, 100, debug=debug)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 360, threshold=100)
    all_lines_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    vertical_lines = []
    x_coordinates = []
    img_width = edges.shape[1]
    img_height = edges.shape[0]
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle_deg = np.degrees(theta)
            if -0.25 < angle_deg < 0.25:
                vertical_lines.append((rho, theta, angle_deg))
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x_coordinates.append(int(x0))
                x1, y1 = int(x0 + img_width * (-b)), int(y0 + img_height * (a))
                x2, y2 = int(x0 - img_width * (-b)), int(y0 - img_height * (a))
                cv2.line(all_lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines
    if debug:
        plt.figure()
        plt.imshow(all_lines_image)
        plt.show()
    x_coordinates = sorted(x_coordinates)
    channel_limits = []
    j = 0
    for i in range(len(x_coordinates)):
        if x_coordinates[i] - x_coordinates[j] > CHANNEL_PITCH:
            channel_limits.append([offset + x_coordinates[j], offset + x_coordinates[i]])
            if debug:
                plt.figure()
                plt.imshow(relevant_area[:, x_coordinates[j]: x_coordinates[i]])
                plt.show()
            j = i
    return channel_limits


def level_data(raw_data, debug=False):
    copy_of_data_for_leveling = copy.deepcopy(raw_data)
    copy_of_data_for_leveling[:1000, :] = np.nan
    copy_of_data_for_leveling[1500:3000, :] = np.nan
    copy_of_data_for_leveling[4000:, :] = np.nan
    theta_x, theta_y = calculate_leveling_angles_nan(copy_of_data_for_leveling, RES)
    leveled_data = apply_leveling(raw_data, -theta_x, -theta_y)
    leveled_section = apply_leveling(copy_of_data_for_leveling, -theta_x, -theta_y)
    if debug:
        plt.figure()
        plt.imshow(leveled_section)
        plt.figure()
        plt.imshow(leveled_data)
        plt.show()
    return leveled_data - np.nanmean(leveled_section)


def add_mech_or_newton_reg(stool_list, flip=False):
    merged_list = []
    i = 0
    for stool in stool_list:
        if i % 2 == 0:
            if flip:
                identity = 'N'
            else:
                identity = 'M'
        else:
            if flip:
                identity = 'M'
            else:
                identity = 'N'
        stool.extend(identity)
        merged_list.append(stool)
        i += 1
    return merged_list


def add_mech_or_newton_special(stool_list, identity):
    merged_list = []
    for stool in stool_list:
        stool.extend(identity)
        merged_list.append(stool)
        if identity == 'M':
            identity = 'N'
        else:
            identity = 'M'
    return merged_list


def add_identifier_to_stool_list(upper_list, lower_list):
    merged_list = []
    lower_with_idnetifier = add_mech_or_newton_reg(lower_list)
    if len(upper_list) == 3:
        upper_with_identifier = add_mech_or_newton_reg(upper_list)
    elif len(upper_list) == 2 and upper_list[0][0][2] > 1000:
        upper_with_identifier = add_mech_or_newton_special(upper_list, 'N')
    else:
        upper_with_identifier = add_mech_or_newton_special(upper_list, 'M')
    merged_list.extend(lower_with_idnetifier)
    merged_list.extend(upper_with_identifier)
    return merged_list


def add_identifier_to_stool_list_flip(upper_list, lower_list):
    merged_list = []
    lower_with_idnetifier = add_mech_or_newton_reg(lower_list, True)
    if len(upper_list) == 3:
        upper_with_identifier = add_mech_or_newton_reg(upper_list, True)
    elif len(upper_list) == 2 and upper_list[0][0][2] > 1000:
        upper_with_identifier = add_mech_or_newton_special(upper_list, 'M')
    else:
        upper_with_identifier = add_mech_or_newton_special(upper_list, 'N')
    merged_list.extend(lower_with_idnetifier)
    merged_list.extend(upper_with_identifier)
    return merged_list


def get_average_stool_height(stool_list):
    interim_sum = 0
    num_elems = 0
    for stool in stool_list:
        if stool[2] == 'M':
            interim_sum += stool[1]
            num_elems += 1
    try:
        return interim_sum/num_elems
    except:
        return 0


def normalize_stool_height_by_si(stool_list, raw_data, file_name):
    # si_part = np.nanmean(raw_data[3100:3900, :])
    si_part = np.nanmean(raw_data[3200:3800, :])
    stool_height_list = []
    master, stamp, rep, chip_name, channel_num, row, col, master_stamp, master_stamp_rep, short_id, row_col = utils.extract_channel_name_params(file_name)
    for stool in stool_list:
        stool_height_list.append([file_name.replace('.npy', ''), stool[1]-si_part, stool[2], master, stamp, rep, col,
                                  row, channel_num, master_stamp_rep, master_stamp])
    return stool_height_list


def filter_data_for_bg_only(roi, debug=False):
    quantized_data = quantize_data(roi, step_size=2, debug=debug)
    min_val = np.nanmin(quantized_data)
    min_mask = quantized_data == min_val
    masked_data = roi*min_mask
    masked_data = np.where(masked_data == 0, np.nan, masked_data)
    if debug:
        plt.figure()
        plt.imshow(masked_data)
        plt.show()
    return masked_data


def elliptical_paraboloid(XY, a, b, c, d, e):
    X, Y = XY
    return a*X**2 + b*Y**2 + c*X + d*Y + e



def fit_elliptical_paraboloid(data, debug=False):
    rows, cols = data.shape
    # Create X, Y grid
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    Z = data  # Heights from your matrix
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    params, _ = curve_fit(elliptical_paraboloid, (X_flat, Y_flat), Z_flat)
    Z_fit = elliptical_paraboloid((X, Y), *params)
    print(np.sqrt(np.mean((Z_fit-data) ** 2)))
    if debug:
        plt.figure()
        plt.imshow(data)
        plt.figure()
        plt.imshow(Z_fit)
        plt.figure()
        plt.imshow(Z_fit-data, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        plt.show()
    return np.sqrt(np.mean((Z_fit-data) ** 2))


def apply_gradient(roi, factor=1, debug=False):
    grad_mat = np.array([
        [0, -1, 0],
        [-1, 0, 1],
        [0, 1, 0]
    ])
    # grad_res = np.convolve(roi, grad_mat)
    grad_res = convolve2d(roi, grad_mat * factor, mode='same', boundary='symm')
    if debug:
        plt.figure()
        plt.imshow(grad_res)
        plt.show()
    return grad_res


def create_mask_for_most_frequent(processed_matrix):
    # Ignore NaNs and get unique values with their counts
    unique, counts = np.unique(processed_matrix[~np.isnan(processed_matrix)], return_counts=True)

    # Find the most frequent value
    most_frequent_value = unique[np.argmax(counts)]

    # Create a mask for the most frequent value
    mask = processed_matrix == most_frequent_value

    return mask, most_frequent_value


def find_alignment_marks(raw_data, debug=False):
    center_list = np.nan
    rect_list = []
    am_center_list = []
    filtered_data = filter_data_for_bg_only(raw_data[-400:, :], debug=debug)
    left_part, right_part = filtered_data[:, :filtered_data.shape[1]//2], filtered_data[:, filtered_data.shape[1]//2:]
    j = 0
    for relevant_part in [left_part, right_part]:
        i = 0
        offset_x = 0
        higher_contrast = increase_contrast(relevant_part, 15, 3, debug)
        while len(rect_list) == 0 and i < 40:
            edges = apply_edge_detection(higher_contrast, 50 - i, 100 - (2 * i), False)
            smoothed = apply_edge_smoothing_and_closing(edges)
            rect_list = rect_finder_by_contours(smoothed, 70, 95, 70, 95,
                                                debug=debug, epsilon_0=0.01)
            if j == 1:
                offset_x = filtered_data.shape[1]//2
            center_list = find_rect_center(rect_list, offset_x=offset_x)
            i += 2
        if len(center_list) > 0:
            am_center_list.append(center_list)
        j += 1
    return am_center_list


def to_calc(pivot_trans_mid, xpander_trans_mid,  alignment_marks, mid_point):
    try:
        pivot_diff_list = []
        xpander_diff_list = []
        for mark_center in alignment_marks:
            if mark_center[0] > mid_point:
                pivot_diff = TO_TARGET - (mark_center[0]-pivot_trans_mid) * RES
                pivot_diff_list.append(pivot_diff)
                xpander_diff = TO_TARGET - (mark_center[0]-xpander_trans_mid) * RES
                xpander_diff_list.append(xpander_diff)
            else:
                pivot_diff = (pivot_trans_mid-mark_center[0]) * RES - TO_TARGET
                pivot_diff_list.append(pivot_diff)
                xpander_diff = (xpander_trans_mid - mark_center[0]) * RES - TO_TARGET
                xpander_diff_list.append(xpander_diff)
        return np.nanmean(pivot_diff_list), np.nanmean(xpander_diff_list)
    except:
        return np.nan, np.nan

def get_channel_limits(upper_stool_limit, lower_stool_limit):
    limits = [upper_stool_limit, lower_stool_limit]  # Or: [np.nan, 1200], [500, np.nan], etc.
    # Convert to array for easier nan checks
    limits = np.array(limits, dtype=np.float64)
    # Case 1: Both are valid numbers
    if not np.isnan(limits[0]) and not np.isnan(limits[1]):
        start = limits[0] + 700#1000 #was 1360
        end = limits[1] - 1000
    # Case 2: Only limits[0] is nan
    elif np.isnan(limits[0]) and not np.isnan(limits[1]):
        end = limits[1] - 1100
        start = end-1200
    # Case 3: Only limits[1] is nan
    elif not np.isnan(limits[0]) and np.isnan(limits[1]):
        start = limits[0] + 1000
        end = start + 1260
    else:
        start = ROI_START
        end = ROI_END
    return start, end


def get_roi_boundaries(channel_limits, scanned_site, upper_stool_limit, lower_stool_limit, debug=False):
    start_point, end_point = get_channel_limits(upper_stool_limit, lower_stool_limit)
    start = False
    end = False
    if channel_limits[0] - CHANNEL_BUFFER < 1:
        s_limit = channel_limits[0]
        e_limit = channel_limits[1] + CHANNEL_BUFFER
        start = True
    elif channel_limits[1] + CHANNEL_BUFFER > scanned_site.shape[1] - 1:
        e_limit = channel_limits[1]
        s_limit = channel_limits[0] - CHANNEL_BUFFER
        end = True
    else:
        s_limit = channel_limits[0] - CHANNEL_BUFFER
        e_limit = channel_limits[1] + CHANNEL_BUFFER
    channel_under_test = scanned_site[int(start_point):int(end_point), s_limit:e_limit] # here
    if debug:
        plt.figure()
        plt.imshow(channel_under_test)
        plt.show()
    return channel_under_test, start, end



def extract_xpander_height(xpander_only, background_height, debug=False):
    xpander_copy = copy.deepcopy(xpander_only)
    radius_px = int(np.round(62 / RES))
    center = (xpander_only.shape[0] // 2, xpander_only.shape[1] // 2)  # (row, col)
    yy, xx = np.ogrid[:xpander_only.shape[0], :xpander_only.shape[1]]
    dist_sq = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    mask = dist_sq <= radius_px ** 2
    xpander_copy[mask] = 0
    quantized_data = quantize_data(xpander_copy, debug=debug, step_size=1.5, min_group_size=10000)
    min_val = np.nanmin(quantized_data)
    flat_section = np.where(quantized_data == min_val, xpander_copy, np.nan)
    if debug:
        plt.figure()
        plt.imshow(xpander_copy)
        plt.figure()
        plt.imshow(flat_section)
        plt.show()
    return np.nanmean(flat_section)-background_height


def pad_matrices_with_nan(matrix_list):
    # Find the max number of rows and columns
    max_rows = max(mat.shape[0] for mat in matrix_list)
    max_cols = max(mat.shape[1] for mat in matrix_list)

    padded_list = []
    for mat in matrix_list:
        rows, cols = mat.shape
        # Create a new matrix filled with nan
        padded = np.full((max_rows, max_cols), np.nan)
        # Copy the original matrix into the top-left corner
        padded[:rows, :cols] = mat
        padded_list.append(padded)

    return padded_list


def extract_mid_stage(raw_data, xpander_coordinates, pivot_coordinates, background_height, debug=False, compensation=0):
    roi = raw_data[xpander_coordinates[1]+MID_STAGE_BUFFER:pivot_coordinates[0]-MID_STAGE_BUFFER,
          pivot_coordinates[2]:pivot_coordinates[3]]
    if debug:
        plt.figure()
        plt.imshow(roi)
        plt.show()
    return roi, abs(np.nanmean(roi) - background_height)-compensation


def create_plane_3d(angle_deg, x_size, y_size, offset_height):
    # x_pixels = 100
    # y_pixels = 100

    # Pixel size in micrometers (µm)
    pixel_size = RES  # µm per pixel

    # Physical size in µm
    x_um = np.linspace(0, x_size * pixel_size, x_size)
    y_um = np.linspace(0, y_size * pixel_size, y_size)
    xx, yy = np.meshgrid(x_um, y_um)

    # Angle in degrees and radians
    # theta_deg = 41
    theta_rad = np.radians(angle_deg)

    # Create the plane: decreasing in +Y direction, height in µm
    zz = np.tan(theta_rad) * yy
    zz -= np.nanmin(zz)
    zz += offset_height
    return zz


def get_rect_pair_average_val(list1, list2, threshold=4.0):
    try:
        result = []
        max_len = max(len(list1), len(list2))
        for i in range(max_len):
            if i < len(list1) and i < len(list2):
                x1, y1 = list1[i]
                x2, y2 = list2[i]

                if abs(y1 - y2) < threshold:
                    avg_x = np.nanmean([x1, x2])
                    avg_y = np.nanmean([y1, y2])
                    result.append([avg_x, avg_y])
                else:
                    result.append([np.nan, np.nan])

            elif i < len(list1):
                result.append(list1[i])
            else:
                result.append(list2[i])
        return result
    except:
        return np.nan


def prepare_run(out_path, stool_offset, pop_file_path):
    if stool_offset == 0:
        compensation = 'No compensation'
    else:
        compensation = f'compensation {stool_offset}'
    os.makedirs(os.path.join(out_path, "Residue Vs. Fit"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "Residue Vs. Target"), exist_ok=True)
    residue_path_fit = os.path.join(out_path, "Residue Vs. Fit")
    residue_path_target = os.path.join(out_path, "Residue Vs. Target")
    try:
        with open(pop_file_path, "r") as pop_file:
            pop_data = json.load(pop_file)
    except:
        pop_data = ''
    residue_path = [residue_path_fit, residue_path_target]
    return compensation, residue_path, pop_data


def run_analysis(data_path, out_path, output_file_name, etched, pop_file_path, scan_configuration, scan_state, stool_offset=0):
    upload_to_merged_data = False
    main_df = pd.read_csv(MAIN_CSV_PATH)
    main_stool_df = pd.read_csv(STOOL_CSV_PATH)
    compensation, residue_path, pop_data = prepare_run(out_path, stool_offset, pop_file_path)
    for path in data_path:
        interim_lst = []
        stool_list_data = []
        for file in os.listdir(path):
            # if file != 'B-067-003-009-06-0U-25.npy':
            #     continue
            iteration = 0
            if file.endswith('.npy'):
                split_name = file.replace('.npy', '').split('-')
                try:
                    raw_data = np.load(os.path.join(path, file), allow_pickle=True)
                    # plt.figure()
                    # plt.imshow(raw_data)
                    # plt.show()
                    leveled_data, theta_x, theta_y = level_by_stools(raw_data, debug=False)
                    # plt.figure()
                    # plt.imshow(leveled_data)
                    # plt.show()
                    upper_stools_list, lower_stool_list = extract_stools(leveled_data, debug=False)
                    upper_stool_limit = find_stool_limit(upper_stools_list)
                    lower_stool_limit = find_stool_limit(lower_stool_list, False)
                    si_section = find_si_sections(upper_stool_limit, lower_stool_limit)
                    merged_stool_list = add_identifier_to_stool_list(upper_stools_list, lower_stool_list)
                    avg_stool_height = get_average_stool_height(merged_stool_list)
                    normalized_by_stool_height = leveled_data - avg_stool_height
                    normalize_stool_height = normalize_stool_height_by_si(merged_stool_list, leveled_data, file)
                    stool_list_data.extend(normalize_stool_height)
                    norm_avg_stool_height = get_average_stool_height(stool_list_data)
                    channel_limits = split_to_channels(normalized_by_stool_height, upper_stool_limit, lower_stool_limit,
                                                       debug=False)
                    si_part = np.nanmean(get_si_section_from_chip(normalized_by_stool_height, si_section, debug=False))
                    pivot_list = []
                    for channel in channel_limits:
                        try:
                            master, stamp, rep, chip_name, channel_num, row, col, master_stamp, master_stamp_rep, short_id, row_col = utils.extract_channel_name_params(
                                file, iteration)
                            var, var_comment = utils.get_var_and_comment(pop_data, chip_name, int(channel_num))
                            if var == 'B-029-000':
                                atp = True
                                pivot_extra_offset = 90
                            else:
                                # continue
                                pivot_extra_offset = 50
                                atp = False
                            # if col in ['2', '3', '4', '5']: #for 52
                            #     stool_offset = 0.0
                            #     compensation = f'compensation {stool_offset}'
                            # else:
                            #     stool_offset = 0.75
                            #     compensation = f'compensation {stool_offset}'
                            # if row in ['0Q', '0R', '0S', '0T', '0U']: # for 56
                            #     stool_offset = 0.4
                            #     compensation = f'compensation {stool_offset}'
                            # elif chip_name == '02-0P':
                            #     stool_offset = 0.4
                            #     compensation = f'compensation {stool_offset}'
                            # else:
                            #     stool_offset = 0.0
                            #     compensation = f'compensation {stool_offset}'

                            element = '-'.join(split_name[0:-1]) + '-' + str(channel_num)
                            channel_under_test, start, end = get_roi_boundaries(channel, normalized_by_stool_height,upper_stool_limit, lower_stool_limit, debug=False)
                            channel_under_test = utils.fix_rotation(channel_under_test, debug=False, plug=False, rep=True)
                            alignment_marks = find_alignment_marks(channel_under_test, debug=False)
                            if atp:
                                xpander_only, xpander_coordinates = find_xpander(channel_under_test, debug=False, step_size=1, atp=atp)
                            else:
                                xpander_only, xpander_coordinates = find_xpander(channel_under_test, debug=False)
                            xpander_height = extract_xpander_height(xpander_only, si_part, debug=False)
                            (xpander_mid_x_coordinate, xpander_mid_y_coordinate, axial_focal, trans_focal, rmse,
                             xpander_mid_axial_geo, apex_height_at_opt_center, apex_height_geo_center) = (
                                extract_xpander_focal_length_and_mid_point(
                                    xpander_only, xpander_coordinates, element, residue_path, si_part,
                                    fit_radius=FIT_RAD, debug=False))
                            pivot_coordinates, pivot_only = find_pivot(channel_under_test, avg_stool_height,
                                                                       xpander_coordinates[1]+pivot_extra_offset,
                                                                       xpander_coordinates[2], xpander_coordinates[3],
                                                                       debug=False, atp=atp)
                            mid_stage, mid_stage_height = extract_mid_stage(channel_under_test, xpander_coordinates,
                                                                            pivot_coordinates, 0,
                                                                            debug=False, compensation=stool_offset)
                            _, mid_stage_height_uncompensated = extract_mid_stage(channel_under_test, xpander_coordinates,
                                                                            pivot_coordinates, 0,
                                                                            debug=False, compensation=0)
                            (pb, pt, phd, pivot_trans_mid, pivot_axial_mid, pivot_optical_mid, angle_from_phd,
                             pivot_optical_mid_uncompensated, sloped_section, upper_cross_y, lower_cross_y, left_squares_mid,
                             right_squares_mid) = extract_pivot_height_and_mid_point_params(
                                pivot_only, pivot_coordinates, debug=False, atp=atp, stool_offset=stool_offset)
                            if atp:
                                pivot_list.append(pivot_only[30:85, int(pivot_trans_mid - pivot_coordinates[2]) - 25:int(
                                    pivot_trans_mid - pivot_coordinates[2]) + 26])
                            else:
                                pivot_list.append(pivot_only[65:120, int(pivot_trans_mid-pivot_coordinates[2])-25:int(pivot_trans_mid-pivot_coordinates[2])+26])
                            angle_at_fiber, angle_at_channel_height, pivot_rmse = extract_pivot_angle(pivot_only, pt, etched,
                                                                                                         stool_height=0, debug=False, artificial_offset=stool_offset)
                            angle_at_fiber_uncompensated, _, _ = extract_pivot_angle(pivot_only, pt, etched,
                                                                                                         stool_height=0, debug=False, artificial_offset=0)
                            optic_section = sloped_section[50:130,10:-10]
                            plane = create_plane_3d(angle_from_phd, optic_section.shape[1], optic_section.shape[0], pt)
                            pivot_squares_avg = get_rect_pair_average_val(left_squares_mid, right_squares_mid)
                            pivot_axial_focal, pivot_trans_focal = extract_pivot_focal_length(optic_section-plane, fit_radius=30,
                                                                                              debug=False, file_name=element)
                            to, xto = to_calc(pivot_trans_mid, xpander_mid_x_coordinate, alignment_marks, channel_under_test.shape[1]//2)
                            print('ANGLE::: ', angle_at_fiber)
                            (pxd, pxto, _, _, pxd_optical, uncompensated_pxd_optical,upper_cross_to_xpander_opt,
                            upper_cross_to_xpander_geo,lower_cross_to_xpander_opt, lower_cross_to_xpander_geo,
                            first_pivot_square_to_xpander_opt_center, first_pivot_square_to_xpander_geo_center,
                            second_pivot_square_to_xpander_opt_center, second_pivot_square_to_xpander_geo_center,
                            third_pivot_square_to_xpander_opt_center, third_pivot_square_to_xpander_geo_center,
                            fourth_pivot_square_to_xpander_opt_center, fourth_pivot_square_to_xpander_geo_center,
                             pxd_geo_pivot_opt_xpander) = (
                                extract_channel_parameters(xpander_mid_x_coordinate, xpander_mid_y_coordinate,
                                                           pivot_trans_mid, pivot_axial_mid, np.nan, pivot_optical_mid,
                                                           pivot_optical_mid_uncompensated, upper_cross_y,
                                                           lower_cross_y, pivot_squares_avg, xpander_mid_axial_geo))
                            pivot_height = norm_avg_stool_height - mid_stage_height + stool_offset
                            pivot_mean = np.mean([pb, pt])
                            data_list = [element, pb, pt, phd, pivot_axial_focal, pivot_trans_focal, angle_at_fiber,
                                         angle_at_channel_height, axial_focal, trans_focal, pxd, pxto, to, xto, pxd_optical,
                                         norm_avg_stool_height, xpander_mid_x_coordinate, xpander_mid_y_coordinate, pivot_rmse,
                                         rmse, var, col, row, channel_num, var_comment, master, stamp, rep, master_stamp,
                                         master_stamp_rep, row_col, short_id, f'{scan_configuration}, {compensation}', scan_state, np.nan, xpander_height,
                                         np.nan, np.nan, iteration, theta_x, theta_y, mid_stage_height, angle_from_phd, pivot_height,
                                         os.path.join(path, file), uncompensated_pxd_optical, mid_stage_height_uncompensated, angle_at_fiber_uncompensated,
                                         upper_cross_to_xpander_opt,
                                         upper_cross_to_xpander_geo, lower_cross_to_xpander_opt,
                                         lower_cross_to_xpander_geo,
                                         first_pivot_square_to_xpander_opt_center,
                                         first_pivot_square_to_xpander_geo_center,
                                         second_pivot_square_to_xpander_opt_center,
                                         second_pivot_square_to_xpander_geo_center,
                                         third_pivot_square_to_xpander_opt_center,
                                         third_pivot_square_to_xpander_geo_center,
                                         fourth_pivot_square_to_xpander_opt_center,
                                         fourth_pivot_square_to_xpander_geo_center, pivot_mean,
                                         pxd_geo_pivot_opt_xpander, apex_height_at_opt_center, apex_height_geo_center
                                         ]
                            interim_lst.append(data_list)
                            iteration += 1
                        except:
                            element = '-'.join(split_name[0:-1]) + '-' + split_name[-1]
                            print('issue with ', element)
                            iteration += 1
                except:
                    element = '-'.join(split_name[0:-1]) + '-' + split_name[-1]
                    print('issue with ', element)
                    iteration += 1
        final_data_as_df = pd.DataFrame(interim_lst, columns=COLUMN_LIST)
        il_val = setup_visual.load_from_confocal_results(verbose=False,
                                                         pic=True,
                                                         boc_angle_conf={'angle_type': 'PHD', 'target': 18.2},
                                                         results_df=final_data_as_df,
                                                         average_per_rep=False,
                                                         outpath=out_path,
                                                         pxd_geo_p=True,
                                                         stool_compression=False,
                                                         ref_file='../report_simulation_ref.csv', )
        il_df = pd.DataFrame(il_val, columns=['component_id', 'il_at_center'])
        final_data_as_df = final_data_as_df.merge(il_df, on='component_id', how='left')
        stool_data_as_df = pd.DataFrame(stool_list_data, columns=STOOL_COLUMN_LIST)
        final_data_as_df.to_csv(os.path.join(out_path, output_file_name + master_stamp_rep + '.csv'), index=False)
        stool_data_as_df.to_csv(os.path.join(out_path, output_file_name + master_stamp_rep  + 'stools.csv'), index=False)
        if upload_to_merged_data:
            main_df = pd.concat([main_df, final_data_as_df], ignore_index=True, sort=False)
            main_df.to_csv(MAIN_CSV_PATH, index=False)
            main_stool_df = pd.concat([main_stool_df, stool_data_as_df], ignore_index=True, sort=False)
            main_stool_df.to_csv(STOOL_CSV_PATH, index=False)


def ask_user_for_data():
    result = {}

    def browse_multiple_folders():
        # For multiple paths, we'll allow user to pick multiple times and add to listbox
        folder = filedialog.askdirectory()
        if folder:
            data_paths_listbox.insert(tk.END, folder)

    def remove_selected_paths():
        for index in reversed(data_paths_listbox.curselection()):
            data_paths_listbox.delete(index)

    def browse_pop_file(entry):
        file = filedialog.askopenfilename(
            defaultextension=".pop",
            filetypes=[("POP files", "*.pop")],
            initialdir=r"G:\Shared drives\Design\Printing Database\Populations of Printed Masters"
        )
        if file:
            entry.delete(0, tk.END)
            entry.insert(0, file)

    def submit():
        # Collect all paths from the listbox
        data_paths = [data_paths_listbox.get(i) for i in range(data_paths_listbox.size())]
        if not data_paths:
            messagebox.showerror("Error", "Please add at least one Data Path.")
            return

        # Validate stool offset
        try:
            stool_offset_value = float(stool_offset_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Stool offset must be a number.")
            return

        result['data_path'] = data_paths
        result['outpath'] = outpath_entry.get()
        result['output_filename'] = output_filename_entry.get()
        result['etched'] = bool(etched_var.get())
        result['pop_file_path'] = pop_path_entry.get()
        result['stool_offset'] = stool_offset_value
        result['scan_configuration'] = scan_config_var.get()

        root.destroy()

    root = tk.Tk()
    root.title("Enter Scan Details")

    # Data Paths (Multiple)
    tk.Label(root, text="Data Paths:").grid(row=0, column=0, sticky='w')
    data_paths_listbox = tk.Listbox(root, width=50, height=4, selectmode=tk.MULTIPLE)
    data_paths_listbox.grid(row=0, column=1, rowspan=2, sticky='w')
    tk.Button(root, text="Add Path", command=browse_multiple_folders).grid(row=0, column=2)
    tk.Button(root, text="Remove Selected", command=remove_selected_paths).grid(row=1, column=2)

    # Output Path
    tk.Label(root, text="Output Path:").grid(row=2, column=0, sticky='w')
    outpath_entry = tk.Entry(root, width=50)
    outpath_entry.grid(row=2, column=1)
    tk.Button(root, text="Browse", command=lambda: outpath_entry.insert(0, filedialog.askdirectory())).grid(row=2, column=2)

    # Output File Name
    tk.Label(root, text="Output File Name:").grid(row=3, column=0, sticky='w')
    output_filename_entry = tk.Entry(root, width=50)
    output_filename_entry.grid(row=3, column=1)

    # Etched Sample Checkbox
    etched_var = tk.IntVar()
    tk.Checkbutton(root, text="Etched sample?", variable=etched_var).grid(row=4, column=1, sticky='w')

    # POP File Path
    tk.Label(root, text="POP File Path:").grid(row=5, column=0, sticky='w')
    pop_path_entry = tk.Entry(root, width=50)
    pop_path_entry.grid(row=5, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_pop_file(pop_path_entry)).grid(row=5, column=2)

    # Stool Offset
    tk.Label(root, text="Stool Offset:").grid(row=6, column=0, sticky='w')
    stool_offset_entry = tk.Entry(root, width=50)
    stool_offset_entry.grid(row=6, column=1)

    # Dropdown: Scan Configuration
    scan_config_options = ["wafer level", "chip level"]
    tk.Label(root, text="Scan Configuration:").grid(row=7, column=0, sticky='w')
    scan_config_var = tk.StringVar(value=scan_config_options[0])
    ttk.Combobox(root, textvariable=scan_config_var, values=scan_config_options, state='readonly').grid(row=7, column=1)

    # Submit button
    tk.Button(root, text="Submit", command=submit).grid(row=8, column=1, pady=10)

    root.mainloop()

    return (result['data_path'], result['outpath'], result['output_filename'],
            result['etched'], result['pop_file_path'], result['stool_offset'],
            result['scan_configuration'])




def main():
    # path_for_data = r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\B-052-001-014\csv and npy'
    debug = False
    if debug:
        outpath = r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\rep_batch_runs'
        output_file_name = 'B-065-polaris_batch_runs'
        etched = False
        stool_offset = 0
        data_path = [
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 3',
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 10',
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 15',
            r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 20',
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 21',
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 22',
            r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 23',
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 24',
            # r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 25',
                       ]
        pop_file_path = r"G:\Shared drives\Design\Printing Database\Populations of Printed Masters\B-065 Dagoba M4R.pop"
        scan_configuration = 'SCIL uncoated_meas'
    else:
        data_path, outpath, output_file_name, etched, pop_file_path, stool_offset, scan_configuration = ask_user_for_data()
    scan_state = 'no coating'
    run_analysis(data_path, outpath, output_file_name, etched, pop_file_path, scan_configuration, scan_state, stool_offset)


if __name__ == '__main__':
    main()

