import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
from scipy import ndimage
from src.confocal_analysis_codes.ellipsoid_fit_confocal import run_on_images
from scipy.ndimage import gaussian_filter
from tkinter import filedialog, simpledialog, messagebox
# from profiler_cv.config import CONFIG
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.ndimage import label
import subprocess
from scipy.signal import convolve2d
import json
from scipy.optimize import leastsq
from tqdm import tqdm
import src.confocal_analysis_codes.scan_analysis_tools as utils


PIVOT_TRANS_BUFFER = 25
MID_STAGE_BUFFER = 10
RECT_BUFFER = 15
EXTERNAL_FIX = 0
RES = 0.138*2
INIT_LOC = 25
STOP_HEIGHT = 17
DELTA = 1
START_FIT_INDEX = 10#350 #10
FIND_PIVOT_X_OFFSET = 200
FIND_PIVOT_Y_OFFSET = 600
FIBER_OFFSET = 0.175
PIVOT_PLANE_BUFFER = 0 * 2
LOW_SECTION_LENGTH = 13
TOP_TO_BOT_DIST = 150//2
BOT_LEN = 25
LONG_BOT_LENGTH = 60//2
FIT_RADIUS_FOCAL = 325//2
FIT_RADIUS_RES = 219
XPANDER_FINDING_OFFSET = 50#25 #1000#p1 was 650 #350
XPANDER_FINDING_LIMIT =650#650#750# 1550 # 1700# p1 was 1700#1500
XPANDER_BOX_LENGTH = 450#920//2
XPANDER_FALLBACK_LENGTH = 450#920
TARGET_HEIGHT_PHD = 17.9
TARGET_ANGLE = 39.8


COLUMN_LIST = ['component_id', 'pivot_bottom', 'pivot_top', 'PHD', 'pivot_axial_focal', 'pivot_trans_focal',
               'pivot_angle_SI_reference', 'pivot_angle_pivot_top_reference', 'axial_focal_length',
               'trans_focal_length', 'PXD_geo', 'PXTO_opt', 'TO_geo', 'XTO_opt', 'PXD_opt',
               'stool_height', 'xpander_mid_x', 'xpander_mid_y', 'pivot_RMSE', 'xpander_RMSE', 'variation_id',
               'column', 'row', 'channel', 'variation_comment', 'master', 'stamp', 'rep', 'master-stamp',
               'wafer_id', 'column-row', 'column-row-channel', 'scan_configuration', 'DUT_state',
               'dicing_vendor', 'xpander_height', 'PXTO_geo', 'XTO_geo','leveling_angle_x', 'leveling_angle_y',
               'effective_channel_height', 'pivot_height']


def apply_leveling(raw_data, theta_x, theta_y):
    x = np.linspace(0, raw_data.shape[1] - 1, raw_data.shape[1]) * RES
    y = np.linspace(0, raw_data.shape[0] - 1, raw_data.shape[0]) * RES
    xx, yy = np.meshgrid(x, y)
    rotated_matrix = np.sin(theta_x) * xx + np.cos(theta_x) * raw_data
    rotated_matrix = np.sin(theta_y) * yy + np.cos(theta_y) * rotated_matrix
    return rotated_matrix


def level_stool(stool_section, debug=False):
    leveled_data = stool_section
    avg_val = np.nanmean(leveled_data)
    filtered_stool = apply_mask_low_pass(leveled_data, avg_val)
    filtered_stool = apply_mask_high_pass(filtered_stool, np.nanmean(filtered_stool) - 0.5)
    theta_x, theta_y = calculate_leveling_angles_nan(filtered_stool, RES)
    leveled_stool = apply_leveling(stool_section, -theta_x, -theta_y)
    # avg_val = np.nanmean(leveled_stool)
    # filtered_stool = apply_mask_low_pass(leveled_stool, avg_val)
    # bandpass = apply_mask_low_pass(filtered_stool, np.nanmean(filtered_stool) + 0.1)
    # bandpass = apply_mask_high_pass(bandpass, np.nanmean(bandpass) - 0.1)
    # avg_val = np.nanmean(bandpass)
    if debug:
        plt.figure()
        plt.imshow(stool_section)
        plt.figure()
        plt.imshow(leveled_stool)
        plt.figure()
        plt.imshow(leveled_stool - stool_section)
        plt.show()
    return leveled_stool, avg_val, theta_x, theta_y

def get_background_height(raw_data, debug=False):
    plot_for_debug = debug
    if plot_for_debug:
        plt.figure()
        plt.imshow(raw_data[950:1000, 200:250])
        plt.show()
    return np.nanmean(raw_data[950:1000, 200:250])


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


def rotate_pivot(relevant_pivot_section, debug=False):
    rotated_pivot = apply_rotation_to_pivot(relevant_pivot_section, RES, -41)
    if debug:
        plt.figure()
        plt.imshow(rotated_pivot)
        plt.show()
    return rotated_pivot


def print_error(message):
    print(f"\033[91m{message}\033[0m")


def print_success(message):
    print(f"\033[92m{message}\033[0m")


def extract_pivot_focal_length(optic_section, fit_radius, debug=False):
    try:
    #     if pivot_only.shape[0] > 150:
    #         lim = 65
    #     else:
    #         lim = 20
    #     x_mid = pivot_only.shape[1] // 2
    #     relevant_pivot_section = np.rot90(pivot_only[lim:-lim, x_mid - 45:x_mid + 45], k=1)
    #     rotated_pivot = rotate_pivot(relevant_pivot_section, debug=debug)
        pivot_focal, original, fit = run_on_images(optic_section, fit_radius, RES, '', '', '', di=False, save_residue=False)
        axial = pivot_focal['focal_length_x']
        trans = pivot_focal['focal_length_y']
        # plt.figure()
        # plt.imshow(original-fit, vmin=-0.1, vmax=0.1)
        # plt.show()
        # print(axial, trans)
        print_success('Extracted Pivot Focal successfully')
        return axial, trans
    except:
        print_error('Failed pivot focal')
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
    # Convert back to 8-bit and absolute values for viewing
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

# min_width=300, max_width=520, min_height=300, max_height=370
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


def find_pivot(raw_data, background_height, y_start, x_start, x_end, debug=False, var=''):
    try:
        y_start += 30
        i = 5
        while i < 100:
            # print(i)
            normalized_data = raw_data - background_height
            filtered = apply_basic_filter(normalized_data, i - 1, i + 1)
            # filtered = apply_mask_low_pass(normalized_data, i)
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
            uint8_image = np.uint8(filtered[y_start:, x_start:x_end])
            filt = apply_gaussian_blur(uint8_image)
            high_cont = increase_contrast(filt, 25, 5)
            # if debug:
            #     plt.figure()
            #     plt.imshow(raw_data)
            #     plt.show()
            # edges = apply_edge_detection(high_cont, 50, 100, debug=False)
            edges = apply_edge_detection(high_cont, 120, 240, debug=False)

            smoothed_and_closed_edges = apply_edge_smoothing_and_closing(edges, debug=False)  # this
            if var in ['B-029-000', 'B-029-000_rot']:
                pivot_surounding_rect = rect_finder_by_contours(edges, debug=debug, min_width=145, max_width=185, min_height=106, max_height=138)
            else:
                smoothed_and_closed_edges = apply_edge_smoothing_and_closing(edges, debug=False)  # this
                pivot_surounding_rect = rect_finder_by_contours(smoothed_and_closed_edges, debug=debug)
            if len(pivot_surounding_rect) == 1:
                break
            i += 0.2
        pivot_coordinates, pivot_only = extract_pivot_from_rect(pivot_surounding_rect, normalized_data, x_start,
                                                                y_start)
        # buffer = -2  # -0.8
        # normalized_data = raw_data - background_height

        # relevant_section = normalized_data[y_start:, x_start:x_end]
        # no_background = apply_mask_high_pass(relevant_section, 11 - buffer)  # was 11
        # normalized_image = cv2.normalize(no_background, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255
        # uint8_image = np.uint8(normalized_image)
        # denoised = cv2.cv2.bilateralFilter(uint8_image, d=5, sigmaColor=75, sigmaSpace=75)
        # max_index = np.unravel_index(np.argmax(denoised), denoised.shape)[0]
        # high_contrast = increase_contrast(denoised[:max_index + 30, :], 50, 3, debug=debug)  # clip was 7, matrix was 50
        # filtered_high_contrast = apply_mask_low_pass(high_contrast, np.max(high_contrast) // 3, np.min(high_contrast))
        # edges = apply_edge_detection(filtered_high_contrast, 20, 40, debug=debug)  # 25 50
        # smoothed_and_closed_edges = apply_edge_smoothing_and_closing(edges, debug=debug)
        # pivot_surounding_rect = rect_finder_by_contours(smoothed_and_closed_edges, debug=debug)
        # pivot_coordinates, pivot_only = extract_pivot_from_rect(pivot_surounding_rect, normalized_data, x_start,
        #                                                         y_start)
        # debug=True
        if debug:
            plt.figure()
            plt.imshow(pivot_only)
            plt.show()
        print_success('Pivot found')
        return pivot_coordinates, pivot_only
    except:
        print_error('Failed finding pivot')
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
    rough_filter = apply_basic_filter(pivot_section, element_height - 2, element_height - 1)#was 0.5 HERE
    cross_section = filter_cross_section(rough_filter)
    if debug:
        plt.figure()
        plt.imshow(rough_filter)
        plt.figure()
        plt.imshow(cross_section)
        plt.show()
    cm = ndimage.center_of_mass(cross_section)
    cm_y = cm[0]
    cm_x = cm[1]
    return cm_x, cm_y


def clear_small_groups(raw_mask):
    labeled_regions, num_features = label(raw_mask)
    region_sizes = np.bincount(labeled_regions.ravel())
    largest_region_label = np.argmax(region_sizes[1:]) + 1
    indices = np.where(labeled_regions == largest_region_label)
    min_row, max_row = indices[0].min(), indices[0].max()
    min_col, max_col = indices[1].min(), indices[1].max()
    return (min_row, max_row, min_col, max_col), (labeled_regions == largest_region_label)


def create_mask_for_most_frequent(processed_matrix):
    # Ignore NaNs and get unique values with their counts
    unique, counts = np.unique(processed_matrix[~np.isnan(processed_matrix)], return_counts=True)

    # Find the most frequent value
    most_frequent_value = unique[np.argmax(counts)]

    # Create a mask for the most frequent value
    mask = processed_matrix == most_frequent_value

    return mask, most_frequent_value


def quantize_section(pivot_only, debug=False, step_size=1, min_group_size=1000):
    min_val, max_val = np.nanmin(pivot_only), np.nanmax(pivot_only)
    bins = np.arange(min_val, max_val + step_size, step_size)
    quantized_matrix = np.digitize(pivot_only, bins) - 1
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

    # Create a mask for values outside the threshold range
    mask = np.abs(section - mean) > num_sigma * std

    # Replace outliers with np.nan
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


def get_high_section(quantized_pivot, pivot_only, debug=False, design=False):
    unique_values, counts = np.unique(quantized_pivot[~np.isnan(quantized_pivot)], return_counts=True)
    if len(unique_values) < 2:
        threshold = unique_values[0]  # If only one unique value, use it
    else:
        largest_value = unique_values[-1]
        second_largest_value = unique_values[-2]
        largest_count = counts[-1]
        second_largest_count = counts[-2]
        if (
                second_largest_count > largest_count
                and largest_value - second_largest_value <= 2
        ):
            threshold = second_largest_value
        else:
            threshold = largest_value
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
        # plt.figure()
        # plt.imshow(masked_original[final_mask[0] + PIVOT_PLANE_BUFFER:final_mask[1] - PIVOT_PLANE_BUFFER,
        #            final_mask[2] + PIVOT_PLANE_BUFFER:final_mask[3] - PIVOT_PLANE_BUFFER])
        plt.show()
    return high_section, np.nanmean(apply_sigma_band_pass(high_section)), final_mask


def get_low_section(pivot_only, top_mask, debug=False, just_section=False, design=False):
    if pivot_only.shape[0] > 150:
        bot_len = LONG_BOT_LENGTH
        if design:
            bot_len = 189
    else:
        bot_len = BOT_LEN
    # suspected_bot = pivot_only[5:top_mask[0] - TOP_TO_BOT_DIST, top_mask[2]:top_mask[3]]
    # min_index = np.unravel_index(np.argmin(suspected_bot), suspected_bot.shape)
    y_coord = 0  # min_index[0] + 5
    # plt.figure(9999)
    # plt.imshow(pivot_only)
    # plt.show()
    if design:
        top_to_bo_dist = int(TOP_TO_BOT_DIST*0.2525/0.04)
    else:
        top_to_bo_dist = TOP_TO_BOT_DIST
    bot_start = top_mask[0] - top_to_bo_dist
    bot_section = pivot_only[bot_start - bot_len:bot_start, top_mask[2]:top_mask[3]]
    # try:
    # bot_section = pivot_only[y_coord - 10:y_coord+15, top_mask[2]:top_mask[3]]
    # bot_section = pivot_only[y_coord - 5:y_coord + 15, top_mask[2]:top_mask[3]]

    # except:
    #     bot_section = pivot_only[y_coord - 5:y_coord + 15, top_mask[2]:top_mask[3]]
    if just_section:
        return bot_section, y_coord - 5  # y_coord - 10
    filtered_bot = apply_mask_high_pass(bot_section, np.nanmean(bot_section) - 0.2)
    bot_sec = apply_sigma_band_pass(filtered_bot)
    if debug:
        plt.figure(9999999)
        plt.imshow(bot_sec)
        plt.show()
    return bot_sec, np.nanmean(bot_sec), bot_start


def find_pivot_geo_center(pivot_only, top_mask, high_section_height, low_section_height, debug=False, design=False):
    if pivot_only.shape[0] > 150:
        bot_len = LONG_BOT_LENGTH
        if design:
            bot_len = 189
    else:
        bot_len = BOT_LEN
    high_section = pivot_only[top_mask[0] + PIVOT_PLANE_BUFFER:top_mask[1] - PIVOT_PLANE_BUFFER,
                   top_mask[2] + PIVOT_PLANE_BUFFER:top_mask[3] - PIVOT_PLANE_BUFFER]
    # bot_start = top_mask[0] - TOP_TO_BOT_DIST
    low_section, bot_start = get_low_section(pivot_only, top_mask,
                                             just_section=True, design=design)  # pivot_only[bot_start-BOT_LEN:bot_start, top_mask[2]:top_mask[3]]

    if design:
        top_to_bo_dist = int(TOP_TO_BOT_DIST*0.2525/0.04)
    else:
        top_to_bo_dist = TOP_TO_BOT_DIST
    bot_start = top_mask[0] - top_to_bo_dist

    # upper_cross_mid_x, upper_cross_mid_y = get_cross_center(high_section[10:-10,15:-15], high_section_height)
    upper_cross_mid_x, upper_cross_mid_y = get_cross_center(high_section, high_section_height, debug=False)
    lower_cross_mid_x, lower_cross_mid_y = get_cross_center(low_section, low_section_height)
    if debug:
        plt.figure(10)
        plt.imshow(high_section)
        plt.figure(11)
        plt.imshow(low_section)
        plt.figure(12)
        plt.imshow(pivot_only)
        plt.scatter(PIVOT_PLANE_BUFFER + top_mask[2] + upper_cross_mid_x,
                    PIVOT_PLANE_BUFFER + top_mask[0] + upper_cross_mid_y)
        plt.scatter(top_mask[2] + lower_cross_mid_x, lower_cross_mid_y + bot_start - bot_len)
        plt.show()
    mid_y = (lower_cross_mid_y + bot_start - bot_len + PIVOT_PLANE_BUFFER + top_mask[0] + upper_cross_mid_y) / 2
    mid_x = (PIVOT_PLANE_BUFFER + top_mask[2] + upper_cross_mid_x + top_mask[2] + lower_cross_mid_x) / 2
    return mid_x, mid_y


def find_rects_from_section(section, debug=False, design=False):
    if design:
        min_width = 113
        max_width = 157
    else:
        min_width = 18
        max_width = 25
    higher_contrast = increase_contrast(section, 15, 8, debug)
    edges = apply_edge_detection(higher_contrast, 15, 30, debug)
    left_rects = rect_finder_by_contours(edges, min_width, max_width, min_width, max_width, debug)
    centers = find_rect_center(left_rects)
    return centers


def compute_average_center(centers):
    if not centers:
        return None  # Handle case with no rectangles
    avg_x = np.mean([c[0] for c in centers])
    avg_y = np.mean([c[1] for c in centers])
    return avg_x, avg_y


def find_pivot_rects_mid_point(pivot_only, rotated=False, design=False):
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
    left_rect_center = find_rects_from_section(left_side, debug=False, design=design)
    right_rects_center = find_rects_from_section(right_side, debug=False, design=design)
    if rotated:
        _, left_avg_center = compute_average_center(left_rect_center)
        _, right_avg_center = compute_average_center(right_rects_center)
    else:
        left_avg_center, _ = compute_average_center(left_rect_center)
        right_avg_center, _ = compute_average_center(right_rects_center)
    return (left_avg_center + right_avg_center + buffer) // 2, left_avg_center, right_avg_center + buffer


def extract_pivot_height_and_mid_point_params(pivot_only, pivot_coordinates, debug=False, st=False, res=RES, design=False):
    try:
        left_mid, right_mid = np.nan, np.nan
        pivot_copy = copy.deepcopy(pivot_only)
        quantized_pivot = quantize_section(pivot_copy, debug=debug, step_size=2, min_group_size=2000)
        pivot_high_plane, high_section_height, top_mask = get_high_section(quantized_pivot, pivot_only, debug=debug, design=design)
        pivot_low_plane, low_section_height, bot_start = get_low_section(pivot_only, top_mask, debug=debug, design=design)
        geo_mid_x_alternative, geo_mid_y = find_pivot_geo_center(pivot_only, top_mask, high_section_height, low_section_height,
                                                     debug=debug, design=design)
        try:
            geo_mid_x, left_mid, right_mid = find_pivot_rects_mid_point(pivot_only, design=design)
        except:
            geo_mid_x = geo_mid_x_alternative
        pivot_mid_point = pivot_only[:top_mask[1], top_mask[2]:top_mask[3]].shape[1] // 2
        length = pivot_only.shape[0]
        vector = np.linspace(0, 0.2525 * (length - 1), length)
        # plt.figure()
        # plt.plot(vector, pivot_only[:, pivot_only.shape[1]//2])
        # plt.show()
        # pivot_mid_point = pivot_only.shape[1] // 2
        sloped_pivot_section = pivot_only[:top_mask[1], top_mask[2]:top_mask[3]]
        pivot_cross_section = sloped_pivot_section[:, pivot_mid_point]
        if st:
            index_closest_height = np.argmin(np.abs(pivot_cross_section - 14.2))
        else:
            index_closest_height = np.argmin(np.abs(pivot_cross_section - 23.41))
        pivot_optical_mid = index_closest_height + pivot_coordinates[0]
        trans_mid = pivot_coordinates[2] + geo_mid_x
        axial_mid = pivot_coordinates[0] + geo_mid_y
        if debug:
            plt.figure(999)
            plt.imshow(pivot_only)
            plt.scatter(geo_mid_x, index_closest_height)
            plt.scatter(geo_mid_x, geo_mid_y)
            plt.show()
        print_success('Extracted Pivot parameters successfully')
        return (high_section_height, low_section_height, high_section_height - low_section_height, trans_mid, axial_mid,
                pivot_optical_mid, pivot_coordinates[2] + left_mid, pivot_coordinates[2] + right_mid, top_mask[0],
                bot_start, sloped_pivot_section)
    except:
        print_error('Failed extracting pivot params')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def generate_fit(dz_list):
    height_vec = np.arange(0, RES * len(dz_list), RES)
    fit_params = np.polyfit(height_vec[START_FIT_INDEX:], dz_list[START_FIT_INDEX:], 2)
    fit_data = (fit_params[0] * height_vec[START_FIT_INDEX:] ** 2 + fit_params[1] * height_vec[START_FIT_INDEX:]
                + fit_params[2])
    residue = dz_list[START_FIT_INDEX:] - fit_data
    # plt.figure(9999999)
    # plt.plot(residue)
    # plt.title('Lext 5100 Pivot residual')
    # plt.xlabel('pixel')
    # plt.ylabel('magnitude [um]')
    # plt.figure()
    # plt.plot(dz_list[START_FIT_INDEX:])
    # plt.show()
    return height_vec, fit_params, fit_data, residue


def get_pivot_cross_section_values(pivot_only, pivot_init_height):
    num_of_iter = 0
    rotated_pivot = np.rot90(pivot_only, 1)
    window_size = 2
    epsilon = 0.05
    mid_y_point = rotated_pivot.shape[0] // 2
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


def find_pivot_target_values(height_and_angle_data, target_height):
    interp_function = interp1d(height_and_angle_data['Pivot height'], height_and_angle_data['Angle'],
                               kind='linear', fill_value="extrapolate")
    over_sampled_pivot_height = np.linspace(min(height_and_angle_data['Pivot height']),
                                            max(height_and_angle_data['Pivot height']), 5000)
    over_sampled_angle = interp_function(over_sampled_pivot_height)
    # plt.figure()
    # plt.plot(over_sampled_pivot_height, over_sampled_angle)
    # plt.show()
    index_closest_height = np.argmin(np.abs(over_sampled_pivot_height - target_height))
    pivot_angle_at_target_height = over_sampled_angle[index_closest_height]
    return pivot_angle_at_target_height


def calc_angle_using_fit(fitted_data, target_height):
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
    angle_at_target_height = find_pivot_target_values(height_and_angle_df, target_height)
    return angle_at_target_height


def extract_pivot_angle(pivot_only, pivot_init_height, pivot_high_plane_height, stool_height=0, element=False):
    """
    calculate the pivot angle
    :param pivot_only: isolated pivot
    :param pivot_init_height: pivot top height
    :param stool_height: relevant stool height - when given will calculate the angle with reference to the stool height
    :return: angel at 23.41, angle at channel height - when stool height is known
    """
    try:
        z_list_si_ref, z_list_pt_ref = get_pivot_cross_section_values(pivot_only, pivot_init_height)
        height_vec_si, fitted_params_si, fitted_data_si, residue_si = generate_fit(z_list_si_ref)
        if element:
            angle_at_23_41_above_si = calc_angle_using_fit(fitted_data_si, pivot_high_plane_height - 7.535)
        else:
            angle_at_23_41_above_si = calc_angle_using_fit(fitted_data_si, 23.41)
        if stool_height > 0:
            relevant_channel_height = stool_height - pivot_init_height - FIBER_OFFSET
            height_vec_pt, fitted_params_pt, fitted_data_pt, residue_pt = generate_fit(z_list_pt_ref)
            angle_at_channel_height = calc_angle_using_fit(fitted_data_pt, relevant_channel_height)
        else:
            angle_at_channel_height = np.nan
        print_success('Extracted Pivot angle successfully')
        return angle_at_23_41_above_si, angle_at_channel_height, np.sqrt(np.mean((residue_si) ** 2))
    except:
        print_error('Failed extracting pivot angle')
        return np.nan, np.nan, np.nan


def reverse_relevant_array(arr):
    """
    reverse order of an array
    :param arr: input array to operate on
    :return: an array with reversed order
    """
    return arr[::-1]


def get_edge(arr):
    """
    find the location of the first maximum value
    :param arr: cross-section of data as an array
    :return: coordinate of the relevant value
    """
    return np.argmax(arr)


def find_xpander_limits(edge_data, debug=False):
    """
    find the edges of the xpander
    :param edge_data: image as an output of the canny edge detector
    :return: xpander limits
    """
    offset = 200
    issues = False
    # debug = True
    if debug:
        plt.imshow(edge_data, cmap='viridis', interpolation='nearest')
        plt.show()
    # y_mid_1 = edge_data.shape[0] // 2 - offset
    # y_mid_2 = edge_data.shape[0] // 2 + offset
    # x_mid_1 = edge_data.shape[1] // 2 - offset
    # x_mid_2 = edge_data.shape[1] // 2 + offset
    x_mid = edge_data.shape[1] // 2 + offset
    y_mid = edge_data.shape[0] // 2 + offset

    x_start = x_mid - get_edge(reverse_relevant_array(edge_data[y_mid, 0:x_mid + 1]))
    x_end = x_mid + get_edge(edge_data[y_mid, x_mid:edge_data.shape[1]])
    i = 1
    while x_end - x_start < 470 or x_end - x_start > 510:
        x_start = x_mid - get_edge(reverse_relevant_array(edge_data[y_mid + i, 0:x_mid + 1]))
        x_end = x_mid + get_edge(edge_data[y_mid + i, x_mid:edge_data.shape[1]])
        i -= 1
        if i > 600:
            break

    # while x_end == x_mid:
    #     x_end = x_mid + get_edge(edge_data[y_mid + i, x_mid:edge_data.shape[1]])
    #     i += 1
    #     issues = True
    # i = 1
    # while x_start == x_mid:
    #     x_start = x_mid - get_edge(reverse_relevant_array(edge_data[y_mid + i, 0:x_mid + 1]))
    #     i += 1
    #     issues = True
    i = 1
    y_start = y_mid - get_edge(reverse_relevant_array(edge_data[0:y_mid + 1, x_mid]))
    y_end = y_mid + get_edge(edge_data[y_mid:edge_data.shape[0], x_mid])
    while y_end - y_start < 470 or y_end - y_start > 510:
        y_start = y_mid - get_edge(reverse_relevant_array(edge_data[:y_mid + 1, x_mid + i]))
        y_end = y_mid + get_edge(edge_data[y_mid:edge_data.shape[0], x_mid + i])
        i -= 1
        if i > 600:
            break
    return x_start, x_end, y_start, y_end, issues



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



def find_xpander(leveled_raw_data, background_height, debug=False, var='', st=False):
    """
    Function that isolates the Xpander - used in the initial labeling process
    """
    no_background = apply_mask_high_pass(leveled_raw_data, background_height + 3)

    try:
        if var in ['B-029-000', 'B-029-000_rot']:
            temp_data = quantize_section(no_background[XPANDER_FINDING_OFFSET:XPANDER_FINDING_LIMIT, :], debug=debug, step_size=1.25, min_group_size=60000)

        else:
            if st:
                temp_data = quantize_section(no_background[XPANDER_FINDING_OFFSET:XPANDER_FINDING_LIMIT, :], debug=debug, step_size=1, min_group_size=60000)
            else:
                temp_data = quantize_section(no_background[XPANDER_FINDING_OFFSET:XPANDER_FINDING_LIMIT, :], debug=debug, step_size=2.5, min_group_size=40000)
        temp_data[temp_data == 0] = np.nan
        # plt.figure()
        # plt.imshow(temp_data)
        # plt.show()
        f = most_frequent_nonzero_nonmax2(temp_data)
        masked_only = np.where(f, no_background[XPANDER_FINDING_OFFSET:XPANDER_FINDING_LIMIT, :], np.nan)
        # plt.imshow(masked_only)
        # plt.show()
        high_contrast_image = increase_contrast(masked_only,
                                                35, 18, debug=debug)  # 1600
        # high_contrast_image = increase_contrast(leveled_raw_data[XPANDER_FINDING_OFFSET:XPANDER_FINDING_LIMIT, :],
        #                                         35, 25, debug=debug)  # 1600
        edges_data = apply_edge_detection(high_contrast_image, 60, 120, debug=debug)
        laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
        x_lines = sorted(apply_hough_lines(laplace_filter, 1, np.pi / 3600, 200, 0.25,
                                           debug=debug))
        i = 1
        while len(x_lines) < 2 and i < 50:
            edges_data = apply_edge_detection(high_contrast_image, 50-i, 100-(2*i), debug=debug)
            laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
            x_lines = sorted(apply_hough_lines(laplace_filter, 1, np.pi / 3600, 200-i, 0.25,
                                               debug=debug))
            i += 1
        y_lines = sorted(apply_hough_lines(laplace_filter[:, :], 1, np.pi / 3600, 400,
                                           0.25, debug=debug, y_axis=True))
        i = 1
        while len(y_lines) < 2 and i < 50:
            edges_data = apply_edge_detection(high_contrast_image, 50-i, 100-(2*i), debug=debug)
            laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
            y_lines = sorted(apply_hough_lines(laplace_filter[:, :], 1, np.pi / 3600, 400-i,
                                               0.25, debug=debug, y_axis=True))
            i += 5
        pair_x, _ = closest_number_pair(x_lines, XPANDER_BOX_LENGTH)
        x_start, x_end = pair_x[1][0], pair_x[1][1]
        pair_y, _ = closest_number_pair(y_lines, XPANDER_BOX_LENGTH)
        if pair_y is None:
            if y_lines[0] > laplace_filter.shape[1] // 2:
                y_start, y_end = XPANDER_FINDING_OFFSET + y_lines[0]-XPANDER_FALLBACK_LENGTH, XPANDER_FINDING_OFFSET + y_lines[0]
            else:
                y_start, y_end = XPANDER_FINDING_OFFSET + y_lines[0], XPANDER_FINDING_OFFSET + y_lines[0]+XPANDER_FALLBACK_LENGTH
        else:
            y_start, y_end = XPANDER_FINDING_OFFSET + pair_y[1][0], XPANDER_FINDING_OFFSET + pair_y[1][1]
        if debug:
            plt.figure()
            plt.imshow(leveled_raw_data[y_start:y_end, x_start:x_end])
            plt.show()
        print_success('found xpander')
        return leveled_raw_data[y_start:y_end, x_start:x_end], [y_start, y_end, x_start, x_end]
    except:
        print_error('Failed finding the xpander')
        return np.nan, np.nan




# def find_xpander(leveled_raw_data, debug=False):
#     """
#     find and isolate the xpander from the rest of the image
#     :param leveled_raw_data: leveled image
#     :param debug: when true will plot figures for debugging and troubleshooting
#     :return: isolated xpander and xpander coordinates
#     """
#     try:
#         normalized_image = cv2.normalize(leveled_raw_data, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255
#         uint8_image = np.uint8(normalized_image[:750, :])
#
#         high_contrast_image = increase_contrast(leveled_raw_data[:750, :], 50, 25, debug=debug)  # was 700
#         edges_data = apply_edge_detection(high_contrast_image, 75, 150, debug=debug)
#         laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
#         smoothed_and_closed_edges = apply_edge_smoothing_and_closing(laplace_filter, debug=debug, kernel_shape=3,
#                                                                      num_iter=3)
#         # rect_finder_by_contours(smoothed_and_closed_edges, 400, 600, 400, 600, debug=debug)
#         x_start, x_end, y_start, y_end, issues = find_xpander_limits(smoothed_and_closed_edges)
#         if debug:
#             plt.figure()
#             plt.imshow(leveled_raw_data[y_start:y_end, x_start:x_end])
#             plt.show()
#         print_success('Found Xpander')
#         return leveled_raw_data[y_start:y_end, x_start:x_end], [y_start, y_end, x_start, x_end]
#     except:
#         # plt.figure()
#         # plt.imshow(edges_data)
#         # plt.show()
#         print_error('Failed finding Xpander')
#         return np.nan, np.nan


def extract_xpander_focal_length_and_mid_point(xpander_only, xpander_coordinates, file, outpath, fit_radius,
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
        if debug:
            plt.figure()
            plt.imshow(xpander_only)
            plt.show()
        result, original, fit = run_on_images(xpander_only, fit_radius, RES, file, outpath, '', True,
                               False, False, symmetric=False, plug=False, save_residue=False)
        # try:
        #     run_on_images(xpander_only, FIT_RADIUS_RES, RES, file, outpath, '', True,
        #                                           False, False, symmetric=True, plug=False, save_residue=True)
        # except:
        #     pass

        # print(result['center_y'])
        xpander_mid_x_coordinate = xpander_coordinates[2] + result['center_x']
        xpander_mid_y_coordinate = xpander_coordinates[0] + result['center_y']
        mid_y_geo = xpander_coordinates[0] + xpander_only.shape[0]//2

        trans_focal = result['focal_length_x']
        axial_focal = result['focal_length_y']
        rmse = result['RMSE']
        if debug:
            plt.figure()
            plt.imshow(original-fit, vmin=-0.05, vmax=0.05)
            plt.colorbar()
            # plt.scatter(result['center_x'], result['center_y'])
            plt.figure()
            plt.imshow(xpander_only)
            plt.scatter(result['center_x'], result['center_y'])
            plt.figure()
            plt.plot(original[:,int(result['center_y'])]-fit[:,int(result['center_y'])])
            plt.show()
        print_success('Extracted Xpander Focal successfully')
        return xpander_mid_x_coordinate, xpander_mid_y_coordinate, axial_focal, trans_focal, rmse, mid_y_geo
    except:
        # plt.figure()
        # plt.imshow(xpander_only)
        # plt.show()
        print_error('Failed Xpander analysis')
        return np.nan, np.nan, np.nan, np.nan


def extract_channel_parameters(xpander_x, xpander_y, pivot_x, pivot_y, alignment_mid, pivot_optical_mid,
                               xpadner_geo_axial_mid,  xpadner_geo_trans_mid):
    """
    calculate cahnnel related parameters
    :param xpander_x: xpander mid x coordinate
    :param xpander_y: xpander mid y coordinate
    :param pivot_x: pivot mid x coordinate
    :param pivot_y: pivot mid y coordinate
    :param alignment_mid: alignment marks middle point
    :return: pivot xpander distance, pivot xpander trans offset, trans offset
    """
    try:
        pxd_optical_center = (pivot_optical_mid - xpander_y) * RES
        pxd_geo = (pivot_y - xpadner_geo_axial_mid) * RES
        pxto = (pivot_x - xpander_x) * RES
        pxto_geo = (pivot_x-xpadner_geo_trans_mid)*RES
        to = (pivot_x - alignment_mid) * RES
        xto = (xpander_x - alignment_mid) * RES
        xto_geo = (xpadner_geo_trans_mid - alignment_mid) * RES
        print_success('Extracted channel parameters successfully')
        return pxd_geo, pxto, to, xto, pxd_optical_center, pxto_geo, xto_geo
    except:
        print_error('Failed channel analysis')
        return np.nan, np.nan, np.nan, np.nan, np.nan


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


def find_rect_center(rect_list, offset=0):
    """
    find the center of all the rectangles in the input list
    :param rect_list: coordinates list of the rectangles
    :return: list with rectangle centers
    """
    centers = []
    try:
        for rect in rect_list:
            x, y, w, h = cv2.boundingRect(rect)  # Get bounding box for the rectangle
            center_x = x + w / 2 + offset  # Calculate the x-coordinate of the center
            center_y = y + h / 2  # Calculate the y-coordinate of the center
            centers.append((center_x, center_y))  # Store the center coordinates
        return centers
    except:
        return False


def apply_gradient(roi, factor=1):
    grad_mat = np.array([
        [0, -1, 0],
        [-1, 0, 1],
        [0, 1, 0]
    ])
    # grad_res = np.convolve(roi, grad_mat)
    grad_res = convolve2d(roi, grad_mat * factor, mode='same', boundary='symm')
    return grad_res


def find_rects_centers(raw_data, bg_height, debug=False):
    """
    find the alignment marks and their center for TO calculation
    :param raw_data: leveled data
    :param background_height: SI height used for basic filtering
    :param debug: when true will plot figures for debugging and troubleshooting
    :return: coordinates of the two alignment marks and their midpoint
    """
    i = 0
    # plt.imshow(raw_data)
    # plt.show()
    try:
        offset = 0
        am_list = []
        locally_leveled_data = raw_data  # level_raw_data(raw_data)
        background_height = get_background_height(locally_leveled_data)
        buffer = 0.4
        only_background = apply_mask_low_pass(locally_leveled_data, bg_height + buffer, 0)
        # plt.figure()
        # plt.imshow(only_background)
        # plt.show()
        quantized_section = quantize_section(only_background[1100:1750, :], step_size=0.025, debug=debug).astype(np.uint8)
        mask, most_frequent_value = create_mask_for_most_frequent(quantized_section)
        negative_mask = ~mask
        inverse_filtered_matrix = np.where(negative_mask, quantized_section, np.nan)
        # grad_image = apply_gradient(quantized_section)
        # plt.figure()
        # plt.imshow(grad_image)
        # plt.show()
        # only_background_filtered = apply_median_filter(quantized_section, 3, turn_to_grayscale=debug)
        # high_contrast_image = increase_contrast(quantized_section, 25, 10, debug=debug, blur=False)  # was 3
        # filtered_data = apply_gaussian_blur(high_contrast_image, 3)
        # while True:
        for val in set(np.unique(inverse_filtered_matrix)):
            filtered_matrix = np.where(inverse_filtered_matrix == val, inverse_filtered_matrix, np.nan).astype(np.uint8)
            # plt.figure(9999)
            # plt.imshow(filtered_matrix)
            # plt.show()
            # unique_values, counts = np.unique(processed_section, return_counts=True)
            # most_frequent = unique_values[np.argmax(counts)]
            # filtered_matrix = np.where(processed_section == most_frequent, processed_section, np.nan)
            # mask = ~np.isnan(filtered_matrix)
            edges = apply_edge_detection(filtered_matrix, 50, 100, False)
            # cleared_data = apply_mask_low_pass(filtered_data, 250 - i, 0)
            # edges = apply_edge_detection(cleared_data, 15, 30, blurr=False)
            # plt.figure()
            # plt.imshow(cleared_data)
            # plt.show()
            smoothed_and_closed_edges = apply_edge_smoothing_and_closing(edges, debug=debug)
            rect_list = rect_finder_by_contours(edges, 70, 90, 70, 90, debug=debug, square=False, epsilon_0=0.05)
            rect_center_coordinates_lst = find_rect_center(rect_list, offset)
            am_list.extend(rect_center_coordinates_lst)
            if len(am_list) > 1:
                break
            if rect_center_coordinates_lst:
                if rect_center_coordinates_lst[0][0] > inverse_filtered_matrix.shape[1] // 2:
                    inverse_filtered_matrix = inverse_filtered_matrix[:, :inverse_filtered_matrix.shape[1] // 2]
                elif rect_center_coordinates_lst[0][0] < inverse_filtered_matrix.shape[1] // 2:
                    offset = inverse_filtered_matrix.shape[1] // 2
                    inverse_filtered_matrix = inverse_filtered_matrix[:, inverse_filtered_matrix.shape[1] // 2:]
        am_list = sorted(am_list)
        if debug:
            plt.figure()
            plt.imshow(only_background)
            # plt.figure()
            # plt.imshow(quantized_section)
            plt.scatter(am_list[0][0], am_list[0][1] + 600)
            plt.scatter(am_list[1][0], am_list[1][1] + 600)
            plt.show()
        mid_point_rects = (am_list[0][0] - am_list[1][0]) / 2 + am_list[1][0]
        print_success('Found alignment marks')
        return am_list, mid_point_rects
    except:
        print_error('Failed finding rect center')
        return np.nan, np.nan


def add_relevant_id(u_d_data, master, rep_num):
    file_ids = []  # List to store file IDs
    for idx, row in u_d_data.iterrows():
        # Generate a file_id for each row (e.g., based on the index or other logic)
        try:
            split_component_name = row['long name'].split(' ')
            row = split_component_name[1].zfill(2)
            col = split_component_name[4].zfill(2)
            channel = split_component_name[6].zfill(2)
        except:
            split_component_name = row['Component ID'].split('-')
            row = split_component_name[0].zfill(2)
            col = split_component_name[1].zfill(2)
            channel = split_component_name[2].zfill(2)
        # full_name_as_list = [master, rep_num, col, row, channel]
        full_name_as_list = [master, rep_num, row, col, channel]

        full_name_joined = '-'.join(full_name_as_list)
        file_ids.append(full_name_joined)
    u_d_data['file_id'] = file_ids
    return u_d_data


def find_stool(leveled_stool_section, background_height, debug, is_st):
    try:
        # section = apply_mask_low_pass(leveled_stool_section, np.nanmax(leveled_stool_section) - 5)
        # leveled_data = leveled_stool_section
        # theta_x, theta_y = calculate_leveling_angles_nan(section)
        # leveled_data = apply_leveling(leveled_data, -theta_x, -theta_y)
        # plt.figure(99999)
        # plt.imshow(leveled_data)
        # plt.show()
        if background_height == 0:
            background_height = np.nanmean(leveled_stool_section[:20, 400:440])
        normalized_data = leveled_stool_section - background_height
        if is_st:
            no_background = apply_mask_high_pass(normalized_data, np.nanmean(normalized_data) + 5)  # was 11

        else:
            no_background = apply_mask_high_pass(normalized_data, np.nanmean(normalized_data) + 15)  # was 11
        # plt.figure()
        # plt.imshow(normalized_data)
        # plt.show()
        blurred_image = apply_gaussian_blur(no_background)
        high_contrast = increase_contrast(blurred_image, 35, 7, debug=debug)  # clip was 7, matrix was 50
        edges = apply_edge_detection(high_contrast, 30, 60, debug=debug)  # was 40 and 80
        smoothed_and_closed_edges = apply_edge_smoothing_and_closing(edges, debug=debug)
        stool_only_rect = rect_finder_by_contours(smoothed_and_closed_edges, 200, 300, 200, 400, debug) # 400 600
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


def analyze_stool(leveled_stool_section, background_height, debug=False, is_st=False):
    if debug:
        plt.figure()
        plt.imshow(leveled_stool_section)
        plt.show()
    try:
        stool_only = find_stool(leveled_stool_section, background_height, debug, is_st)
        if debug:
            plt.figure()
            plt.imshow(stool_only)
            plt.show()
        stool_no_plus = filter_plus_from_stool(stool_only, debug)
        average_stool_height = np.nanmean(stool_no_plus)
        percent_in_range = calculate_percentage_in_range(stool_no_plus, average_stool_height)
        print(average_stool_height, percent_in_range)
        return average_stool_height, percent_in_range
    except:
        return np.nan, np.nan


def find_xpander_edge(data, background_height, debug=False):
    no_background = apply_mask_high_pass(data, background_height + 3)
    temp_data = quantize_section(no_background[200:XPANDER_FINDING_LIMIT, :], debug=debug,
                                 step_size=2, min_group_size=60000)
    temp_data[temp_data == 0] = np.nan
    f = most_frequent_nonzero_nonmax2(temp_data)
    masked_only = np.where(f, no_background[200:XPANDER_FINDING_LIMIT, :], np.nan)

    high_contrast_image = increase_contrast(masked_only,
                                            35, 5, debug=debug)  # 1600
    edges_data = apply_edge_detection(high_contrast_image, 50, 100, debug=debug)
    laplace_filter = apply_laplacian_enhancing(edges_data, debug=debug)
    y_lines = sorted(apply_hough_lines(laplace_filter, 1, np.pi / 360, 650, 0.25,
                                       debug=debug, y_axis=True))
    if debug:
        plt.imshow(laplace_filter)
        plt.show()
    return y_lines[0] + 200



def split_to_channel_stool(raw_data, u_d_data, file, background_height=0):
    extension = '_d'
    if isinstance(u_d_data, pd.DataFrame):
        stitch_type = u_d_data.loc[u_d_data['file_id'] == file.replace('.npy', ''), 'stitching type'].iloc[0]
    else:
        stitch_type = 'd' #CHANGE THIS
    if stitch_type in ['b', 'u']:
        stool_section = raw_data[:1000, :]
        suspected = raw_data[-1000:, :]
        xpander_edge = find_xpander_edge(suspected, background_height, False)
        xpander_edge += raw_data.shape[0]-1000
        # try:
        #     raw_data = raw_data[xpander_edge - 80:xpander_edge + 1000:]
        # except:
        #     raw_data = raw_data[xpander_edge - 80:, :]
        extension = '_u'
    else:
        stool_section = raw_data[-1000:, :]
        suspected = raw_data[:1000, :]
        xpander_edge = find_xpander_edge(suspected, background_height, False)
    buffer = 200
    if xpander_edge < 80:
        raw_data = raw_data[buffer:buffer + xpander_edge+1000, :]
    else:
        raw_data = raw_data[xpander_edge - 80:xpander_edge + 1000:]
        # try:
        #     raw_data = raw_data[xpander_edge - 80:xpander_edge + 1000:]
        # except:
        #     raw_data = raw_data[:xpander_edge+1000, :]
        extension = '_d'
    # plt.imshow(raw_data)
    # plt.show()
    return raw_data, stool_section, extension


def calculate_leveling_angles_nan(heights, res=RES):
    rows, cols = heights.shape

    # Create coordinate grids for x and y
    x, y = np.meshgrid(np.arange(cols) * res, np.arange(rows) * res)

    # Flatten the data
    X = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))  # [x, y, 1] for the plane equation
    z = heights.ravel()

    # Filter out NaN values
    valid_mask = ~np.isnan(z)  # Mask for non-NaN values
    X_valid = X[valid_mask]
    z_valid = z[valid_mask]

    # Fit the plane using linear regression
    coeffs, _, _, _ = np.linalg.lstsq(X_valid, z_valid, rcond=None)  # Solves for [a, b, c]
    a, b, c = coeffs

    # Calculate leveling angles
    theta_x = np.arctan(a)  # Tilt angle in x-direction
    theta_y = np.arctan(b)  # Tilt angle in y-direction

    return theta_x, theta_y


def calculate_leveling_angles(heights):
    rows, cols = heights.shape

    # Create coordinate grids for x and y
    x, y = np.meshgrid(np.arange(cols) * RES, np.arange(rows) * RES)

    # Flatten the data for fitting
    X = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))  # [x, y, 1] for the plane equation
    z = heights.ravel()

    # Fit the plane using linear regression
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)  # Solves for [a, b, c]
    a, b, c = coeffs

    # Calculate leveling angles
    theta_x = np.arctan(a)  # Tilt angle in x-direction
    theta_y = np.arctan(b)  # Tilt angle in y-direction

    return theta_x, theta_y


def level_larger_raw_data(raw_data, u_d_data, file, debug):
    if isinstance(u_d_data, pd.DataFrame):
        stitch_type = u_d_data.loc[u_d_data['file_id'] == file.replace('.npy', ''), 'stitching type'].iloc[0]
    else:
        stitch_type = 'd' #CHANGE THIS
    if stitch_type in ['b', 'u']:
        start = 600
        end = 1400
    else:
        start = 1500
        end = 2100
    section = raw_data[start:end, 50:950]
    quantized_section = quantize_section(section, step_size=4, debug=debug)
    mask, most_frequent_value = create_mask_for_most_frequent(quantized_section)
    section = np.where(mask, section, np.nan)
    theta_x, theta_y = calculate_leveling_angles_nan(section)
    leveled_data = copy.deepcopy(raw_data)
    leveled_data = apply_leveling(leveled_data, -theta_x, -theta_y)
    if debug:
        plt.figure(1)
        plt.imshow(raw_data)
        plt.colorbar()
        plt.figure(2)
        plt.imshow(leveled_data)
        plt.colorbar()
        plt.figure(3)
        plt.imshow(leveled_data - raw_data)
        plt.show()
        plt.figure()
        plt.hist(np.ravel(raw_data[start:end, 50:950]) - np.nanmean(raw_data[start:end, 50:950]))
        plt.figure()
        plt.title('Post leveling')
        plt.xlabel('pixel value')
        plt.ylabel('number of pixels')
        plt.hist(np.ravel(leveled_data[start:end, 50:950]) - np.nanmean(leveled_data[start:end, 50:950]))
        print(np.nanstd(raw_data[start:end, 50:950]))
        print(np.nanstd(leveled_data[start:end, 50:950]))
        plt.show()
    rel_area = np.where(mask, leveled_data[start:end, 50:950], np.nan)
    background_height = np.nanmean(rel_area)
    return leveled_data, background_height, theta_x, theta_y


def get_current_commit_hash():
    # Run the git command to get the current commit hash
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode == 0:
        # If the command was successful, return the commit hash
        return result.stdout.strip()
    else:
        # If the command failed, return None
        return None


def find_relevant_slot_name(wafer_map, target_chip):
    matching_dict = next((d for d in wafer_map if isinstance(d, dict) and d.get('slot_name') == target_chip), None)
    return matching_dict


def extract_channel_name_params(file):
    """
    Function that gets the file name and extract channel ID parameters
    """
    try:
        split_name = file.split('-')
        master = split_name[1].replace('0','')
        stamp = str(int(split_name[2]))
        rep = str(int(split_name[3]))
        channel_num = str(int(split_name[6].replace('.npy', '')))
        row = split_name[5]
        col = split_name[4]
        chip_name = '-'.join([col, row])
        col = col.replace('0','')
        row_col = '-'.join([col, row])
        master_stamp = '-'.join([master, stamp])
        master_stamp_rep = '-'.join([master, stamp, rep])
        short_id = '-'.join([col, row, channel_num])
        return master, stamp, rep, chip_name, channel_num, row, col, master_stamp, master_stamp_rep, short_id, row_col
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def get_wafer_map(pop_data):
    """
    Function that returns the slot catalog from the populator file
    """
    try:
        return pop_data['__Slice__']['slot_catalog']
    except:
        return np.nan


def get_var(matching_chip, channel_num):
    """
    Return the variation ID
    """
    try:
        return matching_chip['content']['__Chip__']['channels'][channel_num - 1]['Project File']["__Path__"].split(
            '\\')[-1].replace('.nanogs', '')
    except:
        return np.nan



def get_var_comment(matching_chip, channel_num):
    """
    Return the variation comment
    """
    try:
        return matching_chip['content']['__Chip__']['channels'][channel_num - 1]['Design Parameters']
    except:
        return np.nan


def get_var_and_comment(pop_data, chip_name, channel_num):
    """
    function that returns the current variation and variation comment
    """
    try:
        wafer_map = get_wafer_map(pop_data)
        matching_chip = find_relevant_slot_name(wafer_map, chip_name)
        var_data = get_var(matching_chip, channel_num)
        var_comment = get_var_comment(matching_chip, channel_num)
        return var_data, var_comment
    except:
        return np.nan, np.nan


def find_xpander_height(xpander_only, background_height, debug=False):
    """
    Function that gets an isolated xpander and returns the xpander height parameter
    """
    try:
        xpander_copy = copy.deepcopy(xpander_only)
        radius_px = int(np.round(62 / RES))
        center = (xpander_only.shape[0] // 2, xpander_only.shape[1] // 2)  # (row, col)
        yy, xx = np.ogrid[:xpander_only.shape[0], :xpander_only.shape[1]]
        dist_sq = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
        mask = dist_sq <= radius_px ** 2
        xpander_copy[mask] = 0
        quantized_data = quantize_section(xpander_copy, debug=debug, step_size=1, min_group_size=10000)
        max_val = np.nanmax(quantized_data)
        flat_section = np.where(quantized_data == max_val, xpander_copy, np.nan)
        if debug:
            plt.figure()
            plt.imshow(xpander_copy)
            plt.figure()
            plt.imshow(flat_section)
            plt.show()
        return np.nanmean(flat_section)-background_height
    except:
        return np.nan


def find_geo_trans_xpander_params(xpander_roi, offset, debug=False):
    """
    Function that calculates the transversal geometrical mid of the Xpander using the 'forehead' section
    """
    try:
        # plt.figure()
        # plt.imshow(xpander_roi)
        # plt.show()
        high_contrast_image = increase_contrast(xpander_roi, 5, 9, debug=debug)
        edge_image = apply_edge_detection(high_contrast_image, 30,60, debug=debug)
        lines_coordinates = apply_hough_lines(edge_image, 1, np.pi / 360, 10, 0.25, debug=debug)
        relevant_pair, _ = closest_number_pair(lines_coordinates, 150,5) #
        trans_mid = np.mean(relevant_pair[1]) + offset
        return trans_mid
    except:
        return np.nan


def extract_neighbouring_pivot(leveled_raw_data, upper_limit):
    left_roi = leveled_raw_data[upper_limit+75:upper_limit+300, 50:200]
    right_roi = leveled_raw_data[upper_limit+75:upper_limit+300, -200:-50]
    left_neighbour = find_rects_from_section(left_roi, debug=False)
    right_neighbour = find_rects_from_section(right_roi, debug=False)
    left_avg_center, _ = compute_average_center(left_neighbour)
    right_avg_center, _ = compute_average_center(right_neighbour)
    print(left_avg_center+50, leveled_raw_data.shape[1] - 200 + right_avg_center)
    # plt.figure()
    # plt.imshow(left_roi)
    # plt.figure()
    # plt.imshow(right_roi)
    # plt.show()
    return left_avg_center + 50, leveled_raw_data.shape[1] - 200 + right_avg_center


def extract_mid_stage(raw_data, xpander_coordinates, pivot_coordinates, background_height, debug=False):
    roi = raw_data[xpander_coordinates[1]+MID_STAGE_BUFFER:pivot_coordinates[0]-MID_STAGE_BUFFER,
          pivot_coordinates[2]:pivot_coordinates[3]]
    if debug:
        plt.figure()
        plt.imshow(roi)
        plt.show()
    return roi, np.nanmean(roi) - background_height


def plane_func(params, x, y):
    a, b, c = params
    return a * x + b * y + c


def error_func(params, x, y, z):
    return z - plane_func(params, x, y)


def fit_pivot_slope(pivot, debug=False):
    rows, cols = pivot.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    z = pivot
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    initial_guess = [1, 1, 1]
    result, _ = leastsq(error_func, initial_guess, args=(points[:, 0], points[:, 1], points[:, 2]))
    a, b, c = result
    xx, yy = np.meshgrid(np.linspace(0, cols, cols), np.linspace(0, rows, rows))
    zz = a * xx + b * yy + c
    pivot_h = np.mean(zz[0]) - np.mean(zz[-1])
    base_len = rows * RES
    slope_angle = abs(np.degrees(np.arctan(pivot_h / base_len)))
    cr_angle = 90 - 2 * slope_angle
    residue = zz - z
    rmse = np.sqrt(np.mean((residue) ** 2))
    if debug:
        plt.figure()
        plt.imshow(z)
        plt.colorbar()
        plt.figure()
        plt.imshow(zz)
        plt.colorbar()
        plt.figure()
        plt.imshow(residue)
        plt.colorbar()
        plt.show()
    return slope_angle, cr_angle, rmse


def extract_st_angle(pivot_only, mid_trans, top_end, bot_start, debug=False):
    try:
        copy_of_pivot = copy.deepcopy(pivot_only)
        # plt.figure()
        # plt.imshow(copy_of_pivot[bot_start + 20:top_end-10,
        #                                           mid_trans - PIVOT_TRANS_BUFFER:mid_trans + PIVOT_TRANS_BUFFER])
        # plt.show()
        reg_pivot_slope, reg_cr, pivot_rmse = fit_pivot_slope(copy_of_pivot[bot_start + 20:top_end-10,
                                                  mid_trans - PIVOT_TRANS_BUFFER:mid_trans + PIVOT_TRANS_BUFFER],
                                                  debug=debug)
        return reg_pivot_slope, reg_cr ,pivot_rmse
    except:
        return np.nan, np.nan, np.nan


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


def run_analysis(data_path, out_path, output_file_name, scan_configuration_file, pop_file_path, scan_configuration,
                 scan_state, dicing_vendor, is_st):
    main_csv_path = r"G:\Shared drives\Design\Reports\Merged Plug and BoC csv data\Merged BoC data.csv"
    main_df = pd.read_csv(main_csv_path)
    upload_to_merged_data = False
    result_list = []
    stool_list_data = []
    os.makedirs(os.path.join(out_path, "Residue Vs. Fit"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "Residue Vs. Target"), exist_ok=True)
    temp_lst = []
    residue_path_fit = os.path.join(out_path, "Residue Vs. Fit")
    residue_path_target = os.path.join(out_path, "Residue Vs. Target")
    residue_path = [residue_path_fit, residue_path_target]
    try:
        with open(pop_file_path, "r") as file:
            pop_data = json.load(file)
    except:
        pop_data = ''
    files = [f for f in os.listdir(data_path) if f.endswith(".npy")]

    for file in tqdm(files, desc="Processing files"):
        print(file)
        # if file != 'B-065-000-000-02-0Q-15.npy':
        #     continue
        master, stamp, rep, chip_name, channel_num, row, col, master_stamp, master_stamp_rep, short_id, row_col = extract_channel_name_params(
            file)
        var, var_comment = get_var_and_comment(pop_data, chip_name, int(channel_num))
        # if var not in ['B-029-000', 'B-029-000_rot']:
        #     continue
        try:
            # raw_data = np.rot90(np.load(os.path.join(data_path, file), allow_pickle=True)*-1,k=0)
            raw_data = np.rot90(np.load(os.path.join(data_path, file), allow_pickle=True),k=0)

            # raw_data = np.load(os.path.join(data_path, file), allow_pickle=True)*-1

            # plt.figure()
            # plt.imshow(raw_data)
            # plt.show()
            raw_data, background_height, theta_x, theta_y = level_larger_raw_data(raw_data, scan_configuration_file, file,
                                                                debug=False)
            leveled_raw_data, leveled_stool_section, extension = split_to_channel_stool(raw_data,
                                                                                        scan_configuration_file,
                                                                                        file, background_height)
            leveled_raw_data = utils.fix_rotation(leveled_raw_data, plug=False, debug=False)
            stool_height, _ = analyze_stool(leveled_stool_section, background_height, debug=False, is_st=is_st)
            stool_list_data.append([file + extension, stool_height])
            # continue

            file = file.replace('.npy', '')
            alignment_rects_center_list, mid_rects_center_point = find_rects_centers(leveled_raw_data,
                                                                                     background_height, debug=False)
            xpander_only, xpander_coordinates = find_xpander(leveled_raw_data, background_height, debug=False, var=var, st=is_st)
            xpander_height = find_xpander_height(xpander_only, background_height, debug=False)
            mid_x_geo = find_geo_trans_xpander_params(leveled_raw_data[xpander_coordinates[0] - 45:xpander_coordinates[0] + 5,
                                    xpander_coordinates[2]+45:xpander_coordinates[3]-45], xpander_coordinates[2] + 45,
                                                               debug=False)
            (xpander_mid_x_coordinate, xpander_mid_y_coordinate, axial_focal, trans_focal, rmse,
             xpander_geo_axial_mid) = extract_xpander_focal_length_and_mid_point(
                xpander_only, xpander_coordinates, file, residue_path, fit_radius=FIT_RADIUS_FOCAL, debug=False)
            pivot_coordinates, pivot_only = find_pivot(leveled_raw_data, background_height, xpander_coordinates[1],
                                                       xpander_coordinates[2], xpander_coordinates[3], debug=False, var=var)
            mid_stage, mid_stage_height = extract_mid_stage(leveled_raw_data, xpander_coordinates, pivot_coordinates,background_height, debug=False)
            (pb, pt, phd, pivot_trans_mid, pivot_axial_mid, pivot_optical_mid, left_mid, right_mid,
             high_section_start, low_section_start, sloped_section) = extract_pivot_height_and_mid_point_params(
                pivot_only, pivot_coordinates, debug=False, st=is_st)
            # left_side, right_side = extract_neighbouring_pivot(leveled_raw_data, xpander_coordinates[1])
            if is_st:
                # pivot_axial_focal, pivot_trans_focal = np.nan, np.nan
                pivot_angle, slope_angle, pivot_rmse = extract_st_angle(pivot_only, int(pivot_trans_mid-pivot_coordinates[2]),
                                                         high_section_start, low_section_start, debug=False)
                angle_at_channel_height = np.nan
            else:
                # pivot_axial_focal, pivot_trans_focal = extract_pivot_focal_length(pivot_only, fit_radius=25,
                #                                                                   debug=False)  # was 30
                pivot_angle, angle_at_channel_height, pivot_rmse = extract_pivot_angle(pivot_only, pt, pb,
                                                                                             stool_height=stool_height)
            optic_section = sloped_section[50:130,5:95]
            plane = create_plane_3d(pivot_angle, optic_section.shape[1], optic_section.shape[0], pt)
            # plt.figure()
            # plt.imshow(optic_section)
            # plt.show()
            # plt.figure()
            # plt.imshow(plane)
            # plt.figure()
            # plt.imshow(optic_section-plane)
            # plt.show()
            pivot_axial_focal, pivot_trans_focal = extract_pivot_focal_length(optic_section-plane, fit_radius=30)
                                       #                                                                   debug=False)  # was 30
            # plane = create_plane_3d(pivot_angle, optic_section.shape[1], optic_section.shape[0], pt)
            stool_to_mid_stage_height_difference = stool_height-mid_stage_height
            # print('####################')
            # print(angle_with_si_ref)
            # print('####################')
            pxd, pxto, to, xto, pxd_optical, pxto_geo, xto_geo = extract_channel_parameters(xpander_mid_x_coordinate,
                                                                         xpander_mid_y_coordinate,
                                                                         pivot_trans_mid, pivot_axial_mid,
                                                                         mid_rects_center_point, pivot_optical_mid,
                                                                         xpander_geo_axial_mid, mid_x_geo)
            # pitch_left = (right_mid-left_side)*RES
            # pitch_right = (right_side-left_mid)*RES
            # average_pitch = np.nanmean([pitch_left,pitch_right])
            # temp_lst.append(average_pitch)
            result_list.append([file, pb, pt, phd, pivot_axial_focal, pivot_trans_focal, pivot_angle,
                                angle_at_channel_height, axial_focal, trans_focal, pxd, pxto, to, xto, pxd_optical,
                                stool_height, xpander_mid_x_coordinate, xpander_mid_y_coordinate, pivot_rmse, rmse,
                                var, col, row, channel_num, var_comment, master, stamp, rep, master_stamp,
                                master_stamp_rep, row_col, short_id, scan_configuration, scan_state, dicing_vendor,
                                xpander_height, pxto_geo, xto_geo, theta_x, theta_y,
                                stool_to_mid_stage_height_difference, mid_stage_height])


        except:
            print('issue with ', file)
    final_data_as_df = pd.DataFrame(result_list, columns=COLUMN_LIST)
    hash0 = get_current_commit_hash()
    final_data_as_df['SW_version'] = hash0
    stool_data_as_df = pd.DataFrame(stool_list_data, columns=['Stool ID', 'Stool Height'])
    stool_data_as_df['SW_version'] = hash0
    final_data_as_df.to_csv(os.path.join(out_path, output_file_name + '.csv'), index=False)
    stool_data_as_df.to_csv(os.path.join(out_path, output_file_name + 'stools.csv'), index=False)
    if upload_to_merged_data:
        main_df = pd.concat([main_df, final_data_as_df], ignore_index=True, sort=False)
        main_df.to_csv(main_csv_path, index=False)


def ask_user_for_data():
    result = {}

    def browse_folder(entry):
        folder = filedialog.askdirectory()
        if folder:
            entry.delete(0, tk.END)
            entry.insert(0, folder)

    def browse_pop_file(entry):
        file = filedialog.askopenfilename(
            defaultextension=".pop",
            filetypes=[("POP files", "*.pop")],
            initialdir=r"G:\Shared drives\Design\Printing Database\Populations of Printed Masters"
        )
        if file:
            entry.delete(0, tk.END)
            entry.insert(0, file)

    def browse_csv_file(entry):
        file = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file:
            entry.delete(0, tk.END)
            entry.insert(0, file)

    def submit():
        result['data_path'] = data_path_entry.get()
        result['outpath'] = outpath_entry.get()
        result['output_filename'] = output_filename_entry.get()
        result['pop_file_path'] = pop_path_entry.get()
        result['scan_configuration'] = scan_config_var.get()
        result['scan_state'] = scan_state_var.get()
        result['dicing_vendor'] = dicing_vendor_var.get()
        result['config_csv'] = config_file_entry.get()
        result['master_stamp'] = master_stamp_entry.get()
        result['replication'] = replication_entry.get()
        result['is_ST'] = is_st_micro.get()
        #
        # temp_file_path = 'tempfile.csv'
        # df_result=pd.DataFrame([result])
        # df_result.to_csv(temp_file_path)
        # root.destroy()

    root = tk.Tk()
    root.title("Enter Scan Details")

    # Row tracker
    row = 0

    # # --- Load defaults from tempfile.csv into dict ---
    # try:
    #     df_defaults = pd.read_csv("tempfile.csv")
    #     # Convert the first row into a dictionary: {column_name: value}
    #     # defaults = df_defaults.iloc[0].to_dict()
    #     defaults = {}
    # except FileNotFoundError:
    #     defaults = {}
    #
    # row = 0
    #
    # def get_default(key, fallback=""):
    #     return defaults.get(key, fallback)


    # Data Path
    tk.Label(root, text="Data Path:").grid(row=row, column=0, sticky='w')
    data_path_entry = tk.Entry(root, width=50)
    # data_path_entry.insert(0, get_default("data_path"))
    data_path_entry.grid(row=row, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_folder(data_path_entry)).grid(row=row, column=2)
    row += 1

    # Output Path
    tk.Label(root, text="Output Path:").grid(row=row, column=0, sticky='w')
    outpath_entry = tk.Entry(root, width=50)
    # outpath_entry.insert(0, get_default("outpath"))
    outpath_entry.grid(row=row, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_folder(outpath_entry)).grid(row=row, column=2)
    row += 1

    # Output File Name
    tk.Label(root, text="Output File Name:").grid(row=row, column=0, sticky='w')
    output_filename_entry = tk.Entry(root, width=50)
    # output_filename_entry.insert(0, get_default("output_filename"))
    output_filename_entry.grid(row=row, column=1)
    row += 1

    # POP File Path
    tk.Label(root, text="POP File Path:").grid(row=row, column=0, sticky='w')
    pop_path_entry = tk.Entry(root, width=50)
    # pop_path_entry.insert(0, get_default("pop_file_path"))
    pop_path_entry.grid(row=row, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_pop_file(pop_path_entry)).grid(row=row, column=2)
    row += 1

    # Configuration CSV File
    tk.Label(root, text="Configuration CSV File:").grid(row=row, column=0, sticky='w')
    config_file_entry = tk.Entry(root, width=50)
    # config_file_entry.insert(0, get_default("config_csv"))
    config_file_entry.grid(row=row, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_csv_file(config_file_entry)).grid(row=row, column=2)
    row += 1

    # Master-Stamp (Free text)
    tk.Label(root, text="Master-Stamp:").grid(row=row, column=0, sticky='w')
    master_stamp_entry = tk.Entry(root, width=50)
    # master_stamp_entry.insert(0, get_default("master_stamp"))
    master_stamp_entry.grid(row=row, column=1)
    row += 1

    # Replication (Free text)
    tk.Label(root, text="Replication:").grid(row=row, column=0, sticky='w')
    replication_entry = tk.Entry(root, width=50)
    # replication_entry.insert(0, str(get_default("replication")))
    replication_entry.grid(row=row, column=1)
    row += 1

    # Dropdowns
    scan_config_options = ["wafer level", "chip level"]
    scan_state_options = ["NA", "original", "post stamp 1", "post stamp 2", "post stamp 3", "post stamp 4", "post stamp 5"]
    dicing_vendor_options = ["NA", "ADT", "Luigi"]


    tk.Label(root, text="Scan Configuration:").grid(row=row, column=0, sticky='w')
    scan_config_var = tk.StringVar(value=scan_config_options[0])
    ttk.Combobox(root, textvariable=scan_config_var, values=scan_config_options, state='readonly').grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Scan State:").grid(row=row, column=0, sticky='w')
    scan_state_var = tk.StringVar(value=scan_state_options[0])
    ttk.Combobox(root, textvariable=scan_state_var, values=scan_state_options, state='readonly').grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Dicing Vendor:").grid(row=row, column=0, sticky='w')
    dicing_vendor_var = tk.StringVar(value=dicing_vendor_options[0])
    ttk.Combobox(root, textvariable=dicing_vendor_var, values=dicing_vendor_options, state='readonly').grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Is the wafer an ST micro:").grid(row=row, column=0, sticky='w')
    is_st_micro = tk.BooleanVar(value=False)  # or tk.IntVar(value=0) if you prefer
    tk.Checkbutton(root, variable=is_st_micro).grid(row=row, column=1, sticky='w')
    row += 1

    # Submit button
    tk.Button(root, text="Submit", command=submit).grid(row=row, column=1, pady=10)

    root.mainloop()

    return (
        result['data_path'],
        result['outpath'],
        result['output_filename'],
        result['pop_file_path'],
        result['scan_configuration'],
        result['scan_state'],
        result['dicing_vendor'],
        result['config_csv'],
        result['master_stamp'],
        result['replication'],
        result['is_ST']
    )


def main():
    local_debug = True
    if local_debug:
        data_path = r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\master\CSV FIles after stamp 4'
        outpath = r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\master\CSV FIles after stamp 4\analysis'
        output_file_name = 'B-065-000-000'
        pop_file_path = r"G:\Shared drives\Design\Printing Database\Populations of Printed Masters\B-065 Dagoba M4R.pop"
        # pop_file_path = r"G:\Shared drives\Design\Printing Database\Populations of Printed Masters\B-053 M4R Dagoba.pop"
        scan_configuration = 'wafer level'
        scan_state = 'post stamp 4'
        dicing_vendor = ''
        config_file = r""
        master_stamp = 'B-065-000'
        rep = '000'
        is_st = False
    else:
        (data_path, outpath, output_file_name, pop_file_path, scan_configuration, scan_state, dicing_vendor,
         config_file, master_stamp, rep, is_st) = ask_user_for_data()
    try:
        config_df = pd.read_csv(config_file)
        config_df = add_relevant_id(config_df, master_stamp, rep)
    except:
        config_df = ''
    run_analysis(data_path, outpath, output_file_name, config_df, pop_file_path, scan_configuration, scan_state,
                 dicing_vendor, is_st)


if __name__ == '__main__':
    main()

# Todo improve Xpander detection - Machine learning
