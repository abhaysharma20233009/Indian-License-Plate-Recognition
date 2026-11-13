import cv2
import numpy as np

import processing.config as config


def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    plate_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            # Relaxed width/height constraints for varying image distances
            if 100 <= w <= 4000 and 30 <= h <= 1000:
                plate_contour = approx
                break
        else:
            continue  # Continue searching remaining contours

    if plate_contour is None:
        return None

    x, y, w, h = cv2.boundingRect(plate_contour)
    img_h, img_w = image.shape[:2]
    ymin, ymax = max(0, y - 5), min(img_h, y + h + 5)
    xmin, xmax = max(0, x - 5), min(img_w, x + w + 5)
    
    return image[ymin:ymax, xmin:xmax]


def sort_polygon_corners(coordinates):
    """
    Sort the corners of a polygon.

    Args:
    coordinates (list): A list of coordinates representing the corners of a polygon.

    Returns:
    list: A list of coordinates sorted in a clockwise order.
    """

    sorted_by_x = sorted(coordinates, key=lambda coord: coord[0])

    bottom_half = sorted_by_x[:2]
    top_half = sorted_by_x[2:]

    bottom_half.sort(key=lambda coord: coord[1])
    top_half.sort(key=lambda coord: coord[1], reverse=True)

    sorted_coordinates = bottom_half + top_half

    return sorted_coordinates


def expand_polygon(corners, size):
    """
    Expand the corners of a polygon.

    Args:
    corners (list): A list of coordinates representing the corners of a polygon.
    size (int): The size to expand the corners by.

    Returns:
    numpy.ndarray: An array of coordinates representing the expanded corners of a polygon.
    """
    corners = np.array(corners, dtype=np.float32)
    center = np.mean(corners, axis=0)
    expanded_corners = []
    for corner in corners:
        vector = corner - center
        vector_normalized = vector / np.linalg.norm(vector)
        expanded_corner = corner + vector_normalized * size
        expanded_corners.append(expanded_corner)

    return np.array(expanded_corners, dtype=np.float32)


def cut_sides(image, x):
    """
    Cuts the left and right sides of an image by x pixels.

    Parameters:
    image (numpy.ndarray): The input image.
    x (int): The number of pixels to cut from both the left and right sides.

    Returns:
    numpy.ndarray: The cropped image.
    """
    _, width = image.shape[:2]
    x = min(x, width // 2)
    new_width = width - 2 * x
    cropped_image = image[:, x : x + new_width]

    return cropped_image
