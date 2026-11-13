
def segment_characters(license_plate):
    """
    Segment individual characters from an Indian license plate.

    Uses Otsu thresholding and connected component analysis
    instead of Canny edge contours.
    """

    # Ensure grayscale image
    if len(license_plate.shape) == 3:
        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = license_plate.copy()

    plate_height, plate_width = gray.shape

    # Convert dark characters on a light background to white foreground
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Handle plates with light characters on a dark background
    if np.count_nonzero(binary) / binary.size > 0.5:
        binary = cv2.bitwise_not(binary)

    # Remove small gaps in character strokes
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find connected components
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    best_components = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        # Relative dimensions
        width_ratio = w / plate_width
        height_ratio = h / plate_height
        area_ratio = area / (plate_width * plate_height)

        if (
            config.OCR_CHARACTER_WIDTH_RATIO_MIN <= width_ratio
            <= config.OCR_CHARACTER_WIDTH_RATIO_MAX
            and config.OCR_CHARACTER_HEIGHT_RATIO_MIN <= height_ratio
            <= config.OCR_CHARACTER_HEIGHT_RATIO_MAX
            and area_ratio >= config.OCR_CHARACTER_AREA_RATIO_MIN
        ):
            best_components.append((x, y, w, h))

    # Sort characters from left to right
    best_components.sort(key=lambda box: box[0])

    characters = []

    for x, y, w, h in best_components:
        character = gray[y:y + h, x:x + w]
        characters.append(character)

    if config.DEBUG:
        debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for x, y, w, h in best_components:
            cv2.rectangle(
                debug_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        display_image("Segmented Characters", debug_image, True)
        display_image("Binary Plate", binary, True)

        print("Characters segmented:", len(characters))
        print("Character dimensions:", best_components)

    return characters
