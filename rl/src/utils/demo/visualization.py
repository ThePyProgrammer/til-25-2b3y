import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize_proba_density(proba_density, output_dir=None):
    """Normalize probability density for visualization and return BGR image with consistent size.

    Args:
        proba_density: 2D numpy array with probability density values
        output_dir: Directory where outputs are saved (for future extensions)

    Returns:
        BGR image with normalized probability density visualization
    """
    # Define a consistent output size
    output_size = (600, 600)

    if proba_density is None or np.max(proba_density) == 0:
        return np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

    # Normalize the probability density to [0, 1]
    normalized = proba_density / np.max(proba_density)

    # Create a colormap
    cmap = plt.get_cmap("hot")

    # Apply colormap to the normalized data (returns RGBA)
    colored_data = cmap(normalized)

    # Convert from RGBA to RGB and then to 8-bit
    rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)

    # Resize to consistent dimensions
    resized_bgr = cv2.resize(bgr_data, output_size, interpolation=cv2.INTER_NEAREST)

    # put the raw probability numbers on top of the image
    for i in range(16):
        for j in range(16):
            # Get the probability value at this pixel
            prob_value = proba_density[i, j]
            if prob_value > 0:
                # Convert to string and put it on the image
                text = f"{prob_value:.2f}"
                cv2.putText(
                    resized_bgr,
                    text,
                    (
                        output_size[1] * j // 16,
                        output_size[0] * (i + 1) // 16,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )

    # Draw a border around the image to make it more visible
    border_size = 2
    cv2.rectangle(
        resized_bgr,
        (0, 0),
        (output_size[0] - 1, output_size[1] - 1),
        (255, 255, 255),
        border_size,
    )

    return resized_bgr


def resize_preserve_aspect_ratio(image, target_height):
    """Resize image to target height while preserving aspect ratio.

    Args:
        image: Input image (numpy array)
        target_height: Desired height for the output image

    Returns:
        Resized image with preserved aspect ratio
    """
    h, w = image.shape[:2]
    aspect = w / h
    target_width = int(target_height * aspect)
    return cv2.resize(image, (target_width, target_height))


def combine_views(views, labels):
    """Combine multiple views horizontally with labels, preserving aspect ratio.

    Args:
        views: List of image arrays to combine
        labels: List of labels for each view

    Returns:
        Combined image with all views side by side
    """
    # Convert all views to BGR if they're in RGB
    processed_views = []
    for view in views:
        if len(view.shape) == 3 and view.shape[2] == 3:
            # Check if it's in RGB format (a heuristic)
            processed_view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        else:
            processed_view = view
        processed_views.append(processed_view)

    # Find common height (use the smallest height)
    target_height = max(view.shape[0] for view in processed_views)

    # Resize all views to have the same height while preserving aspect ratio
    resized_views = [
        resize_preserve_aspect_ratio(view, target_height) for view in processed_views
    ]

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    labeled_views = []
    for i, view in enumerate(resized_views):
        labeled_view = view.copy()
        cv2.putText(
            labeled_view, labels[i], (10, 20), font, font_scale, color, thickness
        )
        labeled_views.append(labeled_view)

    # Combine horizontally
    combined = np.hstack(labeled_views)
    return combined


def combine_views_grid(agent_data, oracle_view):
    """
    Combine views into a layout with:
    - Oracle view in the center (2 wide, 2 tall)
    - Agent 0 and 2 stacked on the left (each with map and probability side by side)
    - Agent 1 and 3 stacked on the right (each with map and probability side by side)

    Args:
        agent_data: Dictionary mapping agent indices (0-3) to their respective data
                   (map and probability density views)
        oracle_view: The global oracle view of the environment

    Returns:
        Combined frame with oracle view in center and agent views in quadrants
    """
    map_size = (400, 400)  # Size for map/probability views
    oracle_size = (800, 800)  # Size for oracle view (2x2)

    oracle_view = cv2.cvtColor(oracle_view, cv2.COLOR_RGB2BGR)

    # Add label to oracle view
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 1

    # Process agent views (map and probability side by side)
    agent_rows = {}  # Will hold rows for agents 0&2 and 1&3

    for agent_idx in range(4):
        # Get data for this agent
        data = agent_data.get(agent_idx, {})
        map_view = data.get("map")
        proba_view = data.get("proba")

        # Create blank images if views are missing
        if map_view is None:
            map_view = np.zeros((400, 400, 3), dtype=np.uint8)
        if proba_view is None:
            proba_view = np.zeros((400, 400, 3), dtype=np.uint8)

        # Convert to BGR if in RGB format
        map_view = cv2.cvtColor(map_view, cv2.COLOR_RGB2BGR)
        proba_view = cv2.cvtColor(proba_view, cv2.COLOR_RGB2BGR)

        # Resize views to consistent size
        map_view_resized = cv2.resize(
            map_view, map_size, interpolation=cv2.INTER_NEAREST
        )
        proba_view_resized = cv2.resize(
            proba_view, map_size, interpolation=cv2.INTER_NEAREST
        )

        # Add labels to views
        cv2.putText(
            map_view_resized,
            f"Agent {agent_idx} - Map",
            (10, 30),
            font,
            font_scale * 0.8,
            color,
            thickness,
        )
        cv2.putText(
            proba_view_resized,
            f"Agent {agent_idx} - Probability",
            (10, 30),
            font,
            font_scale * 0.8,
            color,
            thickness,
        )

        # Put map and probability side by side
        agent_row = np.hstack((map_view_resized, proba_view_resized))

        # Group by column (agents 0,2 on left, agents 1,3 on right)
        col = agent_idx % 2  # 0 for left, 1 for right
        row_key = f"col_{col}"

        if row_key not in agent_rows:
            agent_rows[row_key] = []
        agent_rows[row_key].append(agent_row)

    # Stack agents 0&2 vertically, and 1&3 vertically
    left_column = np.vstack(
        agent_rows.get("col_0", [np.zeros((800, 800, 3), dtype=np.uint8)])
    )
    right_column = np.vstack(
        agent_rows.get("col_1", [np.zeros((800, 800, 3), dtype=np.uint8)])
    )

    # Ensure columns are the correct height (same as oracle)
    if left_column.shape[0] != oracle_size[1]:
        padding = np.zeros(
            (oracle_size[1] - left_column.shape[0], left_column.shape[1], 3),
            dtype=np.uint8,
        )
        left_column = np.vstack([left_column, padding])
    if right_column.shape[0] != oracle_size[1]:
        padding = np.zeros(
            (oracle_size[1] - right_column.shape[0], right_column.shape[1], 3),
            dtype=np.uint8,
        )
        right_column = np.vstack([right_column, padding])

    h, w = oracle_view.shape[:2]
    aspect = w / h
    target_height = right_column.shape[0]
    target_width = int(target_height * aspect)

    oracle_view_resized = cv2.resize(
        oracle_view, (target_width, target_height), interpolation=cv2.INTER_NEAREST
    )

    cv2.putText(
        oracle_view_resized, "Oracle View", (20, 40), font, font_scale, color, thickness
    )

    combined = np.hstack((left_column, oracle_view_resized, right_column))

    return combined
