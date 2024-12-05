import cv2
import numpy as np

# Globals for the moving square
square_size = 50
square_pos_x, square_pos_y = 100, 100
last_square_pos_x, last_square_pos_y = square_pos_x, square_pos_y
shift_x, shift_y = 0, 0
dragging = False

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to move the square.
    """
    global dragging, square_pos_x, square_pos_y, last_square_pos_x, last_square_pos_y, shift_x, shift_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Update square position
        shift_x, shift_y = x - square_pos_x, y - square_pos_y
        square_pos_x, square_pos_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False
        last_square_pos_x, last_square_pos_y = square_pos_x, square_pos_y
        shift_x, shift_y = 0, 0

def detect_motion():
    """
    Detect motion by calculating the displacement of the square.
    """
    global last_square_pos_x, last_square_pos_y, square_pos_x, square_pos_y
    shift_x = square_pos_x - last_square_pos_x
    shift_y = square_pos_y - last_square_pos_y
    return shift_x, shift_y

def main():
    global square_pos_x, square_pos_y, last_square_pos_x, last_square_pos_y

    # Load surface texture
    surface = cv2.imread("surface_texture.jpg", cv2.IMREAD_GRAYSCALE)
    if surface is None:
        raise FileNotFoundError("Please ensure 'surface_texture.jpg' is in the working directory.")
    
    height, width = surface.shape[:2]
    window_name = "Optical Mouse Simulation"

    # Create a window and set the mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click and drag the square to simulate motion.")

    while True:
        # Create a copy of the surface with the square
        display_surface = surface.copy()
        cv2.rectangle(
            display_surface,
            (square_pos_x, square_pos_y),
            (square_pos_x + square_size, square_pos_y + square_size),
            (255, 255, 255),  # White square
            -1  # Fill the square
        )

        # Detect motion
        shift_x, shift_y = detect_motion()

        # Display the detected shift
        display = display_surface.copy()
        cv2.putText(display, f"Shift X: {shift_x:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        cv2.putText(display, f"Shift Y: {shift_y:.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

        cv2.imshow(window_name, display)

        # Exit on pressing 'q'
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
