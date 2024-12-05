import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image from the file system
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded successfully
if img is None:
    print("Error: Could not load image")
    exit()

# Global variables for mouse event handling
dragging = False
square_position = (100, 100)  # Initial position of the square
square_size = 50  # Size of the square
prev_square_position = square_position  # Keep track of the previous square position

# Mouse callback function for dragging the square
def mouse_callback(event, x, y, flags, param):
    global dragging, square_position
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging the square if clicked inside it
        if (square_position[0] < x < square_position[0] + square_size) and (square_position[1] < y < square_position[1] + square_size):
            dragging = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            # Update the square's position while dragging
            square_position = (x - square_size // 2, y - square_size // 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# Function to compute the cross-correlation using DFT
def cross_correlation_dft(img, template, square_position):
    # Extract the region of interest (the square) from the image
    x, y = square_position
    roi = img[y:y + square_size, x:x + square_size]
    
    # Convert to single-channel (grayscale) float32 for DFT (CV_32FC1)
    roi = np.float32(roi)
    template = np.float32(template)
    
    # Compute the DFT of the template (square) and the ROI
    dft_roi = cv2.dft(roi, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_template = cv2.dft(template, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Cross-correlation using frequency domain: Multiply DFTs and take inverse DFT
    dft_roi_conj = cv2.split(dft_roi)[0] + 1j * cv2.split(dft_roi)[1]  # Real + Imaginary parts of DFT
    dft_template_conj = cv2.split(dft_template)[0] + 1j * cv2.split(dft_template)[1]  # Real + Imaginary parts of DFT

    # Cross-correlation = DFT(ROI) * DFT(Template)*
    cross_correlation = dft_roi_conj * dft_template_conj
    correlation_result = cv2.idft(np.array([cross_correlation.real, cross_correlation.imag]).transpose(1, 2, 0), flags=cv2.DFT_SCALE)
    
    # Calculate magnitude (real part) of correlation result
    magnitude = np.sqrt(correlation_result[:, :, 0]**2 + correlation_result[:, :, 1]**2)
    
    # Normalize the magnitude for better visualization
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    # Return the magnitude of the frequency response for display
    return magnitude

# Initialize the template (the first square region)
template = img[square_position[1]:square_position[1] + square_size, square_position[0]:square_position[0] + square_size]
template = np.float32(template)  # Convert template to float32

# Set up the window and mouse callback
cv2.namedWindow('Manual Square Dragging with Cross-Correlation')
cv2.setMouseCallback('Manual Square Dragging with Cross-Correlation', mouse_callback)

# Create a figure for plotting the frequency response
 # Turn on interactive mode for real-time plotting
fig, ax = plt.subplots()
# Main loop
while True:
    img_copy = img.copy()

    # Track the square's position using cross-correlation in the frequency domain (DFT)
    magnitude = cross_correlation_dft(img_copy, template, square_position)
    
    # Update square's position with the result from cross-correlation (this ensures square's position is verified)
    # square_position = new_position  # We no longer need this since it's only for visualizing the correlation

    # Draw the square on the image
    cv2.rectangle(img_copy, square_position, (square_position[0] + square_size, square_position[1] + square_size), (0, 255, 0), 2)
    
    # Calculate the change in coordinates
    delta_x = square_position[0] - prev_square_position[0]
    delta_y = square_position[1] - prev_square_position[1]

    # Display the changes in x and y coordinates on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"x shift: {delta_x}  y shift: {-delta_y}"
    cv2.putText(img_copy, text, (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Update previous position for the next frame
    prev_square_position = square_position

    # Display the image with the moving square
    cv2.imshow('Manual Square Dragging with Cross-Correlation', img_copy)

    # Plot the frequency response (magnitude of the cross-correlation DFT)
    ax.clear()  # Clear the previous plot
    ax.imshow(magnitude, cmap='gray')  # Plot the magnitude
    ax.set_title('Frequency Response (Cross-Correlation DFT)')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    plt.draw()  # Redraw the plot
    plt.pause(0.1)  # Pause to allow real-time plotting

    # Wait for key press and exit if 'q' is pressed
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
plt.close()
