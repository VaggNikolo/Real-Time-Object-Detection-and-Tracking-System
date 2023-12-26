import cv2

# Test image display
image = cv2.imread('/home/vangelis/Downloads/Untitled.jpg')  # Replace with the path to an actual image file
cv2.imshow('Test Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
