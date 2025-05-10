import cv2
import os

class ImageProcessor:
    def __init__(self, image=None, path=None):
        self.image = image
        self.path = path
        if path:
            self.image = self.load_image(path)
        
    def load_image(self, path):
        """Load an image from the specified path"""
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Image not found or cannot be opened: {path}")
        self.image = image
        self.path = path
        return image
        
    def validate_image(self, image):
        """Validate image before processing"""
        if image is None:
            raise ValueError("No image provided")
        if len(image.shape) not in [2, 3]:
            raise ValueError("Unsupported image dimensions")
        if image.size == 0:
            raise ValueError("Empty image")
        return True

    def denoise_image(self, image=None):
        """Denoise the image using appropriate methods based on image type"""
        if image is None:
            image = self.image
            
        self.validate_image(image)
            
        try:
            if len(image.shape) == 2:  # Grayscale
                # Ensure odd kernel size
                ksize = 3
                median = cv2.medianBlur(image, ksize)
                denoised = cv2.fastNlMeansDenoising(
                    median, None, h=10, 
                    templateWindowSize=7, 
                    searchWindowSize=21
                )
            elif len(image.shape) == 3 and image.shape[2] == 3:  # Color
                # Ensure odd kernel size
                ksize = 3  # Reduced from 6 to ensure it's odd
                median = cv2.medianBlur(image, ksize)
                denoised = cv2.fastNlMeansDenoisingColored(
                    median, None, h=10, hColor=10,
                    templateWindowSize=7, 
                    searchWindowSize=21
                )
            else:
                raise ValueError("Unsupported image format")
            
            self.image = denoised
            return denoised
            
        except cv2.error as e:
            raise ValueError(f"OpenCV processing error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error denoising image: {str(e)}")
        
    def save_image(self, path=None):
        """Save the image to the specified path"""
        if path is None:
            path = self.path
        if path is None:
            raise ValueError("No path provided for saving")
        if self.image is None:
            raise ValueError("No image to save")
            
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(path, self.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == '.png':
            cv2.imwrite(path, self.image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            cv2.imwrite(path, self.image)
        print(f"Saved: {path}")