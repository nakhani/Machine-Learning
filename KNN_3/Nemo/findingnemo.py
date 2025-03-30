import numpy as np
import matplotlib.pyplot as plt
import cv2
from knn import KNN


class FindingNemo:
    def __init__(self, train_image):
        
        self.knn = KNN(k=3)
        X_train, Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(X_train, Y_train)

    def convert_image_to_dataset(self, image):
    
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)
        mask_orange = cv2.inRange(image_hsv, light_orange, dark_orange)

        light_white = (0, 0, 200)
        dark_white = (145, 60, 255)
        mask_white = cv2.inRange(image_hsv, light_white, dark_white)

       
        light_black = (0, 0, 0)
        dark_black = (185, 255, 8)
        mask_black = cv2.inRange(image_hsv, light_black, dark_black)

       
        plt.imshow(mask_black, cmap='gray')
        plt.title("Black Mask")
        plt.show()

       
        final_mask = mask_orange + mask_white + mask_black

        
        X_train = image_hsv.reshape(-1, 3) / 255
        Y_train = final_mask.reshape(-1) // 255
        return X_train, Y_train

    def remove_background(self, test_image):
       
    
        test_image = cv2.resize(test_image, (0, 0), fx=0.25, fy=0.25)
        test_image_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

       
        X_test = test_image_hsv.reshape(-1, 3) / 255
        Y_pred = self.knn.predict(X_test)

       
        output_mask = Y_pred.reshape(test_image.shape[:2]).astype('uint8')

        
        final_result = cv2.bitwise_and(test_image, test_image, mask=output_mask)
        return final_result, output_mask

    def visualize_results(self, image, mask, title="Result"):
     
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (Image)")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.title(f"{title} (Mask)")
        plt.show()



if __name__ == "__main__":
   
    nemo = cv2.imread("nemo.jpg")
    nemo_finder = FindingNemo(train_image=nemo)


    dashe_nemo = cv2.imread("dashe-nemo.jpg")
    final_result, output_mask = nemo_finder.remove_background(dashe_nemo)

    nemo_finder.visualize_results(final_result, output_mask, title="Dashe Nemo")
