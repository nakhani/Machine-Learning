import numpy as np
import matplotlib.pyplot as plt
import cv2
from knn import KNN
from sklearn.metrics import confusion_matrix
import seaborn as sns  

class FindingDory:
    def __init__(self, train_image, resize_factor=0.25):
        
        self.knn = KNN(k=3)
        self.resize_factor = resize_factor

        
        train_image_resized = self.resize_image(train_image)
        
        
        X_train, Y_train = self.convert_image_to_dataset(train_image_resized)
        self.knn.fit(X_train, Y_train)
        self.Y_train = Y_train 

        
        self.display_final_result(train_image_resized)

    def resize_image(self, image):
        
        return cv2.resize(image, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

    def convert_image_to_dataset(self, image):
        
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        
        light_blue = (90, 50, 70)
        dark_blue = (128, 255, 255)
        mask_blue = cv2.inRange(image_hsv, light_blue, dark_blue)

        light_yellow = (20, 100, 100)
        dark_yellow = (30, 255, 255)
        mask_yellow = cv2.inRange(image_hsv, light_yellow, dark_yellow)

        
        final_mask = mask_blue + mask_yellow

        
        plt.imshow(final_mask, cmap='gray')
        plt.title("Final Mask")
        plt.axis('off')
        plt.show()

        
        self.final_mask = final_mask
        self.image_hsv = image_hsv

        
        X_train = image_hsv.reshape(-1, 3) / 255
        Y_train = final_mask.reshape(-1) // 255
        return X_train, Y_train

    def remove_background(self, test_image):
       
  
        test_image_resized = self.resize_image(test_image)

       
        test_image_hsv = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2HSV)

        
        X_test = test_image_hsv.reshape(-1, 3) / 255
        Y_pred = self.knn.predict(X_test)

        output_mask = Y_pred.reshape(test_image_resized.shape[:2]).astype('uint8')

        final_result = cv2.bitwise_and(test_image_resized, test_image_resized, mask=output_mask)
        return final_result, output_mask, Y_pred

    def display_final_result(self, image):
      

        final_result = cv2.bitwise_and(image, image, mask=self.final_mask)
        plt.axis('off')  
        plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
        plt.title("Final Result")
        plt.show()

    def plot_confusion_matrix(self, Y_test, Y_pred):
  

        conf_matrix = confusion_matrix(Y_test, Y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Background", "Dory"], yticklabels=["Background", "Dory"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def visualize_results(self, image, mask, title="Result"):
       

        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (Image)")
        plt.axis('off')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.title(f"{title} (Mask)")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    dory_train = cv2.imread("dory.jpg")


    dory_finder = FindingDory(train_image=dory_train)


    dory_test = cv2.imread("dory-test.jpg")

    final_result, output_mask, Y_pred = dory_finder.remove_background(dory_test)

    Y_test = output_mask.reshape(-1)

    dory_finder.plot_confusion_matrix(Y_test, Y_pred)

    dory_finder.visualize_results(final_result, output_mask, title="Dory Detection")
