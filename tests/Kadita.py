import os
import cv2
from kadita import DeepFace
class FaceCapture:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.captured_frame = None

    def capture_face(self, name):
        # Create a window to display the frame
        cv2.namedWindow("Capture Face", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capture Face", 640, 480)

        # Open the webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Capture Face", frame)
            key = cv2.waitKey(1)

            if key == ord('s') or key == ord('S'):
                # Save the captured frame
                self.captured_frame = frame
                break
            elif key == ord('q') or key == ord('Q'):
                break

        # Release the webcam and destroy the window
        cap.release()
        cv2.destroyAllWindows()

        # If a frame was captured, save it
        if self.captured_frame is not None:
            save_path = os.path.join(self.dataset_path, name + '.jpg')
            cv2.imwrite(save_path, self.captured_frame)
            print(f"Image saved as {save_path}")
        else:
            print("No image captured.")

    def delete_representations(self):
        representations_file = os.path.join(self.dataset_path, 'representations_vgg_face.pkl')
        if os.path.exists(representations_file):
            os.remove(representations_file)
            print(f"File {representations_file} deleted.")
        else:
            print("File not found.")

    def delete_all_jpg_files(self):
        jpg_files = [f for f in os.listdir(self.dataset_path) if f.endswith(".jpg")]
        for jpg_file in jpg_files:
            file_path = os.path.join(self.dataset_path, jpg_file)
            os.remove(file_path)
        print(f"All JPG files deleted from {self.dataset_path}.")

    def start_stream(self, detector_backend):
        DeepFace.stream(self.dataset_path, detector_backend=detector_backend)
