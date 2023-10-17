import argparse
import os
from Kadita import FaceCapture


def main():
    parser = argparse.ArgumentParser(description='Face Recognition Project DeepFace')

    parser.add_argument('--capture', action='store_true', help='Capture and save a face image')
    parser.add_argument('--name', type=str, help='Name of the person (used when capturing)')
    parser.add_argument('--delete-file', type=str, help='Delete a specific file by name (including extension)')
    parser.add_argument('--delete-training', action='store_true', help='Delete representations_vgg_face.pkl')
    parser.add_argument('--delete-all-files', action='store_true', help='Delete all image files in dataset folder')
    parser.add_argument('--detector-backend', type=str, default='opencv',
                        choices=['opencv', 'ssd', 'mtcnn', 'dlib', 'retinaface'],
                        help='Detector backend for face recognition')

    args = parser.parse_args()

    dataset_path = "dataset"
    face_capture = FaceCapture(dataset_path)

    if args.capture:
        if args.name:
            face_capture.capture_face(args.name)
        else:
            print("Please provide a name for the captured face (--name).")

    if args.delete_training:
        face_capture.delete_representations()

    if args.delete_file:
        file_to_delete = args.delete_file
        file_path = os.path.join(dataset_path, file_to_delete)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_to_delete} deleted.")
        else:
            print(f"File {file_to_delete} not found.")

    if args.delete_all_files:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # Tambahkan ekstensi yang diizinkan di sini
        for file in os.listdir(dataset_path):
            if file.lower().endswith(tuple(image_extensions)):
                file_path = os.path.join(dataset_path, file)
                os.remove(file_path)
        print("All image files deleted from the dataset folder.")

    if not args.capture and not args.delete_training and not args.delete_file and not args.delete_all_files:
        face_capture.start_stream(args.detector_backend)


if __name__ == '__main__':
    main()
