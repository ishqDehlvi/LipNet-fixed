import logging
import os
import sys
import numpy as np
import cv2
import dlib
from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras import backend as K

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
FACE_PREDICTOR_PATH = "/workspace/LipNet/common/predictors/shape_predictor_68_face_landmarks.dat"
PREDICT_DICTIONARY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lipnet', 'dictionaries', 'grid.txt')
PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200

def get_video_frames(video_path):
    """Function to read video frames from a given video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.array(frames)

def process_frames_face(frames):
    """Process video frames for face detection."""
    detected_faces = []
    detector = dlib.get_frontal_face_detector()
    
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            detected_faces.append(frame)
        else:
            logging.debug(f"Frame {i}: No faces detected")
    
    if len(detected_faces) == 0:
        raise ValueError("No faces detected in any frame")
        
    return np.array(detected_faces)

def predict(weight_path, video_path, absolute_max_string_len=32, output_size=28):
    logging.info("\nLoading data from disk...")
    
    try:
        # Check if files exist
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(FACE_PREDICTOR_PATH):
            raise FileNotFoundError(f"Face predictor file not found: {FACE_PREDICTOR_PATH}")
        
        # Load video data
        logging.info("Creating video processor...")
        video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
        
        logging.info("Reading video frames...")
        frames = get_video_frames(video_path)
        logging.debug(f"Total frames read: {frames.shape[0]}")
        
        logging.info("Processing frames...")
        video.data = process_frames_face(frames)
        
        if video.data is None or video.data.shape[0] == 0:
            raise ValueError("No valid frames were processed")
        
        logging.info("Data loaded.\n")
        
        if K.image_data_format() == 'channels_first':
            img_c, frames_n, img_w, img_h = video.data.shape
        else:
            frames_n, img_w, img_h, img_c = video.data.shape
            
        logging.debug(f"Video shape: {video.data.shape}")

        # Match architecture with weights
        lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                        absolute_max_string_len=absolute_max_string_len, output_size=output_size)

        try:
            adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        except TypeError:
            adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        
        logging.info("Loading weights...")
        try:
            lipnet.model.load_weights(weight_path)
        except ValueError as e:
            logging.error("Weight file does not match the model architecture. Verify model configuration.")
            raise e

        logging.info("Weights loaded successfully")

        spell = Spell(path=PREDICT_DICTIONARY)
        decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                         postprocessors=[labels_to_text, spell.sentence])

        logging.info("Preparing data for prediction...")
        X_data = np.array([video.data]).astype(np.float32) / 255
        input_length = np.array([len(video.data)])

        logging.info("Running prediction...")
        y_pred = lipnet.predict(X_data)
        result = decoder.decode(y_pred, input_length)[0]
        logging.info("Prediction completed")

        return (video, result)
        
    except Exception as e:
        logging.error(f"\nError during prediction: {str(e)}")
        raise

# Main execution
if __name__ == '__main__':
    try:
        if len(sys.argv) < 3:
            logging.error("Usage: python predict.py <weight_path> <video_path> [max_string_len] [output_size]")
            sys.exit(1)
        
        if len(sys.argv) == 3:
            video, result = predict(sys.argv[1], sys.argv[2])
        elif len(sys.argv) == 4:
            video, result = predict(sys.argv[1], sys.argv[2], int(sys.argv[3]))
        elif len(sys.argv) == 5:
            video, result = predict(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        else:
            video, result = None, ""

        if video is not None:
            show_video_subtitle(video.face, result)

        stripe = "-" * len(result)
        logging.info("\n")
        logging.info(f"Prediction Result: {result}")
        logging.info(f"{stripe}")

    except Exception as e:
        logging.error(f"\nError: {str(e)}")
        sys.exit(1)
