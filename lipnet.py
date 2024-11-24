import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, Dense, Dropout, Flatten
from tensorflow.keras.backend import ctc_decode

class LipNet:
    def __init__(self, frames_n=75, img_w=100, img_h=50, img_c=3, output_size=28):
        """
        Initialize the LipNet model.
        Args:
            frames_n (int): Number of video frames.
            img_w (int): Width of each frame.
            img_h (int): Height of each frame.
            img_c (int): Number of color channels in each frame.
            output_size (int): Size of the output vocabulary.
        """
        self.frames_n = frames_n
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c
        self.output_size = output_size
        self.model = self.build_model()

    def build_model(self):
        """
        Build the LipNet model.
        Returns:
            Model: Compiled Keras model.
        """
        input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)
        inputs = Input(shape=input_shape)

        # 3D Convolutional Layers
        x = Conv3D(32, kernel_size=(3, 5, 5), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(64, kernel_size=(3, 5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        x = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)

        # Fully Connected Layers
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.output_size, activation='softmax')(x)

        model = Model(inputs, x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_weights(self, weights_path):
        """
        Load pretrained weights into the LipNet model.
        Args:
            weights_path (str): Path to the .h5 weight file.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weight file not found at {weights_path}")
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")

    def predict(self, video_path):
        """
        Predicts the lipreading output for a given video.
        Args:
            video_path (str): Path to the video file.
        Returns:
            np.array: Predicted character probabilities.
        """
        input_data = self.preprocess_video(video_path)
        prediction = self.model.predict([input_data])
        return prediction

    def preprocess_video(self, video_path):
        """
        Preprocess video into a format compatible with LipNet.
        Args:
            video_path (str): Path to the video file.
        Returns:
            np.array: Preprocessed video frames.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.frames_n:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.img_w, self.img_h))
            frames.append(frame)
        cap.release()
        if len(frames) < self.frames_n:
            raise ValueError(f"Video has less than {self.frames_n} frames.")
        frames = np.array(frames, dtype='float32') / 255.0
        return np.expand_dims(frames, axis=0)  # Add batch dimension

    @staticmethod
    def decode_predictions(predictions):
        """
        Decodes predictions from LipNet using CTC decoding.
        Args:
            predictions (np.array): Character probability predictions.
        Returns:
            list: Decoded text output.
        """
        decoded, _ = ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1])
        return ["".join([chr(c) for c in seq if c > 0]) for seq in decoded]

# Example Usage
if __name__ == "__main__":
    lipnet = LipNet()
    weights_path = "workspace/LipNet/evaluation/models/overlapped-weights368.h5"
    video_path = "workspace/LipNet/evaluation/samples/id2_vcd_swwp2s.mpg"

    try:
        # Load pretrained weights
        lipnet.load_weights(weights_path)

        # Make predictions
        predictions = lipnet.predict(video_path)
        decoded_output = lipnet.decode_predictions(predictions)
        print(f"Decoded Output: {decoded_output}")
    except Exception as e:
        print(f"Error: {e}")
