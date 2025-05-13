import io

import librosa
import numpy as np
import pennylane as qml

from modules.config import Config


class FeatureExtractor:
    def __init__(self, weights=None, n_qubits=7):
        self.weights = weights
        self.n_qubits = n_qubits
        self.n_mfcc = 20
        self.n_fft = 2048
        self.hop_length = 512
        
        self.use_gpu = False
        
        self.dev = qml.device("lightning.gpu" if self.use_gpu else "default.qubit", wires=n_qubits)
    @staticmethod
    def extract_mfcc_from_audio_bytes(audio_bytes):
        try:

            print("start extract mfcc features...")

            audio, sr = librosa.load(
                io.BytesIO(audio_bytes), 
                sr=Config.SAMPLE_RATE
            )
            mfcc_features = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=Config.N_MFCC
            )

            mfcc_delta = librosa.feature.delta(mfcc_features)
            mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)

            combined_mfcc = np.vstack([mfcc_features, mfcc_delta, mfcc_delta2])
            return combined_mfcc.T
        except Exception:
            raise

    @staticmethod
    def extract_mfcc_from_segment(segment, sr):
        try:
            if len(segment) < sr * 0.025:
                segment = np.pad(segment, (0, int(sr * 0.025) - len(segment)))

            mfcc_features = librosa.feature.mfcc(
                y=segment,
                sr=sr,
                n_mfcc=Config.N_MFCC
            )
            mfcc_delta = librosa.feature.delta(mfcc_features)
            mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
            combined_mfcc = np.vstack([mfcc_features, mfcc_delta, mfcc_delta2])
            return combined_mfcc.T
        except Exception as e:
            print(f"Error extracting MFCC for segment of length {len(segment)}: {e}")
            return None

    @staticmethod
    def extract_append_features(audio_path, annotations):
        y, sr = librosa.load(audio_path, sr=None)
        speaker_data = {}
        for start, end, speaker in annotations:
            segment = y[int(start * sr): int(end * sr)]
            mfcc = FeatureExtractor.extract_mfcc_from_segment(segment, sr)

            if mfcc is None:
                print(f"❌ Skipping segment {start}-{end} for {speaker} due to MFCC extraction failure ❌")
                continue

            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append(mfcc)
        return speaker_data


    def extract_mfcc(self, audio_data, sample_rate):
        """Extract MFCC features from audio data"""
        try:

            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )


            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)


            mfcc_features = np.concatenate([mfcc_mean, mfcc_std])

            return mfcc_features
        except Exception as e:
            print(f"Error extracting MFCC: {e}")

            return np.array([])

    def process_qcnn(self, mfcc_features):
        """Process MFCC features through QCNN"""
        quantum_features = []


        if len(mfcc_features) < self.n_qubits:
            mfcc_features = np.pad(mfcc_features, (0, self.n_qubits - len(mfcc_features)))


        input_features = mfcc_features[:self.n_qubits]


        input_features = np.clip(input_features, -np.pi, np.pi)

        q_output = self.random_qcircuit(input_features)

        quantum_features.append(q_output)


        combined_features = np.concatenate([
            mfcc_features,
            np.array(quantum_features).flatten()
        ])

        return combined_features.reshape(1, -1)
    
    def random_qcircuit(self, inputs):
        """Random quantum circuit for feature transformation"""
        @qml.qnode(self.dev)
        def circuit(inputs):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(np.random.uniform(-np.pi, np.pi), wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RY(np.random.uniform(-np.pi, np.pi), wires=i + 1)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(inputs)