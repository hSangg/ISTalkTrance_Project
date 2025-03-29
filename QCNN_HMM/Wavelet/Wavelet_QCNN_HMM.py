import os
import numpy as np
import librosa
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import logging
import time
import joblib  # For saving HMM models
import pywt  # For wavelet transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Wavelet_QCNN_HMM:
    def __init__(self, wav_file, script_file, n_qubits=4, model_dir='wavelet_hmm_qcnn_models', 
                 wavelet='db4', level=5):
        """
        Initialize Quantum Speaker Identification system using Wavelet features
        
        Args:
            wav_file (str): Path to the audio file
            script_file (str): Path to the script file
            n_qubits (int): Number of qubits to use in quantum feature mapping
            model_dir (str): Directory to save trained models
            wavelet (str): Wavelet type to use (default: 'db4')
            level (int): Decomposition level for wavelet transform
        """
        self.wav_file = wav_file
        self.script_file = script_file
        self.n_qubits = n_qubits
        self.model_dir = model_dir
        self.wavelet = wavelet
        self.level = level
        
        # Set up quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Speakers and segments
        self.speakers = []
        self.segments = []
        
        # Quantum feature mapping circuit
        @qml.qnode(self.dev)
        def quantum_feature_map(inputs, weights):
            """
            Quantum feature mapping circuit
            
            Args:
                inputs (array): Input classical features
                weights (array): Parameterized weights for quantum gates
            
            Returns:
                Quantum state representation
            """
            # Encode classical features into quantum states
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RY(inputs[i], wires=i)
            
            # Apply entangling layers
            for i in range(self.n_qubits):
                qml.RZ(weights[i], wires=i)
                qml.CNOT(wires=[i, (i+1) % self.n_qubits])
            
            # Return quantum feature expectations
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]
        
        self.quantum_feature_map = quantum_feature_map

        # Create directory for saving models if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def parse_script(self):
        """Parse the script file to extract speaker segments."""
        logging.info("Parsing script file to extract segments.")
        with open(self.script_file, 'r') as f:
            lines = f.readlines()
        
        self.segments = []
        for line in lines:
            start, end, speaker = line.strip().split()
            self.segments.append({
                'start_time': self.time_to_seconds(start),
                'end_time': self.time_to_seconds(end),
                'speaker': speaker
            })
        
        self.speakers = list(set(seg['speaker'] for seg in self.segments))
        logging.info(f"Found {len(self.speakers)} speakers and {len(self.segments)} segments.")
        
    def time_to_seconds(self, time_str):
        """Convert time string to seconds."""
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    
    def extract_wavelet_features(self, audio, sr):
        """
        Extract wavelet-based features from audio signal
        
        Args:
            audio (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Wavelet-based features
        """
        # Apply wavelet decomposition
        coeffs = pywt.wavedec(audio, self.wavelet, level=self.level)
        
        # Extract statistical features from each coefficient level
        features = []
        for coef in coeffs:
            # Extract statistical features
            mean = np.mean(coef)
            std = np.std(coef)
            energy = np.sum(coef**2)
            entropy = -np.sum(coef**2 * np.log(coef**2 + 1e-10))
            
            features.extend([mean, std, energy, entropy])
        
        # Reshape features to have a 2D structure similar to MFCCs
        # Number of frames is determined by splitting audio into frames
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Create frames
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        wavelet_features = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(audio):
                frame = audio[start:end]
                
                # Apply wavelet on frame
                frame_coeffs = pywt.wavedec(frame, self.wavelet, level=min(3, self.level))
                
                # Extract features from coefficients
                frame_features = []
                for coef in frame_coeffs:
                    frame_features.extend([
                        np.mean(coef),
                        np.std(coef),
                        np.sum(coef**2)
                    ])
                
                wavelet_features.append(frame_features)
        
        # Ensure we have enough features
        if not wavelet_features:
            # If signal is too short, use global features
            wavelet_features = [features]
            
        return np.array(wavelet_features)
    
    def extract_quantum_enhanced_features(self):
        """
        Extract wavelet features and apply quantum feature mapping
    
        Returns:
            dict: Quantum-enhanced features for each speaker
        """
        # Load audio file
        y, sr = librosa.load(self.wav_file)
        
        # Prepare features for each speaker
        speaker_features = {speaker: [] for speaker in self.speakers}
        
        # Quantum weights (learnable parameters)
        quantum_weights = np.random.randn(self.n_qubits)
        
        for segment in self.segments:
            start_time_segment = time.time()  # Log start time for each segment
            # Convert time to sample indices
            start_sample = int(segment['start_time'] * sr)
            end_sample = int(segment['end_time'] * sr)
            
            # Extract audio segment
            segment_audio = y[start_sample:end_sample]
            
            # Extract wavelet features
            print(f"Extracting wavelet features for segment {segment['speaker']} from {segment['start_time']} to {segment['end_time']}")
            wavelet_features = self.extract_wavelet_features(segment_audio, sr)
            
            # Standardize features
            scaler = StandardScaler()
            wavelet_scaled = scaler.fit_transform(wavelet_features)
            
            print(f"Applying quantum feature mapping for segment {segment['speaker']}")
            # Apply quantum feature mapping
            quantum_features = []
            for feature_vector in wavelet_scaled:
                # Take first n_qubits features or pad if needed
                input_features = feature_vector[:self.n_qubits] if len(feature_vector) >= self.n_qubits else \
                                np.pad(feature_vector, (0, self.n_qubits - len(feature_vector)))
                
                # Map to quantum circuit
                quantum_mapped = self.quantum_feature_map(
                    input_features, 
                    quantum_weights
                )
                quantum_features.append(quantum_mapped)
            
            speaker_features[segment['speaker']].append(np.array(quantum_features))
            
            end_time_segment = time.time()  # Log end time for each segment
            print(f"Processed segment {segment['speaker']} in {end_time_segment - start_time_segment:.2f} seconds")
        
        return speaker_features
    
    def train_hmm_models(self, quantum_features):
        """
        Train HMM models using quantum-enhanced features
        
        Args:
            quantum_features (dict): Quantum-enhanced features for each speaker
        
        Returns:
            dict: Trained HMM models
        """
        logging.info("Training HMM models for each speaker.")
        hmm_models = {}
        
        for speaker, features_list in quantum_features.items():
            # Concatenate all features for this speaker
            speaker_features = np.concatenate(features_list)
            
            # Create and train HMM model
            model = hmm.GaussianHMM(
                n_components=4, 
                covariance_type='diag', 
                n_iter=100
            )
            model.fit(speaker_features)
            hmm_models[speaker] = model
        
        logging.info("HMM model training completed.")
        
        # Save the trained HMM models
        for speaker, model in hmm_models.items():
            model_filename = os.path.join(self.model_dir, f"{speaker}.pkl")
            joblib.dump(model, model_filename)
            logging.info(f"Saved HMM model for {speaker} at {model_filename}")
        
        return hmm_models

    def save_quantum_weights(self, weights):
        """
        Save the quantum feature mapping weights
        
        Args:
            weights (np.ndarray): Quantum weights to be saved
        """
        weights_filename = os.path.join(self.model_dir, 'quantum_weights.npy')
        np.save(weights_filename, weights)
        logging.info(f"Saved quantum weights at {weights_filename}")
    
    def identify_speaker(self, hmm_models, test_features):
        """
        Identify speaker using trained HMM models
        
        Args:
            hmm_models (dict): Trained HMM models
            test_features (numpy.ndarray): Quantum-enhanced test features
        
        Returns:
            str: Identified speaker
        """
        logging.info("Identifying speaker using HMM models.")
        # Compute log likelihood for each speaker's model
        likelihoods = {}
        for speaker, model in hmm_models.items():
            try:
                likelihoods[speaker] = model.score(test_features)
            except:
                likelihoods[speaker] = float('-inf')
        
        # Return the speaker with the highest likelihood
        return max(likelihoods, key=likelihoods.get)
    
    def run_speaker_identification(self):
        """
        Run the complete quantum-enhanced speaker identification pipeline
        
        Returns:
            dict: Identification results
        """
        logging.info("Starting speaker identification pipeline.")
        
        # Parse script and extract segments
        self.parse_script()
        
        # Extract quantum-enhanced features
        quantum_features = self.extract_quantum_enhanced_features()
        
        # Train HMM models
        hmm_models = self.train_hmm_models(quantum_features)
        
        # Perform testing
        results = {}
        for segment in self.segments:
            # Load audio segment
            y, sr = librosa.load(
                self.wav_file, 
                offset=segment['start_time'], 
                duration=segment['end_time'] - segment['start_time']
            )
            
            # Extract wavelet features
            wavelet_features = self.extract_wavelet_features(y, sr)
            
            # Standardize features
            scaler = StandardScaler()
            wavelet_scaled = scaler.fit_transform(wavelet_features)
            
            # Apply quantum feature mapping
            quantum_weights = np.random.randn(self.n_qubits)
            test_quantum_features = []
            for feature_vector in wavelet_scaled:
                # Take first n_qubits features or pad if needed
                input_features = feature_vector[:self.n_qubits] if len(feature_vector) >= self.n_qubits else \
                               np.pad(feature_vector, (0, self.n_qubits - len(feature_vector)))
                
                quantum_mapped = self.quantum_feature_map(
                    input_features, 
                    quantum_weights
                )
                test_quantum_features.append(quantum_mapped)
            
            # Convert to numpy array
            test_quantum_features = np.array(test_quantum_features)
            
            # Identify speaker
            predicted_speaker = self.identify_speaker(
                hmm_models, 
                test_quantum_features
            )
            
            results[segment['speaker']] = {
                'true_speaker': segment['speaker'],
                'predicted_speaker': predicted_speaker,
                'correct': predicted_speaker == segment['speaker']
            }
        
        logging.info("Speaker identification completed.")
        return results
wav_path = os.path.join("..", "..", "train_data", "meeting_1", "raw.wav")
script = os.path.join("..", "..", "train_data", "meeting_1", "script.txt")

trainer = Wavelet_QCNN_HMM(wav_path, script)
result = trainer.run_speaker_identification()
print(result)