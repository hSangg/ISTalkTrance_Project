import logging
import os
import time

import joblib
import librosa
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DVECTOR_QCNN_HMM:
    def __init__(self, wav_file, script_file, n_qubits=4, model_dir='dvector_hmm_qcnn_models'):
        """
        Initialize Quantum Speaker Identification system with DVectors
        
        Args:
            wav_file (str): Path to the audio file
            script_file (str): Path to the script file
            n_qubits (int): Number of qubits to use in quantum feature mapping
            model_dir (str): Directory to save trained models
        """
        self.wav_file = wav_file
        self.script_file = script_file
        self.n_qubits = n_qubits
        self.model_dir = model_dir
        
        # Set up quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Speakers and segments
        self.speakers = []
        self.segments = []
        
        # Initialize the DVectorExtractor
        self.dvector_extractor = DVectorExtractor()
        
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
    
    def extract_quantum_enhanced_features(self):
        """
        Extract DVector features and apply quantum feature mapping
    
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
            
            # Extract DVector features
            print(f"Extracting DVectors for segment {segment['speaker']} from {segment['start_time']} to {segment['end_time']}")
            dvectors = self.dvector_extractor.extract(segment_audio, sr)
            
            # Standardize features
            scaler = StandardScaler()
            dvectors_scaled = scaler.fit_transform(dvectors)
            
            print(f"Applying quantum feature mapping for segment {segment['speaker']}")
            # Apply quantum feature mapping
            quantum_features = []
            for feature_vector in dvectors_scaled:
                # Map first n_qubits features to quantum circuit
                quantum_mapped = self.quantum_feature_map(
                    feature_vector[:self.n_qubits], 
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
            num_states = min(4, speaker_features.shape[0])
            # Create and train HMM model
            model = hmm.GaussianHMM(
                n_components=num_states, 
                covariance_type='diag', 
                n_iter=100
            )
            model.fit(speaker_features)
            hmm_models[speaker] = model
        
        logging.info("HMM model training completed.")
        
        # Save the trained HMM models
        for speaker, model in hmm_models.items():
            model_filename = os.path.join(self.model_dir, f"hmm_model_{speaker}.pkl")
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
            
            # Extract DVector features
            dvectors = self.dvector_extractor.extract(y, sr)
            
            # Standardize features
            scaler = StandardScaler()
            dvectors_scaled = scaler.fit_transform(dvectors)
            
            # Apply quantum feature mapping
            quantum_weights = np.random.randn(self.n_qubits)
            test_quantum_features = []
            for feature_vector in dvectors_scaled:
                quantum_mapped = self.quantum_feature_map(
                    feature_vector[:self.n_qubits], 
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


class DVectorExtractor(nn.Module):
    """
    Deep Speaker Embedding (d-vector) extractor based on a deep neural network
    """
    def __init__(self, input_dim=40, hidden_dim=256, embedding_dim=256):
        """
        Initialize the DVectorExtractor
        
        Args:
            input_dim (int): Dimension of input features (typically mel filterbanks)
            hidden_dim (int): Hidden layer dimension
            embedding_dim (int): Dimension of the speaker embedding
        """
        super(DVectorExtractor, self).__init__()
        
        # LSTM-based feature extractor
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal pooling layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Embedding layer
        self.embedding = nn.Linear(hidden_dim * 2, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Speaker embedding of shape (batch_size, embedding_dim)
        """
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # Compute attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply weighted sum to get context vector
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim*2)
        
        # Generate embedding
        embedding = self.embedding(context)  # (batch_size, embedding_dim)
        
        # L2 normalization for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def extract(self, audio, sr):
        """
        Extract d-vector features from audio
        
        Args:
            audio (numpy.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            numpy.ndarray: D-vector features
        """
        # Extract mel spectrograms as input features
        mel_specs = self.extract_mel_spectrograms(audio, sr)
        
        # Convert to torch tensor
        mel_specs_tensor = torch.tensor(mel_specs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Set model to evaluation mode
        self.eval()
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.forward(mel_specs_tensor).squeeze(0).numpy()
        
        return embeddings.reshape(1, -1)  # Reshape to ensure 2D
    
    def extract_mel_spectrograms(self, audio, sr, n_mels=40, n_fft=512, hop_length=160):
        """
        Extract mel spectrograms from audio
        
        Args:
            audio (numpy.ndarray): Audio signal
            sr (int): Sample rate
            n_mels (int): Number of Mel bins
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            
        Returns:
            numpy.ndarray: Mel spectrograms
        """
        # Extract mel spectrograms
        mel_specs = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Convert to log scale
        mel_specs = librosa.power_to_db(mel_specs, ref=np.max)
        
        # Transpose to get time as first dimension
        mel_specs = mel_specs.T
        
        return mel_specs
wav_path = os.path.join("..", "..", "train_data", "meeting_1", "raw.wav")
script = os.path.join("..", "..", "train_data", "meeting_1", "script.txt")

trainer = DVECTOR_QCNN_HMM(wav_path, script)
result = trainer.run_speaker_identification()
print(result)