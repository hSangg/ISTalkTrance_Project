import io

import librosa
import numpy as np
import pennylane as qml



class FeatureExtractor:
    def __init__(self, qcnn_weights=None, n_qubits=7):
            """
            Initialize the feature extractor with saved QCNN weights
            
            Parameters:
            -----------
            qcnn_weights : array-like
                The weights for the quantum circuit
            n_qubits : int
                Number of qubits in the quantum circuit
            """
            self.n_qubits = n_qubits
            self.qcnn_weights = qcnn_weights
            
            # MFCC parameters
            self.n_mfcc = 20
            self.n_fft = 2048
            self.hop_length = 512
            
            # Initialize quantum device
            import pennylane as qml
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            # Define quantum circuit
            @qml.qnode(self.dev)
            def qcnn(inputs, weights, n_qubits=7, num_layers=2):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RY(inputs[i], wires=i)
                
                for layer in range(num_layers):
                    for i in range(n_qubits-1):
                        idx = layer * (n_qubits-1) + i
                        if idx < len(weights):
                            qml.CRZ(weights[idx], wires=[i, i+1])
                    for i in range(n_qubits):
                        weight_idx = layer * n_qubits + i
                        if weight_idx < len(weights):
                            qml.RY(weights[weight_idx], wires=i)
                
                for i in range(0, n_qubits, 2):
                    if i + 1 < n_qubits:
                        qml.CNOT(wires=[i, i+1])
                        qml.RY(weights[-1] * np.pi, wires=i+1)
                
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.qcnn = qcnn
        
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
    

    def extract_mfcc(self, audio_data, sample_rate=16000):
        """
        Extract MFCC features from audio data
        
        Parameters:
        -----------
        audio_data : array-like
            Raw audio signal
        sample_rate : int
            Sample rate of the audio
            
        Returns:
        --------
        array-like
            MFCC features
        """
        import numpy as np
        import librosa
        
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Calculate statistics over time to get fixed-length features
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Combine mean and std for a more comprehensive feature vector
            mfcc_features = np.concatenate([mfcc_mean, mfcc_std])
            
            return mfcc_features
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            # Return empty array if extraction fails
            return np.array([])
    
    def process_qcnn(self, mfcc_features):
        """
        Process MFCC features through QCNN using saved weights
        
        Parameters:
        -----------
        mfcc_features : array-like
            MFCC features to process
            
        Returns:
        --------
        array-like
            Combined MFCC and quantum features
        """
        import numpy as np
        
        quantum_features = []
        
        # Ensure we have enough features for n_qubits
        if len(mfcc_features) < self.n_qubits:
            mfcc_features = np.pad(mfcc_features, (0, self.n_qubits - len(mfcc_features)))
        
        # Use first n_qubits features for quantum circuit
        input_features = mfcc_features[:self.n_qubits]
        
        # Scale features to appropriate range for quantum circuit
        input_features = np.clip(input_features, -np.pi, np.pi)
        
        # Use the saved weights with our quantum circuit
        q_output = self.qcnn(input_features, self.qcnn_weights, n_qubits=self.n_qubits)
        quantum_features.append(q_output)
        
        # Combine classical and quantum features
        combined_features = np.concatenate([
            mfcc_features,  # Original MFCC features
            np.array(quantum_features).flatten()  # Quantum transformed features
        ])
        
        return combined_features.reshape(1, -1)

