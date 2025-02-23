import librosa
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp (HH:MM:SS) to seconds."""
    try:
        parts = timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS format
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
    except Exception as e:
        logger.error(f"Error converting timestamp {timestamp}: {str(e)}")
        raise

def extract_audio_segment(audio_data: np.ndarray, sr: int, start_time: float, end_time: float) -> Optional[np.ndarray]:
    """Extract a segment of audio between start_time and end_time."""
    try:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Validate indices
        if start_sample >= len(audio_data) or end_sample > len(audio_data) or start_sample >= end_sample:
            logger.error(f"Invalid segment indices: start={start_sample}, end={end_sample}, audio_length={len(audio_data)}")
            return None
            
        return audio_data[start_sample:end_sample]
    except Exception as e:
        logger.error(f"Error extracting audio segment: {str(e)}")
        return None

def extract_mfcc_features(audio_path: str, script_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """Extract MFCC features from audio segments defined in the script file."""
    try:
        # Verify files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script file not found: {script_path}")
            
        # Load the full audio file
        logger.info(f"Loading audio file: {audio_path}")
        audio_data, sr = librosa.load(audio_path, sr=None)
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Failed to load audio file or file is empty")
            
        mfcc_features = []
        speaker_labels = []
        speaker_segments: Dict[str, List[np.ndarray]] = {}
        
        # Read and process the script file
        logger.info(f"Processing script file: {script_path}")
        with open(script_path, 'r', encoding='utf-8') as script_file:
            lines = script_file.readlines()
            if not lines:
                raise ValueError("Script file is empty")
                
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 3:
                    logger.warning(f"Skipping invalid line {line_num}: {line.strip()}")
                    continue
                    
                start_time, end_time, speaker = parts
                
                try:
                    # Convert timestamps to seconds
                    start_seconds = timestamp_to_seconds(start_time)
                    end_seconds = timestamp_to_seconds(end_time)
                    
                    # Extract the audio segment
                    segment = extract_audio_segment(audio_data, sr, start_seconds, end_seconds)
                    if segment is None or len(segment) == 0:
                        logger.warning(f"Skipping empty segment at line {line_num}")
                        continue
                    
                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(
                        y=segment, 
                        sr=sr,
                        n_mfcc=20,
                        hop_length=512,
                        n_fft=2048
                    )
                    
                    # Validate MFCC extraction
                    if mfcc is None or mfcc.size == 0:
                        logger.warning(f"Failed to extract MFCC features for segment at line {line_num}")
                        continue
                    
                    # Normalize MFCC features
                    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                    
                    # Add to the lists
                    mfcc_features.append(mfcc.T)
                    speaker_labels.append(speaker)
                    
                    # Store in speaker_segments
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append(mfcc.T)
                    
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        # Validate we have extracted features
        if not mfcc_features or not speaker_labels:
            raise ValueError("No valid features were extracted from the audio file")
            
        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total number of segments: {len(mfcc_features)}")
        logger.info(f"Number of unique speakers: {len(speaker_segments)}")
        logger.info("Segments per speaker:")
        for speaker, segments in speaker_segments.items():
            logger.info(f"  {speaker}: {len(segments)} segments")
        
        return mfcc_features, speaker_labels
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise