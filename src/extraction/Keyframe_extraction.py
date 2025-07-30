import numpy as np
from collections.abc import Iterable
from tqdm import tqdm

from .Kmeans_improvment import kmeans_silhouette
from .Redundancy import redundancy


def shot_keyframe_extraction(shot_start_frame_indices: list[int], features: Iterable, video_path: str) -> list[int]:
    features_np = np.asarray(list(features))

    # Clustering at each shot to obtain keyframe sequence numbers
    keyframe_indices: list[int] = list()
    for i in tqdm(range(len(shot_start_frame_indices)-1), desc="Iterating Over Shots for Keyframe Extraction"):
        start: int = shot_start_frame_indices[i]
        end: int = shot_start_frame_indices[i+1] - 1
        sub_features = features_np[start:end]
        _best_labels, _best_centers, _k, indices = kmeans_silhouette(sub_features)
        indices: list[int] = [x + start for x in indices]
        final_indices: list[int] = redundancy(video_path, indices, 0.94)
        keyframe_indices.extend(final_indices)
    keyframe_indices.sort()

    return keyframe_indices
