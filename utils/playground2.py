"""results = [
    [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
    [[13, 14, 15], [17, 18, 19], [21, 22, 23]],
]
results = results[:, :, -1]
print(results) """

import json
import pickle
from pathlib import Path

pickle_path = Path(
    r"results\backup\(3,4)_game_size_10_vsf_3\standard\0\entropy_scores.pkl"
)
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# print("Keys under 'loss_train':", list(data['loss_train'].keys())[:10])  # print first 10 keys
print(json.dumps(data, indent=4))
