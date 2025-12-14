import pickle

with open("/workspace/my_deltamodel/data/camels_dataset", 'rb') as f:
    forcing, target, _ = pickle.load(f)
    
print(forcing.shape) # (671, 12418, 3) 流域数，时间步长，输入信息 （降雨，气温，蒸发）
print(target.shape) # (671, 12418, 1) 流域数，时间步长，输入信息 （径流）
print(type(forcing)) # <class 'numpy.ndarray'>
print(type(target)) # <class 'numpy.ndarray'>