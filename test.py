from vis_module import vision_module

model = vision_module(32, 64)

if model:
    print("works")

model = None

if model:
    print("does not work")