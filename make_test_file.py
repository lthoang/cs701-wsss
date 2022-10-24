import os
with open('public/val.txt', 'w') as f:
    for root, dirs, files in os.walk("./public/val_image"):
        path = root.split(os.sep)
        for file in files:
            f.write('{}\n'.format(file))
