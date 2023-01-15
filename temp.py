import os

import random
ran_list = random.sample(range(0, 601), 120)
label = "OK"
train_folder = 'C:/Users/Kien_PC/Documents/Workspace/ML/Sign_language/data/train/'
test_folder = 'C:/Users/Kien_PC/Documents/Workspace/ML/Sign_language/data/test/'

for i in ran_list:
    os.rename(train_folder + label + "/" + str(i) + ".png",
              test_folder + label + "/" + str(i) + ".png")

