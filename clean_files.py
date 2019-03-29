import os, shutil


processed_path = "C:/Users/saurabh/Documents/AI_GTA/processed"
orignal_path = "C:/Users/saurabh/Documents/AI_GTA/orignal"
ROI_path = "C:/Users/saurabh/Documents/AI_GTA/ROI"
ROI_raw_path = "C:/Users/saurabh/Documents/AI_GTA/ROI_raw"
Blurred_path = "C:/Users/saurabh/Documents/AI_GTA/Blurred_OG"


for the_file in os.listdir(processed_path):
    file_path = os.path.join(processed_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

for the_file in os.listdir(orignal_path):
    file_path = os.path.join(orignal_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)       

for the_file in os.listdir(ROI_path):
    file_path = os.path.join(ROI_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


for the_file in os.listdir(ROI_raw_path):
    file_path = os.path.join(ROI_raw_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

for the_file in os.listdir(Blurred_path):
    file_path = os.path.join(Blurred_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)