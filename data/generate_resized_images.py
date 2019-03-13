from PIL import Image
import os
import glob 
# Function generates resized omniglot images
#Input description:
# raw_img_dir : Directory containing the original images
# resized_img_dir: Directory where resized image needs to be stored

def GenerateResizedImages(self, raw_img_dir, resized_img_dir):
    if not os.path.isdir(resized_img_dir):
        os.mkdir(resized_img_dir)
    resized_img_dir = resized_img_dir + 'testing/'
    if not os.path.isdir(resized_img_dir):
        os.mkdir(resized_img_dir)
    # Find the list of all images
    char_file_list = glob.glob(raw_img_dir)
    #For each file - resize and save in given path
    for i,fname in enumerate(char_file_list):
        print("Resizing image " + str(i) +'-th image out of ' + str(len(char_file_list)) + ' images..')
        alphabet = fname.split('/')[-3]
        character_num = fname.split('/')[-2][-2:]
        # Create directory if it doesnt exist
        alphabet_dir = resized_img_dir + alphabet
        char_dir = alphabet_dir + '/' + character_num
        if not os.path.isdir(alphabet_dir):
          os.mkdir(alphabet_dir)
        if not os.path.isdir(char_dir):
          os.mkdir(char_dir)
        fn = fname.split('/')[-1]
        img = Image.open(fname).convert('RGB').resize((self.char_w, self.char_h), Image.BILINEAR) # Convert to RGB and resize image to W X H
        print(char_dir+'/'+fn)
        img.save(char_dir+'/'+fn)
    return
