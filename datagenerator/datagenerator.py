import matplotlib.pyplot as plt
import pickle
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import os
import glob
import csv
import shutil
import time
from natsort import natsorted
import logging
import sys
import tensorflow as tf
from tensorflow.python.platform import flags

flags = tf.app.flags 
FLAGS = tf.app.flags.FLAGS

class DataGenerator(object):
    def __init__(self, num_examples_per_task, num_tasks):
        #t = int( time.time() * 1000.0 )
        #rand_seed =  ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)
        #tf.set_random_seed(rand_seed)
        #np.random.seed(rand_seed)
        #random.seed(rand_seed)
        self.canvas_w = 150
        self.canvas_h = 150
        self.char_w = 34
        self.char_h = 34
        self.padding = 25
        self.num_tasks = num_tasks # # Batch Size: Batch of tasks
        self.num_examples_per_task = num_examples_per_task # Number of Examples per task
        self.img_size = (self.canvas_w, self.canvas_h)
        self.dim_input = np.prod(self.img_size)*3
        self.dim_output = 2
        self.num_val = 100
        self.num_train = 1200
        self.training_data_dir = FLAGS.training_data_dir
        self.testing_data_dir = FLAGS.testing_data_dir
        self.resized_data_dir = FLAGS.resized_data_dir

        print(self.num_tasks)
        print(self.num_examples_per_task)

        if FLAGS.data_resize:
          self.GenerateResizedImages('data/omniglot/images_background/*/*/*.png', self.resized_data_dir, is_training=True)
          self.GenerateResizedImages('data/omniglot/images_evaluation/*/*/*.png', self.resized_data_dir, is_training=False)
          input('Done resizing.... Press Any Key to Continue...')

        if FLAGS.data_generation:
          print('Generating Data!!! This will take some time....')
          self.GenerateAllTasks()
          input('Done Generating.... Press Any Key to Continue...')

        self.training_char_folders, self.metaval_char_folders = self.extract_task_folders(self.training_data_dir)

        #Create a training dataset generator
        training_image_files, training_labels = self.get_imagefiles_with_labels(self.training_char_folders, self.num_examples_per_task)
        print(training_image_files)
        training_dataset = tf.data.Dataset.from_tensor_slices((training_image_files,training_labels))
        training_dataset = training_dataset.map(self.image_reader_function)
        training_dataset = training_dataset.batch(self.num_tasks*self.num_examples_per_task)
        training_dataset = training_dataset.prefetch(1)
        train_iter = training_dataset.make_initializable_iterator()
        train_iter_init_op = train_iter.initializer
        with tf.control_dependencies([train_iter_init_op]):
          train_images, train_labels = train_iter.get_next()

        train_images = tf.reshape(train_images, [-1,self.num_examples_per_task, self.dim_input])
        train_labels = tf.reshape(train_labels, [-1,self.num_examples_per_task, 2])
        
        self.train_inputs = {'images': train_images, 'labels': train_labels, 'iter_init_op': train_iter_init_op}
        
        #Create a met validation dataset generator
        metaval_image_files, metaval_labels = self.get_imagefiles_with_labels(self.metaval_char_folders, self.num_examples_per_task)
        metaval_dataset = tf.data.Dataset.from_tensor_slices((metaval_image_files,metaval_labels))
        metaval_dataset = metaval_dataset.map(self.image_reader_function)
        metaval_dataset = metaval_dataset.batch(self.num_tasks*self.num_examples_per_task)
        metaval_dataset = metaval_dataset.prefetch(1)
        metaval_iter = metaval_dataset.make_initializable_iterator()
        metaval_iter_init_op = metaval_iter.initializer
        with tf.control_dependencies([metaval_iter_init_op]):
          metaval_images, metaval_labels = metaval_iter.get_next()
        
        metaval_images = tf.reshape(metaval_images, [-1,self.num_examples_per_task, self.dim_input])
        metaval_labels = tf.reshape(metaval_labels, [-1,self.num_examples_per_task, 2])
        self.metaval_inputs = {'images': metaval_images, 'labels': metaval_labels, 'iter_init_op': metaval_iter_init_op}

        # Test that the images are as per the image files passed
        with tf.Session() as sess:
          for _ in range(self.num_tasks):
            image_batch, label_batch = sess.run([train_images, train_labels]) # index 0 contains image and index 1 contains label
            print(image_batch.shape)
            print(label_batch.shape)
            image_batch = image_batch.reshape([-1,self.num_examples_per_task, self.canvas_w,self.canvas_h,3])
            for i in range(self.num_tasks):
              plt.figure(i+1)
              for j in range(self.num_examples_per_task):
                plt.subplot(2,self.num_examples_per_task,j+1)
                plt.imshow(image_batch[i,j,:,:])
                print(label_batch[i,j,:])
        

    def GenerateResizedImages(self, raw_img_dir, resized_img_dir, is_training=True):
        if not os.path.isdir(resized_img_dir):
            os.mkdir(resized_img_dir)
        if not is_training:
            resized_img_dir = resized_img_dir + 'testing/'
        else:
            resized_img_dir = resized_img_dir + 'training/'
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

    def GenerateAllTasks(self):
        self.M = 24 # Number of random examples for each given character
        self.DeleteDataset()
        self.GenerateTrainingTasks()
        self.GenerateTestingTasks()
        return

    def DeleteDataset(self):
        try:
            shutil.rmtree(self.training_data_dir)
        except FileNotFoundError:
            pass

        try:
            shutil.rmtree(self.testing_data_dir)
        except FileNotFoundError:
            pass
        return
      
    def GenerateTasks(self, char_data_dict, dir_prefix_str):
        os.system('rm -f ' + dir_prefix_str + '*/*')
        char_ids = list(char_data_dict.keys())
        char_ids.sort()
        print(char_ids)
        # For each character create M random examples
        for target_char_id in char_ids:
            print('Generating ' + str(target_char_id) + ' out of ' + str(len(char_ids)))
            column_list = ["char_id", "example #", "x", "y"]
            df = pd.DataFrame([], columns=column_list)
            dir_str = dir_prefix_str + str(target_char_id) + '/'
            try:
                os.mkdir(dir_str)
            except FileExistsError:
                pass
            for m in range(self.M):
                distractor_char_ids = random.sample(char_ids, 4) # Sample 4 unique characters
                if target_char_id in distractor_char_ids:
                    distractor_char_ids.remove(target_char_id) # Remove target character from list if it was sampled
                quadrant_list = [i for i in range(4)]
                random.shuffle(quadrant_list) # quadrant list
                canvas_img = Image.new('RGB', (self.canvas_w, self.canvas_h), color=(255, 255, 255)) # White background image
                # Create image with 4 unique characters
                for i in range(4):
                    pos_x, pos_y = self.get_random_pos(quadrant_list[i]) # Get a random position for the given quadrant

                    if i == 0:
                        # Record relative position of target character on the canvas
                        yp = [pos_x*1./self.canvas_w, pos_y*1./self.canvas_h]
                        val_list = [str(target_char_id), str(m), str(yp[0]), str(yp[1])]
                        df = df.append(pd.DataFrame([val_list], columns=column_list), ignore_index=True)
                        # paste target char image
                        char_img_file = random.choice(char_data_dict[target_char_id]['image_file_list']) # Randomly choose any one sample of the target character
                    else:
                        char_img_file = random.choice(char_data_dict[distractor_char_ids[i-1]]['image_file_list']) # Randomly choose any one sample of the distractor character

                    # Paste Character Image on canvas image
                    char_img = Image.open(char_img_file)
                    canvas_img.paste(char_img, box=(pos_x - self.char_w//2, pos_y - self.char_h//2))

                    if i == 0:
                        stx, sty = pos_x-1+random.randint(-2, 2), pos_y-1+random.randint(-2, 2)
                        for px in range(stx, stx+3):
                            for py in range(sty, sty+3):
                                canvas_img.putpixel((px, py), (255, 0, 0)) # Place red dot near target character on the canvas

                    img_name_str = dir_str + str(target_char_id) + '_' + str(m) + '.png'

                    Image.fromarray(np.array(canvas_img)).save(img_name_str)

            df.to_csv(dir_str + str(target_char_id) + '.csv')
            print('done...')
        return
    
    # Function to generate a character dictionary
    # Input: Path
    # Output: Character Dictionary that contains list of images for each type of character
    def get_data_dict(self, search_path):
        data_dict = {}
        char_file_list = glob.glob(search_path)
        
        for i,fname in enumerate(char_file_list):
            alphabet = fname.split('/')[-3]
            character_num = fname.split('/')[-2][-2:]
            fn = fname.split('/')[-1]
            c_id, img_id = int(fn[:4]), int(fn[5:7])
            if c_id not in data_dict:
                data_dict[c_id] = {}
                data_dict[c_id]['image_file_list'] = []
                data_dict[c_id]['alphabet'] = alphabet
                data_dict[c_id]['character number'] = character_num
                print(data_dict[c_id])
                print(fname)
            data_dict[c_id]['image_file_list'].append(fname)
        return data_dict


    def GenerateTrainingTasks(self):
        self.training_data_dict = self.get_data_dict('data/resized_data/training/*/*/*.png')
        try:
            os.mkdir(self.training_data_dir)
        except FileExistsError:
            pass

        self.GenerateTasks(self.training_data_dict, self.training_data_dir + 'char_')
        return

    def GenerateTestingTasks(self):
        print('Generating resized images!!! This will take a while..')
        self.testing_data_dict = self.get_data_dict('data/resized_data/testing/*/*/*.png')
        print('done...')
        try:
            os.mkdir(self.testing_data_dir)
        except FileExistsError:
            pass

        self.GenerateTasks(self.testing_data_dict, self.testing_data_dir + 'char_')
        return

    # Function to return random position in an image of size w x h given a quadrant of interest
    # Origin is top left corner of image (which also corresponds to quadrant 0)
    # Quadrant count is clockwise
    def get_random_pos(self, q):
        #return [(w/4, h/4), (3*w/4, h/4), (3*w/4, 3*h/4), (w/4, 3*h/4)][q]
        assert self.canvas_w/2 - self.padding > 0
        assert self.canvas_h/2 - self.padding > 0
        assert self.canvas_w > 0
        assert self.canvas_h > 0
        assert self.padding > 0
        if q == 0:
            return random.randint(self.padding, self.canvas_w/2-self.padding), random.randint(self.padding, self.canvas_h/2-self.padding)
        if q == 1:
            return random.randint(self.canvas_w/2+self.padding, self.canvas_w-self.padding), random.randint(self.padding, self.canvas_h/2-self.padding)
        if q == 2:
            return random.randint(self.canvas_w/2+self.padding, self.canvas_w-self.padding), random.randint(self.canvas_h/2+self.padding, self.canvas_h-self.padding)
        if q == 3:
            return random.randint(self.padding, self.canvas_w/2-self.padding), random.randint(self.canvas_h/2+self.padding, self.canvas_h-self.padding)
        return "Unknown Quadrant"
      
    def extract_task_folders(self, data_folder, num_val=300):
      character_folders = []
      for character in os.listdir(data_folder):
          path = os.path.join(data_folder,character)
          if os.path.isdir(path):
              character_folders.append(path)
      
      return character_folders[:-num_val], character_folders[-num_val:]
    
            ## Image helper
    def get_imagefiles_with_labels(self, paths, nb_samples):
        sampler = lambda x: random.sample(x, nb_samples)
        image_files = []
        labels = []
        for path in paths:
            file_list = glob.glob(path+'/*.png')
            try:
              csv_file_path = glob.glob(path+'/*.csv')[0]
            except IndexError:
                print(glob.glob(path+'/*.csv'))

            csv_file = pd.read_csv(csv_file_path, header=None)
            idx = sampler(range(len(file_list))) # Faster to sample integers rather than the list of files
            target_list = [tuple(csv_file.iloc[i,2:]) for i in idx]

            sample_tuple_list = zip(target_list,[file_list[i] for i in idx])
            for label, image in sample_tuple_list:
                image_files.append(image)
                labels.append(label)
        return image_files, labels
      
    def image_reader_function(self,image_file, label):
        image_string = tf.read_file(image_file)
        image = tf.image.decode_png(image_string)
        image.set_shape((self.img_size[0],self.img_size[1],3))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0

        return image, tf.cast(label, dtype=tf.float32)
