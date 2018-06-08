from __future__ import print_function
__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from data_util.bucketdata import BucketData
import sys
if sys.version_info >= (3,0):
    pass
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")


class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 img_width_range = (100, 800),
                 word_len = 60):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """
        print("DATA GEN")
        img_height = 32
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)
        # fixed width and len of text. It have to be changed for long handwritten line
        '''if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
                             (int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
                             (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]'''
        self.bucket_specs = [(int(100 / 4), int(word_len/4) + 2), (int(200 / 4), int(word_len/4) + 2),
                             (int(300 / 4), int(word_len/2) + 2), (int(400 / 4), int(word_len/2) + 2),
                             (int(500 / 4), word_len + 2), (int(600 / 4), word_len + 2), (int(700 / 4), word_len + 2), (int(800 / 4), word_len + 2)]
        self.max_len = word_len + 2

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}
        self.char2index = []
        with open(os.path.join(self.data_root,'sample.txt'), "r") as ins:
            for line in ins:
                self.char2index.append(line.strip())
        self.char2index.append(' ')
        print(self.char2index)
    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):

        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                #print l
                index = l.find(' ')
                img_path, lex = l[:index], l[index+1:]
                if (len(lex.strip()) >= self.max_len):
                    continue
                #print img_path, lex
                try:
                    img_bw, word = self.read_data(img_path, lex.strip())
                    #print (word)
                    #print (img_bw.shape)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def read_data(self, img_path, lex):
        #print 'read_data', self.data_root, img_path
        #assert 0 < len(lex) < self.bucket_specs[-1][1]
        assert 0 < len(lex) < self.max_len
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            #print self.data_root, img_path
            img = Image.open(img_file)
            w, h = img.size
            h_new = self.image_height
            w_new = max(100, math.floor( ( w * h_new / h ) / 100 ) * 100)
            if w_new > 800:
                w_new = 800
            img = img.resize((int(w_new), int(h_new)), Image.ANTIALIAS)
            '''aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)
'''
            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        #print lex
        try:
            for c in lex.decode('utf8'):
                #assert 96 < ord(c) < 123 or 47 < ord(c) < 58
                #print c
                if c in self.char2index:
                    word.append(self.char2index.index(c) + 3)
                else:
                    print("Error: Out of vocabolary")
                    print(lex.decode('utf8'), c, self.char2index)
                    exit(1)
        except:
            for c in lex:
                    #assert 96 < ord(c) < 123 or 47 < ord(c) < 58
                #print c
                if c in self.char2index:
                    word.append(self.char2index.index(c) + 3)
                else:
                    print("Error: Out of vocabolary")
                    print(lex, c, self.char2index)
                    exit(1)
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word
    def get_char(self, id):
        assert 3 <= id <= len(self.char2index)
        return self.char2index[id - 3]

def test_gen():
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = DataGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == img_height
    print(count)

if __name__ == '__main__':
    test_gen()
