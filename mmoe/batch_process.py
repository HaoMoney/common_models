# -*- encode: utf-8 -*-
"""
@author: visionzhong
@date: 2020-10-09
"""


import tensorflow as tf


class BatchClickMasked:

    def __init__(self, infiles, batch_size, is_speech=False, num_epoch=20):
        print("MYINFO - train file:")
        print("\n".join(infiles))
        #num_parallel_calls:cpu cores
        self.dataset = tf.data.TextLineDataset(infiles)
        #self.dataset = self.dataset.repeat(num_epoch)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.map(
                self.line_decode,
                num_parallel_calls=8)
        self.dataset = self.dataset.shuffle(buffer_size=batch_size*20)
        self.dataset = self.dataset.batch(batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()

    def line_decode(self, line):
        #src_line = tf.string_strip(src_line)
        #src_line = tf.string_split([src_line], delimiter="###", skip_empty=True)
        #line = tf.string_split([src_line.values[1]], delimiter="\t", skip_empty=True)
        line = tf.string_strip(line)
        line = tf.string_split([line], delimiter="\t", skip_empty=True)
        # query feature 
        topic_val = tf.string_split([line.values[1]], delimiter=" ", skip_empty=True)
        topic_val = tf.string_to_number(topic_val.values, out_type=tf.float32)
        # prod feature 
        second_topic_val = tf.string_split([line.values[3]], delimiter=" ", skip_empty=True)
        second_topic_val = tf.string_to_number(second_topic_val.values, out_type=tf.float32)
       
        #labels 
        label1 = tf.string_to_number(line.values[8], out_type=tf.float32)
        label2 = tf.string_to_number(line.values[9], out_type=tf.float32)
        return (topic_val, second_topic_val, label1, label2)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dataset = BatchClickMasked(
            infiles=["./test_data"],
            # infiles=["./dataset/pretrain/search_query2title_pretrain.8"
            #          ".pair.speech.mask.segnet2.vocab4.id.2"],
            batch_size=2,
            is_speech=False,
            num_epoch=1000000,
            )
    with tf.Session() as session:
        for i in range(1):
            print(session.run(dataset.next_element))
            # if i % 1000 == 0:
            #     print(i)
            break
