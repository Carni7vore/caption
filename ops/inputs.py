# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ops import read_sentiments
from ops import read_output
from inference_utils import vocabulary
import tensorflow as tf


# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
def parse_sequence_example(serialized, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={
      image_feature: tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
      caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
    })

  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image, caption


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
      data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
      capacity=capacity,
      min_after_dequeue=min_queue_examples,
      dtypes=[tf.string],
      name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
      data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
      capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
    values_queue, enqueue_ops))
  tf.summary.scalar(
    "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
    tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 5 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  # TODO
  fname = "im2txt/all.txt"
  dict_words, _ = read_sentiments.read_sentiments()
  dict_sentence = read_output.read_output(fname)

  # print(dict_words[15])
  # dic_keys= tf.Variable(dict_words.keys, dtype=tf.string)
  keys = dict_words.keys()
  values = dict_words.values()
  # print(keys[0])
  # print(values[0])
  with tf.variable_scope("hashtable"), tf.device("/cpu:0"):

    dic_value = tf.cast(values, dtype=tf.int64)
    dic_keys = tf.cast(keys, dtype=tf.int64)
    # dic_value=tf.convert_to_tensor(dict_words.values, dtype=tf.float32)
    table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(dic_keys, dic_value), -1
    )

    sentence_keys = tf.cast(dict_sentence.keys(), dtype=tf.string)
    sentence_vals = tf.cast(dict_sentence.values(), dtype=tf.int64)
    table2 = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(sentence_keys, sentence_vals), -1
    )

  enqueue_list = []
  for image, caption in images_and_captions:
    caption_length = tf.shape(caption)[0]
    input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
    # vocab = vocabulary.Vocabulary("im2txt/data/mscoco/word_counts.txt")
    # print(vocab.word_to_id('enjoying'))
    # TODO
    sentiments_seq = []
    # float_words = tf.cast(caption, dtype=tf.float32)
    string_words = tf.as_string(caption)
    joined_string = tf.reduce_join(string_words, axis=0, separator="")
    # num1= tf.cast(joined_string,tf.int64)
    sentiment_seq = table2.lookup(joined_string)

    # sentiment= table.lookup(target_word)
    # sentiments_seq=table.lookup(caption)
    # sentiment_seq=tf.stack(sentiments_seq)
    input_seq = tf.slice(caption, [0], input_length)

    target_seq = tf.slice(caption, [1], input_length)
    indicator = tf.ones(input_length, dtype=tf.int32)
    enqueue_list.append([image, input_seq, target_seq, sentiment_seq, indicator])

  images, input_seqs, target_seqs, sentiments_seq, mask = tf.train.batch_join(
    enqueue_list,
    batch_size=batch_size,
    capacity=queue_capacity,
    dynamic_pad=True,
    name="batch_and_pad")

  if add_summaries:
    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

  return images, input_seqs, target_seqs, mask, sentiments_seq
