import tensorflow as tf
tf.random.set_seed(42)
import os
import numpy as np

from hparams import args
from model import WaveGlow

def main(_):
   if args.saving_path:
      if not os.path.exists(args.saving_path):
         os.makedirs(args.saving_path)
   if args.sampling_path:
      if not os.path.exists(args.sampling_path):
         os.makedirs(args.sampling_path)
   if args.summary_dir:
      if not os.path.exists(args.summary_dir):
         os.makedirs(args.summary_dir)
   if args.infer_path:
      if not os.path.exists(args.infer_path):
         os.makedirs(args.infer_path)

   gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
   tfconfig = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

   with tf.compat.v1.Session(config=tfconfig) as sess:
      model = WaveGlow(sess)
      if args.is_training:
          #print(sess.run(model.train()))
         model.train()
      else:
         #print(sess.run(model.infer()))
         model.infer()

if __name__ == '__main__':
   tf.compat.v1.app.run()
