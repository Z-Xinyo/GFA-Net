# Sequence based model arguments
encoder_arguments = {
   "t_input_size":3200, #时间维度输入大小  50*hidden_size(64)
   "s_input_size":4096, #空间维度输入大小  64*hidden_size(64)
   "hidden_size":512,
   "num_head":16,
   "num_class":128,
 }

data_path = "/dadaY/xinyu/dataset/self_supervised_data/"

class  opts_ntu_60_cross_view():

  def __init__(self):

   self.encoder_args = encoder_arguments

   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/ntu60/xview/train_data_joint.npy",
     "num_frame_path": data_path + "/ntu60/xview/train_num_frame.npy",  #帧数路径
     "l_ratio": [0.1,1], #采样比例
     "input_size": 64 #帧数
   }

class  opts_ntu_60_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/ntu60/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/ntu60/xsub/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

class  opts_ntu_120_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/ntu120/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/ntu120/xsub/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }

class  opts_ntu_120_cross_setup():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/ntu120/xsetup/train_data_joint.npy",
     "num_frame_path": data_path + "/ntu120/xsetup/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }


class  opts_pku_part1_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/crossclr_pkummd/pku_part1/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/crossclr_pkummd/pku_part1/xsub/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }

class  opts_pku_part2_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/crossclr_pkummd/pku_part2/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/crossclr_pkummd/pku_part2/xsub/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

