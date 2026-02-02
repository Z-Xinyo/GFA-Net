data_path = "/dadaY/xinyu/dataset/self_supervised_data"

class  opts_ntu_60_cross_view():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
      "t_input_size":3200,
      "s_input_size":4096,
      "hidden_size":2048,
      "num_head":16,
      "num_class":60,
      }
  
    # feeder
    self.train_feeder_args = {
      "data_path": data_path + "/ntu60/xview/train_data_joint.npy",
      "label_path": data_path + "/ntu60/xview/train_label.pkl",
      'num_frame_path': data_path + "/ntu60/xview/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': data_path + "/ntu60/xview/val_data_joint.npy",
      'label_path': data_path + "/ntu60/xview/val_label.pkl",
      'num_frame_path': data_path + "/ntu60/xview/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_ntu_60_cross_subject():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
      "t_input_size":3200,
      "s_input_size":4096,
      "hidden_size":2048,
      "num_head":16,
      "num_class":60,
      }

    # feeder
    self.train_feeder_args = {
      "data_path": data_path + "/ntu60/xsub/train_data_joint.npy",
      "label_path": data_path + "/ntu60/xsub/train_label.pkl",
      'num_frame_path': data_path + "/ntu60/xsub/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': data_path + "/ntu60/xsub/val_data_joint.npy",
      'label_path': data_path + "/ntu60/xsub/val_label.pkl",
      'num_frame_path': data_path + "/ntu60/xsub/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }



class  opts_ntu_120_cross_subject():
  def __init__(self):

    # Sequence based model
    self.encoder_args = {
      "t_input_size":3200,
      "s_input_size":4096,
      "hidden_size":2048,
      "num_head":16,
      "num_class":120,
    }

    # feeder
    self.train_feeder_args = {
      "data_path": data_path + "/ntu120/xsub/train_data_joint.npy",
      "label_path": data_path + "/ntu120/xsub/train_label.pkl",
      'num_frame_path': data_path + "/ntu120/xsub/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': data_path + "/ntu120/xsub/val_data_joint.npy",
      'label_path': data_path + "/ntu120/xsub/val_label.pkl",
      'num_frame_path': data_path + "/ntu120/xsub/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_ntu_120_cross_setup():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
      "t_input_size":3200,
      "s_input_size":4096,
      "hidden_size":2048,
      "num_head":16,
      "num_class":120,
    }

    # feeder
    self.train_feeder_args = {
      "data_path": data_path + "/ntu120/xsetup/train_data_joint.npy",
      "label_path": data_path + "/ntu120/xsetup/train_label.pkl",
      'num_frame_path': data_path + "/ntu120/xsetup/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': data_path + "/ntu120/xsetup/val_data_joint.npy",
      'label_path': data_path + "/ntu120/xsetup/val_label.pkl",
      'num_frame_path': data_path + "/ntu120/xsetup/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_pku_part1_cross_subject():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
      "t_input_size":3200,
      "s_input_size":4096,
      "hidden_size":2048,
      "num_head":16,
      "num_class":51,
    }

    # feeder
    self.train_feeder_args = {
      "data_path": data_path + "/crossclr_pkummd/pku_part1/xsub/train_position.npy",
      "label_path": data_path + "/crossclr_pkummd/pku_part1/xsub/train_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': data_path + "/crossclr_pkummd/pku_part1/xsub/val_position.npy",
      'label_path': data_path + "/crossclr_pkummd/pku_part1/xsub/val_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_pku_part2_cross_subject():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
      "t_input_size":3200,
      "s_input_size":4096,
      "hidden_size":2048,
      "num_head":16,
      "num_class":51,
    }

    # feeder
    self.train_feeder_args = {
      "data_path": data_path + "/crossclr_pkummd/pku_part2/xsub/train_position.npy",
      "label_path": data_path + "/crossclr_pkummd/pku_part2/xsub/train_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': data_path + "/crossclr_pkummd/pku_part2/xsub/val_position.npy",
      'label_path': data_path + "/crossclr_pkummd/pku_part2/xsub/val_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }
