#用于获取预训练（intra）数据集
def get_pretraining_set_intra(opts):

    from feeder.feeder_pretraining import Feeder
    training_data = Feeder(**opts.train_feeder_args)

    return training_data

#用于获取蒸馏（distill）数据集
def get_distill_set_intra(opts):

    from feeder.feeder_distill import Feeder
    training_data = Feeder(**opts.train_feeder_args)

    return training_data

#用于获取蒸馏（distill）训练集
def get_distill_training_set(opts):

    from feeder.feeder_downstream import Feeder

    data = Feeder(**opts.train_feeder_args_test)

    return data
#用于获取蒸馏（distill）验证集
def get_distill_validation_set(opts):

    from feeder.feeder_downstream import Feeder

    data = Feeder(**opts.test_feeder_args)

    return data

#用于获取微调（finetune）训练集
def get_finetune_training_set(opts):

    from feeder.feeder_downstream import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data

#用于获取微调验证集
def get_finetune_validation_set(opts):

    from feeder.feeder_downstream import Feeder
    data = Feeder(**opts.test_feeder_args)

    return data

#用于获取半监督（semi）训练集
def get_semi_training_set(opts):

    from feeder.feeder_semi import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data
