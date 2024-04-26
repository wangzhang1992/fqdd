import argparse

def parse_arguments():
    model_args = argparse.ArgumentParser(
     description='model config', add_help=True )
    model_args.add_argument(
          '--train_path',
          default='raw_data/train/',
          type=str,
          help='train data path')

    model_args.add_argument(
          '--test_path',
          default='raw_data/test',
          type=str,
          help='test data path')

    model_args.add_argument(
          '--dev_path',
          default='raw_data/dev',
          type=str,
          help='dev data path')

    model_args.add_argument(
          '--model_save_path',
          default='model/',
          type=str,
          help='save model path')

    model_args.add_argument(
          '--feat_type',
          default='mfcc',
          type=str,
          help=' extract feat type')

    model_args.add_argument(
        '--feat_cof',
        default='40',
        type=str,
        help=' extract feat cof')

    model_args.add_argument(
          '--epoch_n',
          default=100,
          type=int,
          help='train epoch')

    model_args.add_argument(
          '--batch_size',
          default=1,
          type=int,
          help='batch set')

    model_args.add_argument(
          '--label_status',
          default='word',
          type=str,
          help=' label status')

    model_args.add_argument(
          '--n_threads',
          default=1,
          type=int,
          help='set thread number reading data ')

    model_args.add_argument(
          '--lr',
          default=1e-3,
          type=float,
          help='set learning rate ')

    model_args.add_argument(
          '--lexicon_path',
          default='dict/char.txt',
          type=str,
          help='set lexicon path')

    return  model_args.parse_args() 
