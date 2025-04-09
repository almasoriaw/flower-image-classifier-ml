import argparse

def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', type = str, default = 'flowers', help = 'path where the flowers images will be download')

    parser.add_argument('--arch', type = str, default = 'vgg13', choices = ['vgg11', 'alexnet', 'resnet', 'vgg13'],  help = 'CNN model architecture used for training')

    parser.add_argument('--save_dir', type = str, default = 'checkpoints/', help = 'Folder name where checkpoints will be saved')

    parser.add_argument('--save_name', type = str, default = 'model_checkpoint', help = 'Checkpoint file name')

    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'Learning rate for training')

    parser.add_argument('--epochs', type = int, default = 5, help = 'Number of epochs for training, selecting a big number will require higher training time')

    parser.add_argument('--print_every', type = int, default = 5, help = 'Select the number of steps to wait to print the training accuracy output')

    parser.add_argument('--freeze_parameters', type = bool, default = True, help = 'Select if CNN parameters should be freeze for training')

    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Number of hidden units for your CNN')

    parser.add_argument('--dropout', type = float, default = 0.2, help = 'Dropout normalization value for your CNN')

    parser.add_argument('--training_compute', type = str, default = 'cpu', help = 'Select if you want to use CPU or GPU for trianing')

    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Select path where the category names mapping file is located')

    parser.add_argument('--top_k', type = int, default = 3, help = 'Top probabilities to return from the inference process')

    parser.add_argument('--checkpoint', type = str, default = 'checkpoints/model_checkpoint', help = 'Select the location for the CNN checkpoint file')

    parser.add_argument('--image_path', type = str, default = 'image_06585.jpg', help = 'Select the location of the image you want to predict')

    return parser.parse_args()