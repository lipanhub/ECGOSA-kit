from src.experiment.evaluation import final_test_LeNet5, final_test_without_discarded_segment
from src.experiment.preprocess import load_zsfy_preprocessed_data
from src.experiment.training import train_classifier
from src.experiment.util import setup_gpu, make_log_dir

selected_gpu_devices = '0'


def func():
    x_train, x_train_5min, y_train, x_val, x_val_5min, y_val, x_test, x_test_5min, y_test, groups_test = load_zsfy_preprocessed_data()

    setup_gpu(selected_gpu_devices)
    log_dir = make_log_dir()

    train_classifier(log_dir, x_train, x_train_5min, y_train, x_val, x_val_5min, y_val)

    final_test_without_discarded_segment(log_dir, x_test, x_test_5min, y_test, groups_test)


if __name__ == '__main__':
    func()
