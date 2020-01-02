import argparse
from DenseDesc.train.trainer import Trainer
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train')
parser.set_defaults(detector_check=False)
args = parser.parse_args()


def train_model():
    trainer = Trainer()
    trainer.train_model()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    name2func = {
        'train': train_model,
    }
    name2func[args.task]()
