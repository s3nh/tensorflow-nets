from src.utils import _get_data_info, read_config
from src.read_data  import data_gen, get_structured_dataset, split_dataset 
from models.models import load_model
from models.models import _get_compile 

def main():
    config = read_config()
    model = load_model()
    print(model.summary())
    lr, metrics, loss, optim  = _get_compile(model, config)
