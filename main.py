from src.utils import _get_data_info
from src.read_data  import data_gen, get_structured_dataset, split_dataset
from models.models import load_model

def main():
    model = load_model()
    print(model.summary())
