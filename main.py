from src.utils import _get_data_info, read_config
from src.read_data  import data_gen, get_structured_dataset, split_dataset 
from models.models import load_model
from models.models import _get_compile 



def main():
    #CLASS NAMES AND number of images definition 
    DATA_PATH = 'data/'
    CLASS_NAMES, image_count = _get_data_info(DATA_PATH)

    BATCH_SIZE = 32
    INITIAL_EPOCHS = 10 
    steps_per_epoch = round(int(image_count*0.7))//BATCH_SIZE 
    validation_steps = 20

    # Data preparation part 

    files = data_gen(path = DATA_PATH)
    # Get labeled dataset in tf.data.Dataset format 
    labeled_ds = get_structured_dataset(files)
    train_dataset, test_dataset = split_dataset(labeled_ds, image_count)
    config = read_config(path = 'config/config.yaml')
    model = load_model()
    print(model.summary())
    lr, metrics, loss, optim  = _get_compile(model, config)
    model.compile(optimizer = optim, loss = loss, 
    metrics = metrics)

    history = model.fit(train_dataset,
        epochs=initial_epochs,
        validation_data=test_dataset)


if __name__ == "__main__":
    main()