"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task

from config import cfg
import torch


def main(cfg):

    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
        width_mult=cfg["width_mult"],
        mode='train')
    # print(model)
    # b


    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()
    data.showData(train_loader)
    # b


    # run_task = Task(cfg, model)
    # if (
    #     cfg["previous_saved_model"] is not None
    #     and len(cfg["previous_saved_model"]) > 0
    # ):
    #     run_task.modelLoad(cfg["previous_saved_model"])
    #     print(f'The model loaded:{cfg["previous_saved_model"]}')
    model_path = cfg.get("previous_saved_model", None)
    if model_path is not None and len(model_path) > 0:
        model.load_state_dict(torch.load(model_path), strict=False)
        print(f"The model loaded:{model_path}")
    run_task = Task(cfg, model)
    run_task.train(train_loader, val_loader)


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    main(cfg)