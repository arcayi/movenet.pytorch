"""
@Fire
https://github.com/fire717
"""
import os
import random
import pandas as pd   

from lib import init, Data, MoveNet, Task

from config import cfg





def main(cfg, position=0):

    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    
    
    data = Data(cfg)
    test_loader = data.getTestDataloader()


    run_task = Task(cfg, model)
    run_task.modelLoad("output/e91_valacc0.79763.pth")


    run_task.predict(test_loader, "output/predict", writefile=False, position=position)



# if __name__ == '__main__':
#     main(cfg)
    
    
from multiprocessing.pool import Pool
import time
import concurrent.futures
import torch


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    NUM = 4
    NUMBERS = range(NUM)
    print("multiprocessing.pool.Pool:\n")
    start = time.time()

    l = []
    pool = Pool(NUM)

    # for num in NUMBERS:
    #     pool.apply_async(main)
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出


    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM) as executor:
        future_to_num = {executor.submit(main, cfg=cfg, position =num): num for num in NUMBERS}
        for future in concurrent.futures.as_completed(future_to_num):
            num = future_to_num[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (num, exc))
            else:
                print(f"{num}`s result is {data}")

    print("COST: {}".format(time.time() - start))
