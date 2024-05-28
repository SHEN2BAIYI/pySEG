import multiprocessing as mp
import torch


if __name__ == '__main__':
    length = 2**15+1 # with 2**15 or less, it will be OK
    print("========== randperm in main process ==========")
    print("torch rand perm")
    print("randperm length: {}".format(length) )
    torch.randperm(length)

    def worker(rank, length):
        print("========== worker process {} ==========".format(rank) )
        print("torch rand perm")
        print("randperm length: {}".format(length) )
        torch.randperm(length) # stuck here, if length is greater than 2**15
        print("process finished")

    pool = []
    for i in range(2):
        w = mp.Process(
            target=worker,
            args = (i, length+i-1)
        )
        w.start()
        pool.append(w)
    for w in pool:
        w.join()
    print("done")