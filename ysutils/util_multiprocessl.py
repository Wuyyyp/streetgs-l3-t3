from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import os
import tqdm
def process_task(func, paralist, pool='process'):
    num_threads = int(os.cpu_count()) * 4
    num_threads = min(num_threads, os.cpu_count() - 1)
    num_threads = max(num_threads, 1)
    fs = []
    if pool == 'process':
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for i in range(len(paralist)):
                result = executor.submit(func, *paralist[i])
                fs.append(result)

    if pool == 'thread':
        with ThreadPoolExecutor(max_workers=os.cpu_count()*4) as executor:
            for i in range(len(paralist)):
                result = executor.submit(func, *paralist[i])
                fs.append(result)
    for i in range(len(fs)):
        fs[i] = fs[i].result()
    return fs
