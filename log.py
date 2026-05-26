import os
def create_logger(log_filename, display=True):
    f = open(log_filename, 'a') # 追加模式，即可以不断地写，但写完后都要加一句f.close，否则不会真的写上去
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text) # 不仅写入到文件中，还可以直接在控制台输出
        f.write(text + '\n') # 写入文件
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush() # 每写10次会刷新文件缓存并强制写入磁盘，而不是等到f.close后再统一写入磁盘
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close
# 返回logger和f.close供外部调用，不必一定通过create_logger