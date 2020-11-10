from tqdm.contrib.concurrent import process_map 
# from multiprocessing import Pool

def basic(x):
	return x
if __name__ == '__main__':
	# pool = Pool(5)
	process_map(basic,[1,1,1])
	# print("X")
	# print(pool.map(basic, [1,1,1]))
