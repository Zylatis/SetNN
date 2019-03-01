from itertools import islice
import copy
def my_acc( v1, v2 ):
  count = 0.
  n = len(v1)
  for i in range(n):
    if v1[i] == v2[i]:
      count = count + 1.
  return count/(1.*n)

def make_batches( x, y, batch_size ):
	batches = []
	ndata = len(x)

	for i in range(0,ndata, batch_size):
		batches.append( [ x[i:i+batch_size], y[i:i+batch_size] ] )

	return batches