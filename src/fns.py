
def my_acc( v1, v2 ):
  count = 0.
  n = len(v1)
  for i in range(n):
    if v1[i] == v2[i]:
      count = count + 1.
  return count/(1.*n)

