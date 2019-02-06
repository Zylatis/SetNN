import numpy as np 
import itertools

colours = ['red', 'purple', 'green']
counts = ['single','triple','double']
shape = ['pill', 'diamond', 'squiggle']
fill = ['empty', 'grid', 'solid']


features = [colours,counts, fill,shape]
all_labels = list(itertools.product(*features))

class_map = {}

for label in all_labels:
	concat = "_".join( label )
	print concat