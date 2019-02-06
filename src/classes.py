import numpy as np 
import itertools
	
def get_labels():	
	colours = ['red', 'purple', 'green']
	counts = ['single','triple','double']
	shape = ['pill', 'diamond', 'squiggle']
	fill = ['empty', 'grid', 'solid']


	features = [colours,counts, fill,shape]
	all_labels = list(itertools.product(*features))

	class_map = {}
	inverse_class_map = {}
	i = 0
	for label in all_labels:
		concat = "_".join( label )
		class_map[concat] = i 
		inverse_class_map[i] = concat
		i = i+1

	return class_map, inverse_class_map