import numpy as np 
import itertools

class_labels = {
	'colours':['red', 'purple', 'green'],
	'counts': ['single','double', 'triple'],
	'fill' : ['empty', 'grid', 'solid'],
	'shape' : ['pill', 'diamond', 'squiggle']
}

def get_labels():	

	features = [colours,counts, fill,shape]
	all_labels = list(itertools.product(*features))

	class_map = {}
	# inverse_class_map = {}
	class_vec_map = {}
	i = 0
	for label in all_labels:
		concat = "_".join( label )
		class_map[concat] = i 
		# inverse_class_map[i] = concat
	
		class_vec_map[concat] =  [
			class_labels['colours'].index(label[0]),
			class_labels['counts'].index(label[1]),
			class_labels['fill'].index(label[2]),
			class_labels['shape'].index(label[3])
		]
		i = i+1

	return class_map, class_vec_map