def yield_files():
	for i in os.listdir( imgs_folder + "processed/"):
    if i.endswith('.png'):
        label =  ("_").join(  (i[:-4]).split('_')[2:6] ).strip()
        class_val = class_map[label]
        classes_seen.append(class_val)
        im = np.asarray(Image.open( imgs_folder + "processed/"+str(i) )).astype(np.uint8)
        yield [im, class_val]

imgs_folder = "../imgs/"
labeled_data = []
class_map, inverse_class_map = classes.get_labels()
classes_seen = []
# can think about best way to represent output: do we make each card unique
# or fit on subfeatures as a vector output (colour etc)

if len(labeled_data) == 0:
    print("Run resize.py first, dumbarse.")
