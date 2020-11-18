# for img, vec_label in img_generator:
		# ckpt.append(executor.submit(augment_img, img, vec_label, n_replicates))

	wait(ckpt, return_when = ALL_COMPLETED)
	r = [x.result() for x in ckpt]
	print(f"Augmentation time: {round(time.time()-st,2)}s")
	# 	replicated_data = np.asarray([ img for i in range(n_replicates)])
	# 	images_aug = seq.augment_images( replicated_data )

	# 	all_vec_labels = np.concatenate( (all_vec_labels, [vec_label]*n_replicates))

	# 	for i in range(n_replicates):
	# 		im = Image.fromarray(images_aug[i])
	# 		im.save( f"../data/train/augmented/{count}.png")
	# 		count += 1
		
	#all_vec_labels =  np.delete(all_vec_labels,(0),axis = 0)
	#print("Saving labels")
	#np.savetxt("../imgs/aug_imgs/aug_vec_labels.dat", all_vec_labels, fmt = "%d")




	
# def yield_files(class_vec_map):
# 	# Deprecated file read using yield
# 	# It's a nice idea to have this and iteratively grab shit but it's a pain for parallel processing as far as I can tell
# 	for i in os.listdir("../data/train/raw_resized"):
# 		if i.endswith('.png'):
# 			label =  "_".join(i[:-4].split("_")[:4])
# 			im = np.asarray(Image.open(f"../data/train/raw_resized/{i}")).astype(np.uint8)
# 			yield [im, class_vec_map[label]]
