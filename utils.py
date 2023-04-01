import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def args_print(args):
	print(args)
	if args.multimodal:
		modal_list = ['lyrics_word2vec', 'mfcc_bow', 'chroma_bow', 'emobase_bow', 'essentia', 'lyrics_sentiment', 'resnet']
		print("The modals have: ", end='')
		for i in args.modals:
			print(modal_list[i], end=',')
		print()
	current_time = datetime.datetime.now()
	print("model training start at time: " + current_time.strftime("%Y-%m-%d %H:%M:%S"))

def reduce_dimensions(modal_dict, dim):
	pca = PCA(n_components=dim)
	modal_list = np.array(list(modal_dict.values()))
	reduced_modal_list = []
	for i in range(7):
		reduced_modal_list.append(pca.fit_transform(modal_list[:, i, :]))
	reduced_modal_list = np.transpose(reduced_modal_list, (1, 0, 2))
	modal_dict_new = dict.fromkeys(modal_dict, None)
	for key in modal_dict_new:
		modal_dict_new[key] = reduced_modal_list[0]
	return modal_dict_new

def tenet_itemID2entityID(modal_dict, entity_encoder, filepath, sep):
	itemID2entityID = pd.read_csv(filepath, sep=sep, names=['itemID', 'entityID'], header=None, skiprows=1)
	new_modal_dict = {itemID2entityID.loc[itemID2entityID['itemID'] == k, 'entityID'].values[0]: v for k, v in modal_dict.items()}
	modal_dict = new_modal_dict
	keys_list = list(modal_dict.keys())
	value_list = list(modal_dict.values())
	encoded_list = entity_encoder.transform(keys_list)
	modal_dict = dict(zip(encoded_list, value_list))
	return modal_dict
