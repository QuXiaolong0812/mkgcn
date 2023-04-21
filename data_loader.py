import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data import data_config
from utils import reduce_dimensions, tenet_itemID2entityID


class MKGCNDataset(Dataset):
	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len( self.dataset )

	def __getitem__(self, idx):
		userID = np.array( self.dataset.iloc[idx]['userID'] )
		itemID = np.array( self.dataset.iloc[idx]['itemID'] )
		label = np.array( self.dataset.iloc[idx]['label'], dtype=np.float32 )
		return userID, itemID, label


# Data Loader class which makes dataset for training / knowledge graph dictionary
class MKGCNDataLoader:
	def __init__(self, args):
		self.dataset_config = data_config.data_config

		self.itemID2entityID = pd.read_csv( self.dataset_config[args.dataset]['itemID2entityID_path'],
		                                    sep=self.dataset_config[args.dataset]['itemID2entityID_sep'],
		                                    header=None, names=['itemID','entityID'], skiprows=1 )
		self.kg = pd.read_csv( self.dataset_config[args.dataset]['kg_path'], sep=self.dataset_config[args.dataset]['kg_sep'],
		                       header=None, names=['head', 'relation', 'tail'], skiprows=1)   # head, tail are entity id not item id
		self.userID2itemID4ratings = pd.read_csv( self.dataset_config[args.dataset]['userID2itemID4ratings_path'],
		                                          sep=self.dataset_config[args.dataset]['userID2itemID4ratings_sep'],
		                                          names=['userID', 'itemID', 'rating'], skiprows=1 )

		# item_index2entity_id['itemID'], rating['itemID'] both represents old item id
		self.userID2itemID4ratings = self.userID2itemID4ratings[self.userID2itemID4ratings['itemID'].isin( self.itemID2entityID['itemID'] )]
		# reset index
		self.userID2itemID4ratings.reset_index( inplace=True, drop=True )

		self.user_encoder = LabelEncoder()
		self.entity_encoder = LabelEncoder()
		self.relation_encoder = LabelEncoder()

		self._build_num_info()
		self._build_dataset(args)
		self._build_negative_sampling(args)

	# Fit each label encoder and encode knowledge graph
	def _build_num_info(self):
		print( 'Build dataset num info ...', end=' ' )
		self.user_encoder.fit( self.userID2itemID4ratings['userID'] )
		# itemID2entityID['entity_id'] and kg[['head', 'tail']] represents new entity ID
		self.entity_encoder.fit(pd.concat([self.itemID2entityID['entityID'],
		                                   self.kg['head'], self.kg['tail']]))
		self.relation_encoder.fit(self.kg['relation'])
		self.kg['head'] = self.entity_encoder.transform( self.kg['head'] )
		self.kg['tail'] = self.entity_encoder.transform( self.kg['tail'] )
		self.kg['relation'] = self.relation_encoder.transform( self.kg['relation'] )
		self.num_user = len(self.user_encoder.classes_)
		self.num_item = len( set( self.itemID2entityID['itemID'] ) )
		self.num_entity = len(self.entity_encoder.classes_)
		self.num_relation = len(self.relation_encoder.classes_)
		print( 'Done' )

	# Build dataset for training (rating data);It contains negative sampling process
	def _build_dataset(self, args):
		print( 'Build dataset dataframe ...', end=' ' )
		# userID2entityID4rating update to dataset
		self.dataset = pd.DataFrame()
		self.dataset['userID'] = self.user_encoder.transform( self.userID2itemID4ratings['userID'] )
		# update rating.tsv itemID to entityID
		itemID2entityID_dict = dict( zip( self.itemID2entityID['itemID'], self.itemID2entityID['entityID'] ) )
		# get all item's entityID
		self.full_item_set = set( self.userID2itemID4ratings['itemID'].apply( lambda x: itemID2entityID_dict[x] ) )
		self.userID2itemID4ratings['itemID'] = self.userID2itemID4ratings['itemID'].apply( lambda x: itemID2entityID_dict[x] ) # make ratings.tsv itemID to entityID
		self.dataset['itemID'] = self.entity_encoder.transform( self.userID2itemID4ratings['itemID'] )
		self.dataset['label'] = self.userID2itemID4ratings['rating'].apply( lambda x: 0 if x < self.dataset_config[args.dataset]['threshold'] else 1 )
		# keeping positive samples
		self.dataset = self.dataset[self.dataset['label'] == 1]
		print('Done')

	# generate negative samplings to dataset
	def _build_negative_sampling(self, args):
		print( 'Build negative sampling ...', end=' ' )
		user_list = []
		item_list = []
		label_list = []
		if args.sampling_strategy == 'proportional':
			# 统计物品流行度
			item_pop = self.dataset['itemID'].value_counts()
			# 计算物品采样概率
			alpha = 1
			item_prob = {item: 1 / (pop ** alpha) for item, pop in item_pop.items()}
			# 对概率值进行归一化处理
			item_prob = {k: v / sum( item_prob.values() ) for k, v in item_prob.items()}
		for user, group in self.dataset.groupby( ['userID'] ):
			item_set = set( group['itemID'] )
			if args.sampling_strategy == 'proportional':
				negative_sampled = np.random.choice( list( item_prob.keys() ), size=len( item_set ), replace=False,
				                                     p=list( item_prob.values() ) )
			else:
				negative_set = self.full_item_set - item_set
				negative_sampled = random.sample(negative_set, len(item_set))
			user_list.extend( [user] * len( negative_sampled ) )
			item_list.extend( negative_sampled )
			label_list.extend( [0] * len( negative_sampled ) )
		negative = pd.DataFrame( {'userID': user_list, 'itemID': item_list, 'label': label_list} )
		self.dataset = pd.concat( [self.dataset, negative] )
		self.dataset = self.dataset.sample( frac=1, replace=False, random_state=999 )
		self.dataset.reset_index( inplace=True, drop=True )
		print('Done')

	# get user history dict
	def load_user_history_dict(self):
		user_history_dict = dict()
		for index in self.dataset.index:
			user = self.dataset.loc[index].values[0]
			item = self.dataset.loc[index].values[1]
			if user not in user_history_dict:
				user_history_dict[user] = []
			user_history_dict[user].append( item )
		return user_history_dict

	# get KG triples Bi-directional; Knowledge graph is dictionary form; 'head': [(relation, tail), ...]
	def load_knowledge_graph(self):
		print( 'Loading knowledge graph ...', end=' ' )
		kg = dict()
		for i in range( len( self.kg ) ):
			head = self.kg.iloc[i]['head']
			relation = self.kg.iloc[i]['relation']
			tail = self.kg.iloc[i]['tail']
			if head in kg:  # head is key in dict() named kg
				kg[head].append( (relation, tail) )
			else:
				kg[head] = [(relation, tail)]
			if tail in kg:
				kg[tail].append( (relation, head) )
			else:
				kg[tail] = [(relation, head)]
		print( 'Done' )
		return kg

	# get user, entity and relation quantities
	def load_dataset_info(self):
		print( 'Loading dataset info ... Done')
		return len(self.user_encoder.classes_), self.num_item, len(self.entity_encoder.classes_), len(self.relation_encoder.classes_)

	# get train_loader and test_loader
	def load_dataset(self, args):
		print( 'Loading data loader ...', end=' ' )
		dataset = self.dataset
		train_dataset, test_dataset, train_label_dataset, test_label_dataset = train_test_split( dataset, dataset['label'], test_size=1 - args.ratio,
                                                     shuffle=False, random_state=999 )
		train_mkgcn_dataset = MKGCNDataset(train_dataset)
		test_mkgcn_dataset = MKGCNDataset(test_dataset)
		train_loader = DataLoader(train_mkgcn_dataset, batch_size=args.batch_size)
		test_loader = DataLoader(test_mkgcn_dataset, batch_size=args.batch_size)
		print('Done')
		return train_loader, test_loader, train_dataset, test_dataset

	# get multimodal vector list
	def load_modals_dict(self, args):
		print('Loading modals ... ', end=' ')
		modals = dict()
		for i in range(len(args.modals)):
			with open(self.dataset_config[args.dataset]['modals_path'].format(str(i)), 'r') as f:
				for line in f:
					line = line.strip().split(self.dataset_config[args.dataset]['modals_sep'])
					item_id = line[0]
					features = list(map(float, line[1:]))
					if item_id not in modals:
						modals[item_id] = [ [] for _ in range(len(args.modals))]
					modals[item_id][i] = features
		if args.dim < 300:
			reduce_dim_modals = reduce_dimensions(modals, args.dim, args.modals)
		else:
			raise Exception( "Too large embedding dimensions for multimodal: " + args.dim )
		tenet_modals = tenet_itemID2entityID(reduce_dim_modals, self. entity_encoder,self.dataset_config[args.dataset]['itemID2entityID_path'],
		                                    sep=self.dataset_config[args.dataset]['itemID2entityID_sep'])
		print('Done')
		return tenet_modals

