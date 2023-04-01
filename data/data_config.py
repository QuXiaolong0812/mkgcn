# This is a data configuration
# if the rating great than or equal the threshold, then it will be regarded as positive sample
data_config = {
			'music-d': {
				'itemID2entityID_path': 'data/music-d/itemID2entityID.txt',
				'itemID2entityID_sep': '\t',
				'kg_path': 'data/music-d/kg.txt',
				'kg_sep': '\t',
				'userID2itemID4ratings_path': 'data/music-d/userID2itemID4ratings.tsv',
				'userID2itemID4ratings_sep': '\t',
				'threshold': 5.0,
				'modals_path': 'data/music-d/modals/modal_{}.tsv',
				'modals_sep': '\t'
			}
}
