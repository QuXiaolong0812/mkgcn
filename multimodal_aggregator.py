import torch


class MultiModalAggregator(torch.nn.Module):
	def __init__(self, batch_size, dim, modals, multimodal_aggregator):
		super(MultiModalAggregator, self).__init__()
		self.batch_size = batch_size
		self.dim = dim
		self.modals = modals
		if multimodal_aggregator == 'concat':
			self.weights = torch.nn.Linear( 2 * self.dim, self.dim, bias=True )
		else:
			self.weights = torch.nn.Linear( self.dim, self.dim, bias=True )
		self.multimodal_aggregator = multimodal_aggregator

	# ['max', 'mean', 'concat', 'sum', 'residual']
	def forward(self, entities_embedding, entities_with_mm_embedding):
		batch_size = entities_embedding.size( 0 )
		if batch_size != self.batch_size:
			self.batch_size = batch_size

		if self.multimodal_aggregator == 'sum':
			# entities_with_mm_embedding sum:[batch_size, num, 8, dim] -> [batch_size, num, dim]
			output = torch.sum(entities_with_mm_embedding, dim=2, keepdim=False)
		elif self.multimodal_aggregator == 'mean':
			output = torch.mean(entities_with_mm_embedding, dim=2, keepdim=False)
		elif self.multimodal_aggregator == 'max':
			output, _ = torch.max(entities_with_mm_embedding, dim=2, keepdim=False)
		elif self.multimodal_aggregator == 'concat':
			output = torch.mean(entities_with_mm_embedding, dim=2, keepdim=False)
			output = torch.cat( (entities_embedding, output), dim=-1 )
			output = output.view( (-1, 2 * self.dim) )
		elif self.multimodal_aggregator == 'residual':
			output = torch.mean( entities_with_mm_embedding, dim=2, keepdim=False )
			output = (entities_embedding + output).view( (-1, self.dim) )
		else:
			raise Exception( "Unknown entity-multimodal aggregator: " + self.neighbor_aggregator )
		output = self.weights(output)
		return torch.tanh(output.view((self.batch_size, -1, self.dim)))



