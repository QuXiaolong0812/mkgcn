import torch
import torch.nn.functional as F


# Aggregator mode in ['sum', 'concat', 'neighbor'] perhaps 'mean'
class NeighborAggregator( torch.nn.Module ):
    def __init__(self, batch_size, dim, neighbor_aggregator):
        super( NeighborAggregator, self ).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if neighbor_aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.neighbor_aggregator = neighbor_aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act_fn):
        # return samples num
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        neighbors_agg = self._agg_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        if self.neighbor_aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
        elif self.neighbor_aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
        elif self.neighbor_aggregator == 'neighbor':  # neighbors
            output = neighbors_agg.view((-1, self.dim))
        else:
            raise Exception("Unknown entity aggregator: " + self.neighbor_aggregator)

        output = self.weights(output)
        return act_fn(output.view((self.batch_size, -1, self.dim)))

    # aggregate neighbor vectors
    # Compute the relation scores between users and neighbor nodes using self-attention mechanism.
    def _agg_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated
