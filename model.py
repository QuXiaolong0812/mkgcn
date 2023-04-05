import torch
import random
from neighbor_aggregator import NeighborAggregator
from multimodal_aggregator import MultiModalAggregator


# Multi-modal Knowledge Graph Convolutional Network for music
class MKGCN(torch.nn.Module):
    def __init__(self, args, num_user, num_entity, num_relation, kg, user_history_dict, multimodal_dict, device):
        super(MKGCN, self).__init__()
        self.num_user = num_user
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.n_neighbor
        self.kg = kg
        self.device = device
        self.modals = args.modals
        self.multimodal = args.multimodal

        self.entity_embedding = torch.nn.Embedding(num_entity, args.dim)
        self.relation_embedding = torch.nn.Embedding(num_relation, args.dim)

        self._generate_adj()
        self._generate_user_history(user_history_dict)
        self._generate_multimodal(multimodal_dict)

        self.neighbor_aggregator = NeighborAggregator( self.batch_size, self.dim, args.neighbor_aggregator )
        self.multimodal_aggregator = MultiModalAggregator(self.batch_size, self.dim, self.modals, args.multimodal_aggregator)

        if args.user_aggregator == 'multi-head':
            self.attention = torch.nn.MultiheadAttention( args.dim, num_heads=8 )
        elif args.user_aggregator == 'max-pool':
            self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.user_aggregator = args.user_aggregator

    # users, items are both indices, shape is [batch_size]
    def forward(self, users, items):
        batch_size = users.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        users = users.view((-1, 1))
        items = items.view((-1, 1))

        # [batch_size, n_neighbor]
        user_history_items = self._get_user_history_items(users)

        # entities {0:(batch_size, 1), 1:(batch_size, n_neighbors),...}
        # relations {0:(batch_size, n_neighbors), 1:(batch_size, n_neighbors*n_neighbors),...}
        entities, relations = self._get_neighbors( items )

        # [batch_size, dim]
        if self.user_aggregator == 'multi-head':
            user_embeddings = self.entity_embedding( user_history_items ).transpose( 0, 1 )
            attn_output, _ = self.attention( user_embeddings, user_embeddings, user_embeddings )
            user_embeddings = attn_output.mean( dim=0 )
        elif self.user_aggregator == 'max-pool':
            user_embeddings = self.entity_embedding(user_history_items).permute(0, 2, 1)
            user_embeddings = self.max_pooling(user_embeddings).squeeze(-1)
        elif self.user_aggregator == 'mean-pool':
            user_embeddings = torch.mean( self.entity_embedding( user_history_items ), dim=1 )
        else:
            raise Exception("Unknown user aggregator: " + self.user_aggregator)

        entity_embeddings = self._multimodal_aggregate(entities)

        relation_embeddings = [self.relation_embedding( relation ) for relation in relations]

        # [batch_size, dim]
        item_embeddings = self._neighbor_aggregate(user_embeddings, entity_embeddings, relation_embeddings)

        scores = (user_embeddings * item_embeddings).sum(dim=1)

        return torch.sigmoid(scores)

    # Generate adjacency matrix for all entities and relations
    def _generate_adj(self):
        self.adj_entity = torch.empty(self.num_entity, self.n_neighbor, dtype=torch.long)
        self.adj_relation = torch.empty(self.num_entity, self.n_neighbor, dtype=torch.long)

        for entity in self.kg:
            if len(self.kg[entity]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[entity], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[entity], k=self.n_neighbor)

            self.adj_entity[entity] = torch.LongTensor([entity for _, entity in neighbors])
            self.adj_relation[entity] = torch.LongTensor([relation for relation, _ in neighbors])

    # Generate user_history matrix for all users
    def _generate_user_history(self, user_history_dict):
        self.all_user_history = torch.empty(self.num_user, self.n_neighbor, dtype=torch.long)
        for userID, history_list in user_history_dict.items():
            if len(history_list) >= self.n_neighbor:
                neighbor = random.sample(history_list, self.n_neighbor)
            else:
                neighbor = random.choices(history_list, k=self.n_neighbor)
            self.all_user_history[userID] = torch.LongTensor(neighbor)

    # Generate multimodal features for all entities   return dit=ct(entityID, tensor[len(modals),dim])
    def _generate_multimodal(self, multimodal_dict):
        item_list = list(multimodal_dict.keys())
        all_entities_list = [i for i in range(self.num_entity)]
        entity_list = list(set(all_entities_list)-set(item_list))
        entity_list2tensor = torch.tensor(entity_list).unsqueeze(1)
        entity_list4embeddings = self.entity_embedding(entity_list2tensor)
        entity_list4embeddings = torch.broadcast_to(entity_list4embeddings, (len(entity_list), len(self.modals), self.dim))
        entity_list4embeddings = torch.split(entity_list4embeddings, 1, dim=0)
        entity_tensor_list = [tensor.detach().squeeze(0) for tensor in entity_list4embeddings]
        entity_tensor_dict = {key: value for key, value in zip(entity_list, entity_tensor_list)}
        item_tensor_dict = {key: torch.tensor(multimodal_dict[key]).reshape(len(self.modals), self.dim) for key in item_list}
        self.multimodal_tensor_dict = dict()
        self.multimodal_tensor_dict.update( item_tensor_dict )
        self.multimodal_tensor_dict.update(entity_tensor_dict)

    # get n_iter neighbors and relations for item
    def _get_neighbors(self, items):
        entities = [items]
        relations = []

        for iter in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_entity[entities[iter]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_relation[entities[iter]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    # get user's historical items [batch_size, n_neighbors]
    def _get_user_history_items(self, users):
        user_history_items = torch.LongTensor(self.all_user_history[users]).view((self.batch_size, -1)).to(self.device)
        return user_history_items

    # get multimodal vector by entity indices
    def _multimodal_aggregate(self, entities):
        entities_embeddings = [self.entity_embedding( entity ) for entity in entities]
        if self.multimodal:
            pass
        else:
            return entities_embeddings
        multimodal_aux_embeddings = []
        for tensor in entities:
            tensor = tensor.cpu()
            batch_size, num = tensor.size()
            result = torch.zeros((batch_size, num, len(self.modals), self.dim), dtype=torch.float32, device=self.device)

            for i in range(batch_size):
                for j in range(num):
                    key = int(tensor[i][j].item())
                    result[i][j] = self.multimodal_tensor_dict[key]

            multimodal_aux_embeddings.append(result)
        entities_with_mm_embeddings = []
        for entities_embedding, multimodal_aux_embedding in zip(entities_embeddings, multimodal_aux_embeddings):
            # entities_with_mm_embedding  -> torch.size([batch_size, num, 8, dim])
            entities_with_mm_embedding = torch.cat([entities_embedding.unsqueeze(2), multimodal_aux_embedding], dim=2)
            entities_with_mm_embeddings.append(entities_with_mm_embedding)

        entities_combine_mm_embeddings = []
        for entities_embedding, entities_combine_mm_embedding in zip(entities_embeddings, entities_with_mm_embeddings):
            entities_combine_mm_embeddings.append(self.multimodal_aggregator(entities_embedding, entities_combine_mm_embedding))

        # remember: return is a list! ->[(batch_size, 1, dim), (batch_size, 1*n_neighbor, dim),...]
        return entities_combine_mm_embeddings

    # Make item embeddings by aggregating neighbor vectors
    def _neighbor_aggregate(self, user_embeddings, entity_embeddings, relation_embeddings):

        for iter in range(self.n_iter):
            if iter == self.n_iter - 1:
                act_fn = torch.tanh
            else:
                act_fn = torch.sigmoid

            entity_vectors_next_iter = []

            for hop in range(self.n_iter - iter):
                vector = self.neighbor_aggregator(
                    self_vectors=entity_embeddings[hop],
                    neighbor_vectors=entity_embeddings[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_embeddings[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act_fn=act_fn
                )
                entity_vectors_next_iter.append(vector)
            entity_embeddings = entity_vectors_next_iter

        return entity_embeddings[0].view((self.batch_size, self.dim))