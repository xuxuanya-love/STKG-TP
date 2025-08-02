import torch
import torch.nn as nn
#from  torch.nn import MultiheadAttention
import copy
import math




# Recognition Network
class RecognitionNetwork(nn.Module):
    '''
    num_layers:  The number of GNN Layers
    embed_dim: STSL_embedding dimension
    hidden_size:STSL_hidden size
    num_heads:
    ...
    Input of the network:
    STKG_embed_vector:
    A:

    Output of the network:
    z:


    '''

    def __init__(self, num_layers, embed_dim, hidden_size, num_heads, attention_prob_dropout_prob, dropout_rate):
        super(RecognitionNetwork, self).__init__()

        self.gnn = GNN(num_layers, embed_dim, hidden_size, num_heads, attention_prob_dropout_prob, dropout_rate)
        self.gcn = GCN(embed_dim, hidden_size, embed_dim, dropout_rate)
        self.AttentionLayer = AttentionLayer(embed_dim, num_heads)
        self.Dense = DenseLayer(input_dim=embed_dim, output_dim=embed_dim)
        self.Dense1 = DenseLayer(input_dim=128, output_dim=embed_dim)
    def forward(self, STSL_embed_vector,STSL_adj):
        STSL_embed_vector = self.Dense1(STSL_embed_vector)
        STSL_embed_vector = self.gcn(STSL_adj, STSL_embed_vector)
        print(STSL_embed_vector.shape)
        hidden_state = self.gnn(STSL_embed_vector)
        hidde_state, attn = self.AttentionLayer(hidden_state)
        # print(attn.shape)
        # print(hidde_state.shape)
        z = self.Dense(hidde_state)

        return z,attn










class GNN(nn.Module):
    def __init__(self,num_layers,embed_dim,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate):
        super(GNN, self).__init__()
        self.n_layers = num_layers
        self.LayerNorm = nn.LayerNorm(embed_dim)
        layers = GATlayers (hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate)
        self.gnn_layers = nn.ModuleList([copy.deepcopy(layers) for _ in range (num_layers)])


    def forward(self,graph_vector):
        graph_vector = self.gnn_layers[0](graph_vector)
        for i in range(1,self.n_layers,1):
            graph_vector = self.gnn_layers[i](graph_vector)

        return graph_vector







class GATlayers(nn.Module):
    def __init__(self,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate):
        super(GATlayers,self).__init__()
        self.att_layer = SelfAttention(hidden_size,num_heads,attention_prob_dropout_prob)
        self.Output = GATOutput(hidden_size,dropout_rate)

    def forward(self,graph_vector):

        attn_prob,graph_vector = self.att_layer(hidden_states=graph_vector)
        #print(attn_prob.shape)



        graph_vector = self.Output(graph_vector)

        return graph_vector







class AttentionLayer(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super( AttentionLayer, self ).__init__()
        self.attention = nn.MultiheadAttention(embed_dim,num_heads)

    def forward(self,hidden_state):
        context_vector,attn = self.attention(query=hidden_state,key=hidden_state,value=hidden_state)
        return context_vector,attn





class SelfAttention(nn.Module):
    def __init__(self,hidden_size,num_heads,attention_prob_dropout_prob,output_attention=False,keep_multi_head = False):
        super(SelfAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, num_heads))
        self.att_drop = True
        self.output_attention = output_attention
        self.keep_multi_head = keep_multi_head

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size/num_heads)
        self.all_head_size = self.num_attention_heads*self.attention_head_size

        self.query = nn.Linear(hidden_size,self.all_head_size)
        self.key = nn.Linear(hidden_size,self.all_head_size)
        self.value = nn.Linear(hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(attention_prob_dropout_prob)
        self.do_softmax = True

    def transpose_for_score(self,x):
        new_x_shape = x.size()[:-1]+(self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self,hidden_states):
        batch_size = hidden_states.shape[0]
        query_layers = self.query(hidden_states)
        key_layers = self.key(hidden_states)
        value_layers = self.value(hidden_states)


        query_layers =self.transpose_for_score(query_layers)
        key_layers =self.transpose_for_score(key_layers)
        value_layers = self.transpose_for_score(value_layers)

        attention_score = torch.matmul(query_layers,key_layers.transpose(-1,-2))
        attention_score = attention_score/math.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_score)
        attention_probs = self.dropout(attention_probs)
        context_layers =torch.matmul(attention_probs,value_layers)



        context_layers = context_layers.permute(0,2,1,3).contiguous()
        #Do  you need to shape reverse
        #print(self.all_head_size)
        #new_context_layer_shape = context_layers.size()[:2] + (self.all_head_size,)
        context_layers = context_layers.view((batch_size,62,64))#(*new_context_layer_shape)
        #print(context_layers.shape)


        return attention_probs,context_layers


class GATOutput(nn.Module):
    def __init__(self,hidden_size,dropout_rate):
        super(GATOutput,self).__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
        self.LayerNorm =  nn.LayerNorm(hidden_size,eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,hidden_state):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.relu(self.LayerNorm(hidden_state))
        hidden_state = self.LayerNorm(hidden_state)
        output = self.dropout(hidden_state)
        return output








class DenseLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        output = self.fc(x)
        return output




class MultiAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiAttention, self).__init__()

        self.num_heads = num_heads
        self.attention_size = hidden_size // num_heads
        self.scale = self.attention_size ** -0.5

        self.query_layer = nn.Linear(hidden_size, num_heads * self.attention_size)
        self.key_layer = nn.Linear(hidden_size, num_heads * self.attention_size)
        self.value_layer = nn.Linear(hidden_size, num_heads * self.attention_size)

        self.dropout_layer = nn.Dropout(attention_dropout_rate)
        self.ouput_layer = nn.Linear(num_heads * self.attention_size, hidden_size)

    def forward(self, q, k, v, attention_bias=None):
        orig_q_size = q.size()

        d_k = self.attention_size
        d_v = self.attention_size
        batch_size = q.size(0)

        q = self.query_layer(q).view(batch_size, 1, self.num_heads, d_k)
        k = self.key_layer(k).view(batch_size, 1, self.num_heads, d_k)
        v = self.value_layer(v).view(batch_size, 1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)

        # Attention_score computing
        q = q * self.scale
        x = torch.matmul(q, k)
        if attention_bias is not None:
            x = x + attention_bias
        x = torch.softmax(x, dim=3)
        x = self.dropout_layer(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.ouput_layer(x)
        assert x == orig_q_size
        return x








num_layers =3
embed_dim = 128
hidden_size =128
num_heads =8
attention_prob_dropout_prob =0.1
dropout_rate =0.1
input_dim = 128
output_dim =128






#graph_vector = torch.randn(1,128,128)
#att_scores = torch.randn(1,128,128)
#attention = SelfAttention(hidden_size,num_heads,attention_prob_dropout_prob)
#attn_prob,graph_vector = attention(graph_vector,att_scores)
#gnn = GNN(num_layers=3,embed_dim=128,hidden_size=128,num_heads=8,attention_prob_dropout_prob=0.1,dropout_rate=0.1)

#graph_vector = gnn(graph_vector,att_scores)
#print(attn_prob.shape)
#print(graph_vector)
#model = RecognitionNetwork(num_layers,embed_dim,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate,input_dim,output_dim)
#z = model(graph_vector,att_scores)
#print(z)