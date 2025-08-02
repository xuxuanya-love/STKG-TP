'''
      This py doc is aim to achieve  the Graph Transformer Encoder Part of our SYN module.
      The function of GraphTransformerEncoder take the embedding of KnowledgeGraph as input :
          STKG_Node_Feature: (Node_Num,Embed_dim)
          STKG_In_Degree:(Node_Num,1)
          STKG_Out_Degree:(Node_Num,1)
          STKG_spatial_pos:(Node_Num,Node_Num)
          graph_attn_bias:[optional]  I don't know   how does it work
          First,we add the centrality embedding to the node feature to capture the node position information.Specificially,we

          distribute eacch degree with a learnable embedding vector .By using the centrality embeding in the input ,the softmax
          attention can capture both the node informance and the graph  contextual correlation .
          Second ,we build it upon the standard Transformer Architectural,which consist of the multi-attention and FeedForward Network
          we can get a global encoder  hidden state of graph .
      Output of GraphTransformereEncoder:
          Graph_Contextual_vector:  Embeded with Graph Contextual Information
'''

import torch
import torch.nn as nn

# Encoder
class GraphTransformerEncoder(nn.Module):
    def __init__(self,
                 dropout_rate: float,
                 STKG_num,
                 embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 ffn_size,
                 num_layer
                 ):
        super(GraphTransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.node_encoder = nn.Embedding(STKG_num, embed_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, embed_dim, padding_idx=0)

        encoders = [EncoderLayer(embed_dim, num_heads, hidden_size, ffn_size, num_in_degree, num_out_degree)
                    for _ in range(num_layer)]
        self.encoder_layers = nn.ModuleList(encoders)

    def forward(self, STKG_embed_vector, in_degree, out_degree):
        # node_features = self.node_encoder(STKG_embed_vector)

        node_features = (
                STKG_embed_vector + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        ).to('cuda')
        output = self.dropout(node_features)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output)

        output = self.LayerNorm(output)

        return output



class EncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,hidden_size,ffn_size,num_indegree,num_outdegree):
        super(EncoderLayer, self).__init__()


        #Graph Information Embedding
        self.in_degree = nn.Embedding(num_indegree,hidden_size,padding_idx=0)
        self.out_degree = nn.Embedding(num_outdegree,hidden_size,padding_idx=0)


        self.selfattention = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads)
        self.LayerNorm =  nn.LayerNorm(embed_dim)
        self.layers1 = nn.Linear(embed_dim,ffn_size)
        self.gelu = nn.GELU()
        self.layers2 = nn.Linear(ffn_size,embed_dim)
        self.dropout = nn.Dropout(0.1)


    def  FeedForwardNetwork(self,x):
        x = self.layers1(x)
        x = self.gelu(x)
        x = self.layers2(x)
        return x

    def forward(self,x:torch.Tensor):
        '''

        :param x:
        :return:x,attn
        Build it upon the Transformer Architecture
        '''
        #Multi-Attention
        residual = x
        x_norm = self.LayerNorm(x)
        x, attn= self.selfattention(query=x_norm,key=x_norm,value=x_norm)
        x = self.dropout(x)
        x = x + residual
        #print(x.shape)

        #FeedForward Network
        residual = x
        x = self.LayerNorm(x)
        x = self.FeedForwardNetwork(x)
        x = self.dropout(x)
        x = residual + x
        return x



class GraphNodeFeatures(nn.Module):
    def __init__(self,STKG_num,
                 embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 ffn_size,
                 num_indegree,
                 num_outdegree,
                 num_layer
                 ):
        super(GraphNodeFeatures, self).__init__()

        self.Dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.node_encoder = nn.Embedding(STKG_num,embed_dim,padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree,embed_dim,padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree,embed_dim,padding_idx=0)
        encoders = [EncoderLayer(embed_dim,num_heads,hidden_size,ffn_size,num_indegree,num_outdegree)
                   for _ in range(num_layer) ]
        self.encoder_layers = nn.ModuleList(encoders)



    def forward(self, STKG_embed_vector, in_degree, out_degree):

        node_features = self.node_encoder(STKG_embed_vector)
        node_features = (
            node_features+self.in_degree_encoder(in_degree)+self.out_degree_encoder(out_degree)
        )
        output = self.Dropout(node_features)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output)

        output = torch.softmax(output).dim(-1)

        return output








class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

    def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias







