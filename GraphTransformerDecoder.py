import torch
import torch.nn as nn


# Decoder
class GraphTransformerDecoder(nn.Module):
    def __init__(self, hidden_size, dropout_rate, num_in_degree, embed_dim, num_out_degree, num_decoder_layers,
                 num_heads, ffn_size):
        super(GraphTransformerDecoder, self).__init__()

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.softmax = nn.Softmax(dim=-1)
        self.Dropout = nn.Dropout(dropout_rate)
        self.in_degree_encoder = nn.Embedding(num_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, embed_dim, padding_idx=0)

        decoders = [DecoderLayer(embed_dim, num_heads, ffn_size)
                    for _ in range(num_decoder_layers)]
        self.decoder_layers = nn.ModuleList(decoders)
        self.layers_fn = nn.Linear(128, 62)

    def forward(self, STSL_hidden_state):
        STSL_hidden_state = self.Dropout(STSL_hidden_state)
        for dec_layer in self.decoder_layers:
            output = dec_layer(STSL_hidden_state)
        STSL_Construct = output
        STSL_Construct = self.layers_fn(STSL_Construct)
        ##print(STSL_Construct.shape)
        # STSL_Construct = nn.Softmax(output)
        return STSL_Construct

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim,num_heads,ffn_size):
        super(DecoderLayer,self).__init__()
        self.selfattention = nn.MultiheadAttention(embed_dim,num_heads)
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




    def forward(self,x):
        # Multi-Attention
        residual = x
        x_norm = self.LayerNorm(x)
        x, attn = self.selfattention(query=x_norm, key=x_norm, value=x_norm)
        x = self.dropout(x)
        x = x + residual
        #print(x.shape)

        # FeedForward Network
        residual = x
        x = self.LayerNorm(x)
        x = self.FeedForwardNetwork(x)
        x = self.dropout(x)
        x = residual + x

        # Multi-Attention
        residual = x
        x_norm = self.LayerNorm(x)
        x, attn = self.selfattention(query=x_norm, key=x_norm, value=x_norm)
        x = self.dropout(x)
        x = x + residual


        return x




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



