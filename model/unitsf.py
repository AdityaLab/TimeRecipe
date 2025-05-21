import torch
import torch.nn as nn

from module.embed import NoEmbedding, TokenEmbedding, PatchEmbedding, InvertEmbedding, FreqEmbedding
from module.architecture import MLP, RNN, Transformer
from module.decomp import NoDecomp, SeriesDecomp
from module.norm import InstNorm


import pdb

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name 

        self.use_norm = configs.use_norm  
        self.use_decomp = configs.use_decomp  
        self.emb_type = configs.emb_type 
        self.ff_type = configs.ff_type  
        self.fusion_type = configs.fusion

        assert isinstance(self.use_norm, bool)  
        assert isinstance(self.use_decomp, bool)  
        assert self.emb_type in ['token', 'patch', 'invert', 'freq', 'none']
        assert self.ff_type in ['mlp', 'rnn', 'trans']    
        assert self.fusion_type in ['temporal', 'feature']   

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.embed = configs.embed
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.factor = configs.factor
        self.freq = configs.freq

        # patch embedding param
        self.stride = 8
        self.padding = self.stride
        self.patch_len = configs.patch_len
        self.num_patches = (configs.seq_len-configs.patch_len)//self.stride + 2

        # series decomp param
        self.moving_avg = configs.moving_avg

        # Shape config to connect all modules accordingly
        shape_config_dict = {
            'temporal_token_mlp': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.d_model},
            'temporal_token_rnn': {'model_in_size': configs.d_model, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.d_model},
            'temporal_token_trans': {'model_in_size': configs.d_model, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.d_model},
            'feature_token_mlp': {'model_in_size': configs.d_model, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.d_model},
            'feature_token_rnn': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.d_model},
            'feature_token_trans': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.d_model},

            'temporal_patch_mlp': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            'temporal_patch_rnn': {'model_in_size': self.num_patches, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            'temporal_patch_trans': {'model_in_size': self.num_patches, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            'feature_patch_mlp': {'model_in_size': self.num_patches, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            'feature_patch_rnn': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            'feature_patch_trans': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},

            'temporal_invert_mlp': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            'temporal_invert_rnn': {'model_in_size': configs.enc_in, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            'temporal_invert_trans': {'model_in_size': configs.enc_in, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            'feature_invert_mlp': {'model_in_size': configs.enc_in, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            'feature_invert_rnn': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            'feature_invert_trans': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},

            'temporal_freq_mlp': {'model_in_size': configs.seq_len//2+1, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            'temporal_freq_rnn': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            'temporal_freq_trans': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            'feature_freq_mlp': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            'feature_freq_rnn': {'model_in_size': configs.seq_len//2+1, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            'feature_freq_trans': {'model_in_size': configs.seq_len//2+1, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},

            'temporal_none_mlp': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
            'temporal_none_rnn': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
            'temporal_none_trans': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
            'feature_none_mlp': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
            'feature_none_rnn': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
            'feature_none_trans': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},

            # 'patch_temporal': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            # 'patch_feature': {'model_in_size': self.num_patches, 'proj_l_size': configs.d_model*self.num_patches, 'proj_d_size': configs.enc_in},
            # 'invert_temporal': {'model_in_size': configs.d_model, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            # 'invert_feature': {'model_in_size': configs.enc_in, 'proj_l_size': configs.d_model, 'proj_d_size': configs.enc_in},
            # 'freq_tempoal': {'model_in_size': configs.seq_len//2+1, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            # 'freq_feature': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len//2+1, 'proj_d_size': configs.enc_in},
            # 'none_temporal': {'model_in_size': configs.seq_len, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
            # 'none_feature': {'model_in_size': configs.enc_in, 'proj_l_size': configs.seq_len, 'proj_d_size': configs.enc_in},
        }

        self.fusion_emb_config = self.fusion_type + '_' + self.emb_type + '_' + self.ff_type
        self.fusion_config = self.fusion_type + '_' + self.ff_type
        self.model_in_size = shape_config_dict[self.fusion_emb_config]['model_in_size']
        # self.model_out_size = shape_config_dict[self.fusion_emb_config]['model_out_size']
        self.proj_l_size = shape_config_dict[self.fusion_emb_config]['proj_l_size']
        self.proj_d_size = shape_config_dict[self.fusion_emb_config]['proj_d_size']

        # Build model
        self._build_model()
        self._build_emb()

        if self.emb_type == 'freq':
            self.ff_model.to(torch.cfloat)
            self.proj_l.to(torch.cfloat)
            self.proj_d.to(torch.cfloat)

        if self.use_norm:
            self.norm = InstNorm()

        if self.use_decomp:
            self.decompsition = SeriesDecomp(self.moving_avg)
        else:
            self.decompsition = NoDecomp()

        print(self.fusion_config)


    def forward(self, x_enc):
        # Norm
        if self.use_norm:
            x_enc = self.norm.norm(x_enc)

        # Series Decomposition
        x_enc = self.decompsition(x_enc)

        # Input Embedding
        # [B,L,D] -> [B,L,H] token 
        #            [B,H,D] invert
        #            [B*D,H,(L-patch)//stried +2] patch
        #            [B,L//2+1,D] freq
        #            [B,L,D] noemb
        x_emb = [self.emb[i](x_enc[i]) for i in range(len(x_enc))] 

        # Temporal/Feature Fusion Input
        if self.fusion_config in ['temporal_mlp', 'feature_rnn', 'feature_trans']:
            x_emb = [x.permute(0,2,1) for x in x_emb] 

        enc_out = [self.ff_model[i](x_emb[i]) for i in range(len(x_emb))]

        if self.fusion_config in ['temporal_mlp', 'feature_rnn', 'feature_trans']:
            enc_out = [x.permute(0,2,1) for x in enc_out] 

        if self.emb_type == 'patch':
            enc_out = [x.reshape(-1, self.enc_in, self.d_model*self.num_patches).permute(0,2,1) for x in enc_out]

        dec_out = [self.proj_l[i](enc_out[i].permute(0,2,1)).permute(0,2,1) for i in range(len(enc_out))]

        # Temporal/Feature Fusion Input
        if self.emb_type == 'token':
            dec_out = [self.proj_d[i](dec_out[i]) for i in range(len(dec_out))]

        if self.emb_type == 'freq':
            dec_out = [self.emb[i].inverse(dec_out[i]) for i in range(len(dec_out))]

        # Series Decomposition Sum
        dec_out = dec_out[0] if len(dec_out) == 1 else dec_out[0] + dec_out[1]

        # pdb.set_trace()
        # Denorm
        if self.use_norm:
            dec_out = self.norm.denorm(dec_out[:,-self.pred_len:])

        return dec_out
        

    def _build_emb(self):
        n = 2 if self.use_decomp else 1
        if self.emb_type == 'token':
            self.emb = nn.ModuleList([TokenEmbedding(self.enc_in, self.d_model, dropout=self.dropout) for i in range(n)])
        elif self.emb_type == 'patch':
            self.emb = nn.ModuleList([PatchEmbedding(self.d_model, self.patch_len, stride=self.stride, padding=self.padding, dropout=self.dropout) for i in range(n)])
        elif self.emb_type == 'invert':
            self.emb = nn.ModuleList([InvertEmbedding(self.seq_len, self.d_model, dropout=self.dropout) for i in range(n)])
        elif self.emb_type == 'freq':
            self.emb = nn.ModuleList([FreqEmbedding() for i in range(n)])
        else:
            self.emb = nn.ModuleList([NoEmbedding() for i in range(n)])


    def _build_model(self):
        n = 2 if self.use_decomp else 1
        if self.ff_type == 'mlp':
            self.ff_model = nn.ModuleList([
                MLP(model_in_size=self.model_in_size, d_model=self.d_model, e_layers=self.e_layers, emb_type=self.emb_type)
                for i in range(n)])  
            
        elif self.ff_type == 'rnn':
            self.ff_model = nn.ModuleList([
                RNN(model_in_size=self.model_in_size, d_model=self.d_model, e_layers=self.e_layers, emb_type=self.emb_type, dropout=self.dropout)
                for i in range(n)])
            
        elif self.ff_type == 'trans':
            self.ff_model = nn.ModuleList([
                Transformer(model_in_size=self.model_in_size, d_ff=self.d_ff, e_layers=self.e_layers, emb_type=self.emb_type,\
                            n_heads=self.n_heads, dropout=self.dropout, factor=self.factor, activation=self.activation)
                for i in range(n)])

        self.proj_l = nn.ModuleList([nn.Linear(self.proj_l_size, (self.seq_len+self.pred_len)//2+1) if self.emb_type=='freq' \
                                     else nn.Linear(self.proj_l_size, self.pred_len) for i in range(n)])
        self.proj_d = nn.ModuleList([nn.Linear(self.proj_d_size, self.enc_in) for i in range(n)])

        print('Number of FF Model Parameters: {}'.format(count_parameters(self.ff_model)))
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)