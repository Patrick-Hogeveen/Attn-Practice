import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 50 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)

class scaledDotProduct(nn.Module):
    
    '''
        Attention(Q, K, V ) = softmax( QK^T/√d_k)V 
    '''
    
    #Takes number of embedded, head_size, context length
    def __init__(self, embn, hdim, con_l, drop=0.0):

        super(scaledDotProduct, self).__init__()
        #dim is (d_k) when sqrt'd it is meant to counter small gradients in large sets of queries and keys
        self.k = nn.Linear(embn, hdim, bias=False)
        self.q = nn.Linear(embn, hdim, bias=False)
        self.v = nn.Linear(embn, hdim, bias=False)
        self.d_k = np.sqrt(hdim)
        con_l = 200
        self.register_buffer('mask', torch.tril(torch.ones(con_l,con_l)))
        #Simple drop out 
        self.drop = nn.Dropout(drop)

    def forward(self, x, ret_att=False):
        #first two dimensions are batch and number of heads?
        B,T,C = x.shape
        k = self.k(x)
        q = self.q(x)

        n = torch.matmul(q, k.transpose(-2,-1)) * k.shape[-1]**-0.5
        n = n.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        #Drop out referenced later in paper but not in original diagram
        att = self.drop(F.softmax(n, dim=-1))

        v = self.v(x)

        out = torch.matmul(att, v)
        if ret_att:
            return out, att 
        return out
        
    
class multiHeadedAttention(nn.Module):
    def __init__(self, n_heads, dims, embn, con_l, dropout=0.0):
        super(multiHeadedAttention, self).__init__()
        #d_k=d_v = dims/h

        self.n_heads = n_heads

        self.attn = nn.ModuleList([scaledDotProduct(embn, dims, con_l) for _ in range(n_heads)])
        #Final linear layer after concat and attention
        self.fc = nn.Linear(n_heads*dims, embn)

        self.drop = nn.Dropout(dropout)
        

    def forward(self, x):
        out = torch.cat([h(x) for h in self.attn], dim=-1)
        out = self.drop(self.fc(out))
        return out
    
class positionFeedFoward(nn.Module):
    def __init__(self, inp, hid, drop=0.0):
        super(positionFeedFoward, self).__init__()
        self.w1 = nn.Linear(inp,4*hid)
        self.w2 = nn.Linear(4*hid,inp)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.w2(F.relu(self.w1(x)))
        x = self.drop(x)

        return x
    
class Decoder(nn.Module):
    '''Combinds MultiHeadedAttention and FeeForward, three layers'''
    def __init__(self, dims, nheads, embn, con_l, drop=0.0):
        super(Decoder, self).__init__()
        head_size = embn // nheads
        self.slf_attn = multiHeadedAttention(nheads, head_size,embn, con_l, dropout=drop)
        
        self.ffn = positionFeedFoward(embn, embn, drop=drop)

        self.norm1 = nn.LayerNorm(embn)
        self.norm2 = nn.LayerNorm(embn)

    def forward(self, x):
        x = x + self.slf_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
class languageModel(nn.Module):
    '''Decoder model'''
    def __init__(
            self, n_vocab, embn, n_layers, n_head, dims, dropout=0.2 , con_l=200
    ):
        super(languageModel, self).__init__()
        self.con_l = con_l
        self.word_emb = nn.Embedding(n_vocab, embn)
        self.pos_enc = nn.Embedding(con_l, embn)
        self.stack = nn.Sequential(
            *[Decoder(dims, n_head, embn, con_l, drop=dropout) for _ in range(n_layers)]
        )
       
        self.layer_norm = nn.LayerNorm(embn)
        self.fc = nn.Linear(embn, n_vocab)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, tar=None):
        #batch, time
        B, T = x.shape

        tok = self.word_emb(x)
        pos = self.pos_enc(torch.arange(T, device=device))
        x = tok + pos
        x = self.stack(x)
        x = self.layer_norm(x)
        logits = self.fc(x)

        if tar is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            tar = tar.view(B*T)
            loss = F.cross_entropy(logits, tar)

        return logits, loss
    
    def generate(self, x, max_length):
        #x is a BxT array of in current context
        for _ in range(max_length):
            x_cond = x[:, -20:]
            logits, loss = self(x_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)

        return x
    
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



torch.manual_seed(1337)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)[0]
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = F.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = languageModel(vocab_size,  384,6, 6, 50, con_l=50
    )
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print(next(m.parameters()).is_cuda)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    #B, T, C = logits.shape
    #logits = logits.view(B*T, C)
    #targets = yb.view(B*T)
    #loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_length=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))