import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import argparse

with open ("vocab.txt", "r", encoding="UTF-8") as f:
    text = f.read()
    chars = sorted(list(set(text)))
# print(chars) 
vocab_size = len(chars) # 词汇表的长度，所有的输出都来自这里

# tokenizer 
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]    
decode = lambda l: "".join(int_to_string[d] for d in l)

class Head(nn.Module): 
    # One head of self attention
    def __init__(self, head_size):
        super().__init__()
        # 初始化 W_K, W_Q, W_V
        self.W_K = nn.Linear(n_embed, head_size, bias=False) 
        self.W_Q = nn.Linear(n_embed, head_size, bias=False)
        self.W_V = nn.Linear(n_embed, head_size, bias=False)
        '''
        创建一个下三角矩阵, 来制造mask
        例如4 x 4的全1矩阵, 进过torch.tril处理后, 只保留下三角部分:
        tensor([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 1., 1.]])
        '''
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # register_buffer的作用是用于注册模型中那些 不需要梯度更新 的参数
        # （比如batch normalization层中的运行均值和方差）
        # 同时确保这些参数被包括在模型的state_dict中

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # input or size (batch, time-step, channels) 
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.W_K(x) # X * W_K = K 下面的Q，V同理
        q = self.W_Q(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # '@'表示Tensor的点乘, (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1)  # (B, T, T)
        wei = self.dropout(wei) 

        v = self.W_V(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return out 
        
class MultiheadHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)]) # 创建多个Head对象，并行
        self.proj = nn.Linear(head_size * num_heads, n_embed) # Projection 投影：对应W_O矩阵，将全部头的输出结果投影到 嵌入维度
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # 将多个头拼起来，沿最后一个维度，h指的是每个头
        out = self.proj(out) # 乘以W_O，把维度转换到 嵌入维度
        out = self.dropout(out) 
        return out 

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4 指的是扩展因子（超参数）
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # 以概率 dropout 随机将输入的部分元素设置为零。使得网络在训练过程中不依赖于某些特定的神经元，从而增加网络的泛化能力，降低过拟合的风险。
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        # n_embd; embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadHeadAttention(n_head, head_size) # sa 是self-attention的简写
        self.ffwd = FeedForward(n_embed) 
        self.ln1 = nn.LayerNorm(n_embed) # ln 是layer-norm的简写
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.sa(x) 
        x = self.ln1(x + y) # 残差连接，将多头前(x)和多头后(y)相加，进行layer-norm，见Transformer结构图
        y = self.ffwd(x)
        x = self.ln2(x + y) # 残差连接，将ffwd前(x)和ffwd后(y)相加
        
        return x


class GPTLanguageMoudle(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embbeding_table = nn.Embedding(vocab_size, n_embed) # 词嵌入层
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # 位置嵌入层
        # 创建(n_layers)个decoder layers，使用torch.nn.Sequential将这些Transformer Blocks连接起来, 这里只体现连接，具体的实现在Block类中
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)]) 

        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) # 将logits映射回词汇表上各个词语的概率分布

        # 初始化模型各个层的权重
        self.apply(self._init_weights) # 继承自nn.Moudule，为它的所有子类(如池化层，线性层等)递归地应用括号中的方法

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    
    def forward(self, index, targets = None):
        # idx and targets are both （B,T）tensor of integers
        B, T = index.shape
        tok_emb = self.token_embbeding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # 创建step为1的数列[1, 2, ..., T] (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C) # 将三位张量展开为二维张量，将一个批次中的多个文本序列连接起来
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # 计算实际值和预测值之间的损失

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond) # 在这里获得训练后的logits
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 「sample」 from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Hyperparameters')
    
    # 定义参数
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device for training')
    parser.add_argument('--block_size', type=int, default=64, help='Block size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=2000, help='Maximum number of iterations')
    parser.add_argument('--eval_iters', type=int, default=100, help='Number of iterations between evaluations')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--n_embed', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of decoder layers')
    
    # 解析参数
    args = parser.parse_args()
    return args

args = parse_args()

################################################################################################

device = torch.device(args.device)
block_size = args.block_size # [1, 2, 3, 4, 5, 6, 7, 8]
batch_size = args.batch_size
# learning_rate = 2e-5
# max_iters = 2000
# eval_iters = 100
dropout = args.dropout # 随机丢弃（置零）神经元的输出，以防止过拟合
n_embed = args.n_embed # 嵌入维度
n_head = args.n_head # 多头的数量
n_layers = args.n_layers # Decoder的数量

################################################################################################

model = GPTLanguageMoudle(vocab_size)

# 若要加载训练好的模型，在这里使用下面的代码：
model = torch.load('model-exp.pth')
print("Loaded Successfully!")

m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(generated_chars)
