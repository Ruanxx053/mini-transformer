#!/usr/bin/env python3
# train.py  ——  从零开始真正训练德英翻译 Transformer
import math, time, os, random
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import sacrebleu
# 移除 FP16 相关的导入，使用 FP32 以提高稳定性
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch.nn.functional as F

# ------------------------------------------------------------
# 1. 基础配置
# ------------------------------------------------------------
SRC_LANG  = 'de'
TGT_LANG  = 'en'
BATCH_SZ  = 12           # 进一步减小批量大小以减少内存使用
EMB_DIM   = 512         # 保持嵌入维度
HID_DIM   = 2048        # 保持前馈网络维度
HEADS     = 8           # 保持注意力头数
ENC_LAYERS = DEC_LAYERS = 6  # 保持层数
DROPOUT   = 0.4          # 适中的 dropout 正则化
MAX_LEN   = 100          # 减小句子最大长度以减少内存使用
DEVICE    = torch.device('cpu')
import platform
is_windows = platform.system() == 'Windows'
NUM_WORKERS = 0 if is_windows else min(4, os.cpu_count() or 1)  # Windows 上使用单进程
print('device =', DEVICE)

# ------------------------------------------------------------
# 2. BPE分词器训练和加载
# ------------------------------------------------------------
def train_bpe_tokenizer(corpus_file, save_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(
        vocab_size=8000,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    tokenizer.train([corpus_file], trainer)
    tokenizer.save(save_path)
    return tokenizer

def load_or_train_tokenizer(corpus_file, save_path):
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)
    return train_bpe_tokenizer(corpus_file, save_path)

data_dir = 'data/raw/de-en'
src_train_path = f'{data_dir}/train.de'
tgt_train_path = f'{data_dir}/train.en'
src_val_path   = f'{data_dir}/IWSLT14.TED.dev2010.de'
tgt_val_path   = f'{data_dir}/IWSLT14.TED.dev2010.en'

# 分别训练/加载德语和英语分词器
bpe_tokenizer_de = load_or_train_tokenizer(src_train_path, "tokenizer-de.json")
bpe_tokenizer_en = load_or_train_tokenizer(tgt_train_path, "tokenizer-en.json")

# BPE Dropout: 训练时分词器 encode 增加 dropout
TRAIN_BPE_DROPOUT = 0.1

def tokenize_de(text, train=False):
    return bpe_tokenizer_de.encode(text.strip()).ids

def tokenize_en(text, train=False):
    return bpe_tokenizer_en.encode(text.strip()).ids

def detokenize_de(ids):
    return bpe_tokenizer_de.decode(ids)

def detokenize_en(ids):
    return bpe_tokenizer_en.decode(ids)

# special token id
PAD_IDX_DE = bpe_tokenizer_de.token_to_id("<pad>")
SOS_IDX_DE = bpe_tokenizer_de.token_to_id("<sos>")
EOS_IDX_DE = bpe_tokenizer_de.token_to_id("<eos>")
UNK_IDX_DE = bpe_tokenizer_de.token_to_id("<unk>")
VOCAB_SIZE_DE = bpe_tokenizer_de.get_vocab_size()

PAD_IDX_EN = bpe_tokenizer_en.token_to_id("<pad>")
SOS_IDX_EN = bpe_tokenizer_en.token_to_id("<sos>")
EOS_IDX_EN = bpe_tokenizer_en.token_to_id("<eos>")
UNK_IDX_EN = bpe_tokenizer_en.token_to_id("<unk>")
VOCAB_SIZE_EN = bpe_tokenizer_en.get_vocab_size()

# ------------------------------------------------------------
# 3. Dataset / DataLoader
# ------------------------------------------------------------
class IWSLTDataset(Dataset):
    def __init__(self, src_path, tgt_path, max_size=None, train=False):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.train = train
        with open(src_path, encoding='utf-8') as f:
            self.total_lines = sum(1 for _ in f)
        if max_size and max_size < self.total_lines:
            self.total_lines = max_size
        print(f"Dataset size: {self.total_lines} parallel sentences")

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        if idx >= self.total_lines:
            raise IndexError("Index out of bounds")
        import linecache
        src_line = linecache.getline(self.src_path, idx + 1).strip()
        tgt_line = linecache.getline(self.tgt_path, idx + 1).strip()
        if not src_line or not tgt_line:
            src_line = linecache.getline(self.src_path, 1).strip()
            tgt_line = linecache.getline(self.tgt_path, 1).strip()
        src = tokenize_de(src_line, train=self.train)[:MAX_LEN-2]
        tgt = tokenize_en(tgt_line, train=self.train)[:MAX_LEN-2]
        src = [SOS_IDX_DE] + src + [EOS_IDX_DE]
        tgt = [SOS_IDX_EN] + tgt + [EOS_IDX_EN]
        return torch.tensor(src), torch.tensor(tgt)

def collate(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(list(src_batch), padding_value=PAD_IDX_DE)
    tgt_batch = nn.utils.rnn.pad_sequence(list(tgt_batch), padding_value=PAD_IDX_EN)
    return src_batch.T, tgt_batch.T   # (batch, len)

# 限制训练集大小以避免过拟合
MAX_TRAIN_SIZE = 100000  # 减少训练集大小
# 训练集用 BPE Dropout
train_ds = IWSLTDataset(src_train_path, tgt_train_path, max_size=MAX_TRAIN_SIZE, train=True)
# 验证集用完整 dev2010
val_ds   = IWSLTDataset(src_val_path, tgt_val_path, train=False)

train_loader = DataLoader(train_ds, BATCH_SZ, shuffle=True,
                         collate_fn=collate,
                         num_workers=NUM_WORKERS,
                         pin_memory=False if is_windows else True,
                         persistent_workers=False if is_windows else True)
val_loader   = DataLoader(val_ds, BATCH_SZ, shuffle=False,
                         collate_fn=collate,
                         num_workers=NUM_WORKERS,
                         pin_memory=False if is_windows else True,
                         persistent_workers=False if is_windows else True)

# 检查数据是否对齐
print("\nSample training pairs (detokenized):")
for i in range(5):
    src, tgt = train_ds[i]
    print("SRC:", detokenize_de(src.tolist()))
    print("TGT:", detokenize_en(tgt.tolist()))

# ------------------------------------------------------------
# 4. 模型
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].transpose(0, 1)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_tok_emb = nn.Embedding(VOCAB_SIZE_DE, EMB_DIM)
        self.tgt_tok_emb = nn.Embedding(VOCAB_SIZE_EN, EMB_DIM)
        self.pos_encoding = PositionalEncoding(EMB_DIM, DROPOUT)
        self.transformer = nn.Transformer(
            d_model=EMB_DIM,
            nhead=HEADS,
            num_encoder_layers=ENC_LAYERS,
            num_decoder_layers=DEC_LAYERS,
            dim_feedforward=HID_DIM,
            dropout=DROPOUT,
            batch_first=True)
        self.generator = nn.Linear(EMB_DIM, VOCAB_SIZE_EN)
        # 不再共享权重
        # self.src_tok_emb.weight = self.tgt_tok_emb.weight
        # self.generator.weight = self.tgt_tok_emb.weight

    def make_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(DEVICE)
        src_padding = src != PAD_IDX_DE
        tgt_padding = tgt != PAD_IDX_EN
        src_key_padding_mask = ~src_padding
        tgt_key_padding_mask = ~tgt_padding
        memory_key_padding_mask = src_key_padding_mask
        return tgt_mask.bool(), src_key_padding_mask.bool(), tgt_key_padding_mask.bool(), memory_key_padding_mask.bool()

    def forward(self, src, tgt):
        tgt_mask, src_key, tgt_key, mem_key = self.make_mask(src, tgt)
        src_emb = self.pos_encoding(self.src_tok_emb(src))
        tgt_emb = self.pos_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key,
            tgt_key_padding_mask=tgt_key,
            memory_key_padding_mask=mem_key)
        return self.generator(outs)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=MAX_LEN, topk=1, repetition_penalty=1.2):
        self.eval()
        src_mask = (src != PAD_IDX_DE).bool()
        src_key_padding_mask = ~src_mask
        src_emb = self.pos_encoding(self.src_tok_emb(src))
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask)
        ys = torch.ones(src.shape[0], 1).fill_(SOS_IDX_EN).long().to(DEVICE)
        for i in range(max_len-1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(DEVICE).bool()
            tgt_key_padding_mask = (ys == PAD_IDX_EN).bool()
            tgt_emb = self.pos_encoding(self.tgt_tok_emb(ys))
            out = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask)
            logits = self.generator(out[:, -1])
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for j in range(logits.size(0)):
                    for previous_token in set(ys[j].tolist()):
                        if previous_token != PAD_IDX_EN and previous_token != SOS_IDX_EN and previous_token != EOS_IDX_EN:
                            logits[j, previous_token] /= repetition_penalty
            
            log_probs = F.log_softmax(logits, dim=-1)
            # n-gram 阻断（仅支持 batch=1）
            if ys.shape[0] == 1:
                log_probs = block_repeating_ngrams(log_probs, ys[0].tolist(), n=3, penalty=5.0)
                log_probs = penalize_repetition(log_probs, ys[0].tolist(), penalty=1.0)
                if topk > 1:
                    top_log_probs, top_indices = torch.topk(log_probs, topk, dim=-1)
                    probs = torch.softmax(top_log_probs, dim=-1)
                    next_word = top_indices[0, torch.multinomial(probs[0], 1)].unsqueeze(0)
                else:
                    next_word = log_probs.argmax(dim=-1, keepdim=True)
            else:
                next_word = log_probs.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
            if (next_word == EOS_IDX_EN).all():
                break
        return ys

    @torch.no_grad()
    def beam_search_decode(self, src, max_len=MAX_LEN, beam_width=5, length_penalty=0.1, repetition_penalty=1.2):
        self.eval()
        src = src.to(DEVICE)
        src_mask = (src != PAD_IDX_DE).bool()
        src_key_padding_mask = ~src_mask
        src_emb = self.pos_encoding(self.src_tok_emb(src))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        batch_size = src.size(0)
        device = src.device
        ys = torch.full((batch_size, 1), SOS_IDX_EN, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device).bool()
            tgt_key_padding_mask = (ys == PAD_IDX_EN).bool()
            tgt_emb = self.pos_encoding(self.tgt_tok_emb(ys))
            out = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = self.generator(out[:, -1])  # (batch, vocab)
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for j in range(logits.size(0)):
                    for previous_token in set(ys[j].tolist()):
                        if previous_token != PAD_IDX_EN and previous_token != SOS_IDX_EN and previous_token != EOS_IDX_EN:
                            logits[j, previous_token] /= repetition_penalty
            
            log_probs = F.log_softmax(logits, dim=-1)
            # n-gram 阻断（仅支持 batch=1）
            if ys.shape[0] == 1:
                log_probs = block_repeating_ngrams(log_probs, ys[0].tolist(), n=3, penalty=5.0)
                log_probs = penalize_repetition(log_probs, ys[0].tolist(), penalty=1.0)
            # 长度惩罚
            log_probs = log_probs - length_penalty * (step + 1)
            # 取 top-k
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            next_word = top_indices[:, 0].unsqueeze(-1)
            ys = torch.cat([ys, next_word], dim=1)
            finished = finished | (next_word == EOS_IDX_EN)
            if finished.all():
                break
        return ys

def block_repeating_ngrams(log_probs, generated_ids, n=3, penalty=1e20):
    if len(generated_ids) < n:
        return log_probs
    last_ngram = tuple(generated_ids[-(n-1):])
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i+n-1]) == last_ngram:
            log_probs[0, generated_ids[i+n-1]] -= penalty
    return log_probs

def penalize_repetition(log_probs, generated_ids, penalty=1.0):
    """对重复token施加软惩罚"""
    for token_id in set(generated_ids):
        count = generated_ids.count(token_id)
        # 使用指数惩罚，重复次数越多惩罚越重
        log_probs[0, token_id] -= penalty * (count ** 1.5)
    return log_probs

model = TransformerModel().to(DEVICE)
# 检查 Embedding 层大小
assert VOCAB_SIZE_DE == model.src_tok_emb.num_embeddings, f"vocab/embedding size mismatch: {VOCAB_SIZE_DE} vs {model.src_tok_emb.num_embeddings}"
assert VOCAB_SIZE_EN == model.tgt_tok_emb.num_embeddings, f"vocab/embedding size mismatch: {VOCAB_SIZE_EN} vs {model.tgt_tok_emb.num_embeddings}"

# ------------------------------------------------------------
# 5. 损失函数 / 优化器 / 学习率
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_EN, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
# 使用更合适的学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ------------------------------------------------------------
# 6. 训练 & 验证循环
# ------------------------------------------------------------
def train_epoch(epoch):
    model.train()
    losses = 0
    total_tokens = 0
    
    # 梯度累积步数
    ACCUMULATION_STEPS = 8  # 增加梯度累积步数以进一步减少内存使用
    optimizer.zero_grad()
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 使用较低精度
    if DEVICE.type == 'cuda':
        model.half()  # 使用 FP16
        print("Using half precision to save memory")
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        # 索引越界断言
        assert src.max().item() < VOCAB_SIZE_DE, f"src 越界: max={src.max().item()}, vocab_size={VOCAB_SIZE_DE}"
        assert tgt.max().item() < VOCAB_SIZE_EN, f"tgt 越界: max={tgt.max().item()}, vocab_size={VOCAB_SIZE_EN}"
        try:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 计算这个批次中的实际token数（不包括padding）
            non_pad_mask = tgt_output != PAD_IDX_EN
            num_tokens = non_pad_mask.sum().item()
            
            # 前向传播和损失计算（禁用混合精度以提高稳定性）
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), 
                           tgt_output.reshape(-1))
            # 根据累积步数缩放损失
            loss = loss / ACCUMULATION_STEPS
            
            # 反向传播
            loss.backward()
            
            # 累积足够的梯度后更新
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新统计信息
            losses += loss.item() * ACCUMULATION_STEPS
            total_tokens += num_tokens
            
            # 定期打印进度
            if (batch_idx + 1) % 50 == 0:
                avg_loss = losses / total_tokens if total_tokens > 0 else float('inf')
                print(f"\rEpoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Tokens: {total_tokens}", end="")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nCUDA OOM in batch {batch_idx}. Skipping this batch...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    # 处理最后一个不完整的累积周期
    if (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # epoch结束后清理内存
    torch.cuda.empty_cache()
    
    return losses / total_tokens if total_tokens > 0 else float('inf')

@torch.no_grad()
def evaluate():
    model.eval()
    losses = 0
    total_tokens = 0
    refs, hyps = [], []
    print("\nStarting evaluation...")
    for batch_idx, (src, tgt) in enumerate(val_loader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        non_pad_mask = tgt_output != PAD_IDX_EN
        num_tokens = non_pad_mask.sum().item()
        total_tokens += num_tokens
        losses += loss.item() * num_tokens
        # 使用 beam search 解码，增加重复惩罚
        pred_tokens = model.beam_search_decode(src, max_len=MAX_LEN, beam_width=5, length_penalty=0.2, repetition_penalty=1.5)
        for ref, hyp in zip(tgt, pred_tokens):
            ref_ids = [int(t) for t in ref if t not in [PAD_IDX_EN, SOS_IDX_EN, EOS_IDX_EN]]
            hyp_ids = [int(t) for t in hyp if t not in [PAD_IDX_EN, SOS_IDX_EN, EOS_IDX_EN]]
            ref_words = detokenize_en(ref_ids)
            hyp_words = detokenize_en(hyp_ids)
            refs.append([ref_words])
            hyps.append(hyp_words)
            if len(refs) % 100 == 0:
                print(f"\nExample {len(refs)}:")
                print(f"HYP: {hyp_words}")
                print(f"REF: {ref_words}")
    avg_loss = losses / total_tokens if total_tokens > 0 else float('inf')
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print(f"\nValidation Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"BLEU: {bleu.score:.2f}")
    print(f"BLEU Components: {bleu.precisions}")
    return avg_loss, bleu.score

# ------------------------------------------------------------
# 7. 主循环
# ------------------------------------------------------------
def main():
    best_val_loss = float('inf')  # 修复变量名和初始值
    patience = 3  # 增加早停的耐心值
    counter = 0   # 计数器
    
    for epoch in range(1, 51):  # 增加到50个epoch，确保模型有足够时间收敛
        start = time.time()
        train_loss = train_epoch(epoch)
        val_loss, val_bleu = evaluate()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率（基于验证损失）
        scheduler.step(val_loss)
        
        # 如果学习率改变，打印新的学习率
        new_lr = optimizer.param_groups[0]['lr']
        if current_lr != new_lr:
            print(f'Learning rate changed from {current_lr:.6f} to {new_lr:.6f}')
        
        mins = (time.time() - start) // 60
        print(f'Epoch {epoch:02d} | {mins:2.0f}m | lr {current_lr:.6f} | '
              f'train loss {train_loss:.4f} | val loss {val_loss:.4f} | BLEU {val_bleu:.2f}')
        
        # 早停监控 val_loss
        if val_loss < best_val_loss:  # 修复变量名
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best.pt')
            print(f'New best val_loss: {best_val_loss:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping: No improvement for {patience} epochs. Best val_loss: {best_val_loss:.4f}")
                break

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

    