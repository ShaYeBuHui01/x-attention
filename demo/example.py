import torch
from xattn.src.Xattention import Xattention_prefill
import flashinfer
import time


bsz = 1
heads = 32
seq_len = 1024
dim = 128
q = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to("cuda")
k = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to("cuda")
v = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to("cuda")

start = time.time()
for i in range(10):
    # attention_output1 = Xattention_prefill(query_states=q,key_states=k,value_states=v,stride=16,block_size=128,use_triton=True,chunk_size=2048)
    # attention_output1 = Xattention_prefill(query_states=q,key_states=k,value_states=v,stride=8,threshold=0.9,chunk_size=2048)
    attention_output1 = Xattention_prefill(query_states=q,key_states=k,value_states=v,stride=8,norm=1,threshold=0.9,use_triton=True)
end = time.time()
# print(attention_output1)
print("time1 ", (end - start) / 10)


for i in range(10):
    # attention_output1 = Xattention_prefill(query_states=q, key_states=k, value_states=v, stride=8, norm=1, threshold=torch.ones((32,32)).to("cuda")*0.9, use_triton=True)
    attention_output2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
end = time.time()
# print(attention_output2)
print("time2 ", (end - start) / 10)

for i in range(10):
    attention_output3 = flashinfer.single_prefill_with_kv_cache(
        q.transpose(1, 2).squeeze(0),
        k.transpose(1, 2).squeeze(0),
        v.transpose(1, 2).squeeze(0),
        custom_mask=None,
        causal=False
    ).unsqueeze(0).transpose(1, 2)
end = time.time()
# print(attention_output3)
print("time3 ", (end - start) / 10)
