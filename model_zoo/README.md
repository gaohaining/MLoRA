pretrain_model layers:
uid
pid
domain_1
domain_2
user_emb
item_emb
domain_emb1
domain_emb2
concatenate
flatten
m_lo_rafcn
m_lo_rafcn_1
m_lo_rafcn_2
dense

model layers:
uid
pid
domain_1
domain_2
user_emb
item_emb
domain_emb1
domain_emb2
concatenate_1
flatten_1
m_lo_rafcn_3
m_lo_rafcn_4
m_lo_rafcn_5
dense_1

导入参数时，需要注意层在第二次创建时名称发生了改变，需要自行匹配。