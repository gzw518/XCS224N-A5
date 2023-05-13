import torch

# 原始张量
x = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])

# 掩码张量
mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])

# 将填充位置的值替换成0
y = x.masked_fill(mask == 0, 0)

print(y)






# import torch
#
# # 输入张量
# x = torch.randn(2, 3, 4, 4)
#
# # 掩码张量
# T = 3
# mask = torch.zeros(2, 3, 4, 4, dtype=torch.bool)
# mask[:,:,:T,:T] = 1
# print(mask)
# # 将需要被替换的位置的值替换成0
# y = x.masked_fill(mask == 0, 0)
# print(y)



# import random
# data = "First Citizen speak."
# print(data)
# #print(data[:])
# content_len_diff = random.randint(0, 2)
# print(content_len_diff)


# MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
# PAD_CHAR = u"\u25A1"  # the empty square character, for pad
#
# #data = "First Citizen:Before we proceed any further, hear me speak."
# data = "First Citizen speak."
# content_len_diff = random.randint(-1, 1)
# masked_content_len = len(data) // 4 + content_len_diff
# print("masked_content_len:",masked_content_len)
#
# masked_start_position = random.randint(0, len(data) - masked_content_len)
# print("masked_start_position:",masked_start_position)
#
# print(data)
# data = data[:masked_start_position] + MASK_CHAR*masked_content_len + data[masked_start_position + masked_content_len:]
# print(data)



# content_len_diff = []
# for i in range(100000):
#     content_len_diff.append(round(random.normalvariate(0, 1)))
#     #content_len_diff.append(random.randint(-1, 1))
#
# print(sum(content_len_diff))
# print(len(content_len_diff))
# print(sum(content_len_diff)/len(content_len_diff))




# data = "First Citizen:Before we proceed any further, hear me speak."
# idx = 0
# inp, oup = data[idx].split('\t')
# print(inp)
# print(oup)


# import torch
#
# data = "First Citizen:Before we proceed any further, hear me speak."
# block_size = 4
# idx = 0
# chars = sorted(list(set(data)))
# stoi = { ch:i for i,ch in enumerate(chars) }
# chunk = data[idx:idx + block_size + 1]
# dix = [stoi[s] for s in chunk]
# x = torch.tensor(dix[:-1], dtype=torch.long)
# y = torch.tensor(dix[1:], dtype=torch.long)
# print(dix)
# print(x)
# print(y)





# data = "First Citizen:Before we proceed any further, hear me speak."
# chars=sorted(list(set(data)))
# print(chars)
# print(set(data))




# chars = "hello"
# stoi = { ch:i for i,ch in enumerate(chars) }
#
# print(stoi)
# print(stoi['o'])
#
# chars_2 = ["h","e","l","l","o"]
# stoi_2 = { ch:i for i,ch in enumerate(chars_2) }
# print(stoi_2)
