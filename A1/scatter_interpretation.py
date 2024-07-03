import torch


def custom_index_copy(dim, self, index, src):
    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            for k in range(self.shape[2]):
                if dim == 0:
                    self[index[i][j][k]][j][k] = src[i][j][k]
                elif dim == 1:
                    self[i][index[i][j][k]][k] = src[i][j][k]
                elif dim == 2:
                    self[i][j][index[i][j][k]] = src[i][j][k]
    return self


# Create example tensors
src = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
index = torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])
print("Source tensor:")
print(src)
print("\nIndex tensor:")
print(index)

# Example for dim = 0
self_dim0 = torch.zeros(2, 2, 2)
result_dim0 = custom_index_copy(0, self_dim0, index, src)
print("\nResult for dim = 0:")
print(result_dim0)

# Example for dim = 1
self_dim1 = torch.zeros(2, 2, 2)
result_dim1 = custom_index_copy(1, self_dim1, index, src)
print("\nResult for dim = 1:")
print(result_dim1)

# Example for dim = 2
self_dim2 = torch.zeros(2, 2, 2)
result_dim2 = custom_index_copy(2, self_dim2, index, src)
print("\nResult for dim = 2:")
print(result_dim2)

# along the column
idx_lst_1 = [[0, 0, 1, 3]]
idx_tsr_1 = torch.tensor(idx_lst_1)
print("the shape of idx_tsr_1 is", idx_tsr_1.shape)
result_1 = torch.zeros(4, 4).scatter_(1, idx_tsr_1, 1)
print("result_1:")
print(result_1)

idx_lst_2 = [0, 0, 1, 3]
idx_tsr_2 = torch.tensor(idx_lst_2).unsqueeze(1)
print("the shape of idx_tsr_2 is", idx_tsr_2.shape)
result_2 = torch.zeros(4, 4).scatter_(1, idx_tsr_2, 1)
print("result_2:")
print(result_2)

# along the row
idx_lst_3 = [[0, 0, 1, 3]]
idx_tsr_3 = torch.tensor(idx_lst_3)
print("the shape of idx_tsr_3 is", idx_tsr_3.shape)
result_3 = torch.zeros(4, 4).scatter_(0, idx_tsr_3, 1)
print("result_3:")
print(result_3)

idx_lst_4 = [0, 0, 1, 3]
idx_tsr_4 = torch.tensor(idx_lst_4).unsqueeze(1)
print("the shape of idx_tsr_4 is", idx_tsr_4.shape)
result_4 = torch.zeros(4, 4).scatter_(0, idx_tsr_4, 1)
print("result_4:")
print(result_4)
