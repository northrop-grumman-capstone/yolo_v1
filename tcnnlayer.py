import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ConcatHidden(nn.Module):
	def __init__(self, kernel_size):
		super(ConcatHidden, self).__init__()
		self.k = kernel_size

	def forward(self, x, h):
		if(h is None): h = torch.zeros(x.shape[0], x.shape[1], self.k-1).to(x.device)
		return torch.cat([h, x], dim=2)

class TCNN(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride=1, dilation=1, dropout=0.2, p_in=False, p_out=False):
		super(TCNN, self).__init__()
		self.p_in = p_in
		self.p_out = p_out
		self.kernel_size = kernel_size
		self.ch = ConcatHidden(kernel_size)
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
			stride=stride, dilation=dilation))
		self.relu = nn.LeakyReLU(0.1)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, h=None):
		if(self.p_in): x = x.permute(0,2,1)
		h_out = x[:, :, -self.kernel_size+1:]
		ch = self.ch(x,h)
		conv1 = self.conv1(ch)
		ret = self.relu(conv1)
		if(self.training): ret = self.dropout(ret)
		if(self.p_out): ret = ret.permute(0,2,1)
		return ret, h_out
