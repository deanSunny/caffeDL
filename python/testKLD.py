import scipy.stats
import numpy as np

def kl(p):
	q = [0.5, 0.5]
	s = scipy.stats.entropy(p, q)
	pk = np.asarray(p, dtype=np.float)
	qk = np.asarray(q, dtype=np.float)
	
	
	skl = np.sum(np.where(pk != 0, pk * np.log10(pk / qk), 0))
	return s, skl


if __name__ == '__main__':
	p = [0.4999, 0.5001]
	sc, skl = kl(p)
	print sc, skl
