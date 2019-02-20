import numpy as np

def pq_to_fg(p, q):
	f = np.divide(2*p, 1 + np.sqrt((1 + p*p + q*q)))
	g = np.divide(2*q, 1 + np.sqrt((1 + p*p + q*q)))

	return f, g

def fg_to_pq(f, g):
	p = np.divide(4*f, 4 - f*f - g*g)
	q = np.divide(4*g, 4 - f*f - g*g)

	return p, q