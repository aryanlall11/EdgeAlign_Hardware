N = 15     # 15 x 15 image
k = 3      # 3 x 3 kernel
out_size = N-k+1     # Stride = 1, No padding

s = ""

for x in range(k):
	for y in range(out_size):
		a = x*out_size + y
		st = str(x) + ", " + str(y) + ", "
		st += "stream_ker[%d]"%(x*(out_size+1) + y) + ", " + "stream_ker[%d]"%(x*(out_size+1) + y+1) + ", "
		st += ("stream_in[%d]"%(x+y) if (x==k-1 or y==0) else "stream_fea[%d]"%((x+1)*out_size + y-1)) + ", " + "stream_fea[%d]"%(x*out_size + y) + ", "
		st += "stream_acc[%d]"%((x+1)*out_size + y) + ", " + "stream_acc[%d]"%(x*out_size + y)

		st = "ProcessingElement(" + st + ");\n"

		s += st

with open("pe.txt", 'w') as f:
	f.write(s)
f.close()