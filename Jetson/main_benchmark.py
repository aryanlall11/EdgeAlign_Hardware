import os
import numpy as np
from EdgeAlign.Param.params import *
from EdgeAlign.RL.agent import EdgeAlignAgent
from EdgeAlign.RL.environment import EdgeAlignEnv
from EdgeAlign.Tools.model import Model
from EdgeAlign.RL.evaluate import EdgeAlignEvaluate
from EdgeAlign.Tools.SeqGen import *
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import pandas as pd

SAMPLES = 100

seqgen = SeqGen()
eval = EdgeAlignEvaluate()
data = {'Sample Index':[], 'RL Matches':[], 'BLAST Matches':[], 'RL Time(s)':[], 'BLAST Time(s)':[]}

i = 0
while(i < SAMPLES):
	# Generate random DNA test sequences
	seq1, seq2 = seqgen.genSequences()
	seqgen.saveSequences(seq1=seq1, seq2=seq2, filename='Samples/sample' + str(i) + '.txt')

	seq1 = "".join([BP[k] for k in seq1])
	seq2 = "".join([BP[k] for k in seq2])

	#print(match)
	#print(tot_time)
	"""-----------------------------------------------------------------------------"""
	with open('seq1.fasta', 'w') as f:
		f.write('>seq0\n' + seq1)
	f.close()
	with open('seq2.fasta', 'w') as f:
		f.write('>seq0\n' + seq2)
	f.close()

	os.system('(time blastn -query seq1.fasta -subject seq2.fasta > out.txt) 2> time.txt')

	with open('out.txt', 'r') as f:
		lines = f.readlines()
		for line in lines:
			if(('Identities' in line) or ('No hits found' in line)):
				break
	if('No hits found' in line):
		continue

	line = line[: line.find('/')]
	line = line.split('=')[1]
	blast_match = int(line)

	with open('time.txt', 'r') as f:
		line = f.readline()
		line = line.split(' ')[: 2]

	t1 = float(line[0].replace('user', ''))
	t2 = float(line[1].replace('system', ''))

	data['BLAST Matches'].append(blast_match)
	data['BLAST Time(s)'].append(t1+t2)
	#print("BLAST Matches: ", blast_match)
	#print(t1 + t2)

	"""-----------------------------------------------------------------------------"""
	# Evaluate
	print(str(i+1) + '/' + str(SAMPLES))
	data['Sample Index'].append(i+1)

	_, _, _, match, tot_time = eval.align(seq1=seq1, seq2=seq2, filename='alignment.txt')
	print("BLAST Matches : {} | BLAST time(s) : {}".format(blast_match, t1 + t2))
	data['RL Matches'].append(match)
	data['RL Time(s)'].append(tot_time)
	i += 1

df = pd.DataFrame(data)   
df.to_csv('Benchmarked_Jetson.csv', index=False, columns=["Sample Index","RL Matches", "BLAST Matches", "RL Time(s)",  "BLAST Time(s)"])

