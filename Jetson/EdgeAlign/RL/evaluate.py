import numpy as np
from EdgeAlign.Param.params import *
from EdgeAlign.Tools.alignment import Alignment
from EdgeAlign.Tools.SeqGen import SeqGen
import time
import tensorflow as tf

seqgen = SeqGen()
align = Alignment()

# Initialize the interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_type = input_details['dtype']
#print('input: ', input_type)
output_type = output_details['dtype']
#print('output: ', output_type)

tot_time = 0

# Helper function to run inference on a TFLite model
def run_tflite_model(state):

  # Check if the input type is quantized, then rescale input data to uint8
  if input_details['dtype'] == np.uint8:
    input_scale, input_zero_point = input_details["quantization"]
    state = state / input_scale + input_zero_point

  state = np.expand_dims(state, axis=0).astype(input_details["dtype"])
  interpreter.set_tensor(input_details["index"], state)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details["index"])[0]
  #print(output)

  action = output[1: ].argmax()
  return action

class EdgeAlignEvaluate():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.model = None
        pass

    def preprocess(self, seq1, seq2):
        s1, s2, length = seqgen.lcs(seq1, seq2)
        seqs = {'s1': s1, 's2': s2, 'lcs': length}
        # Reverse sequences
        if(s1 > 0 and s2 > 0):
            seq1_R = seq1[s1 - 1::-1]
            seq2_R = seq2[s2 - 1::-1]
            seqs['r'] = [seq1_R, seq2_R]
        # Forward sequences
        if(s1 + length < len(seq1)) and (s2 + length < len(seq2)):
            seq1_F = seq1[s1 + length: ]
            seq2_F = seq2[s2 + length: ]
            seqs['f'] = [seq1_F, seq2_F]
        seqs['c'] = seq1[s1: s1+length]

        return seqs

    def subAlign(self, seq1, seq2):
        global tot_time
        self.x = 0
        self.y = 0
        seq1_s = ""
        seq2_s = ""
        action_s = ""
        align.reset(seq1, seq2)
        state = align.renderSeq()

        done = False
        while(not done):
            state = np.reshape(state, (1, state.size))  #(1, 1, state.size)
            start = time.time()
            #a = np.argmax(self.model.predict(state))#np.random.randint(0, n_actions)
            a = run_tflite_model(state)
            end = time.time()
            tot_time += (end -start)
            state, reward, done = align.updateSeq(a)

            if(a==0):
                seq1_s += BP[seq1[self.x]]
                seq2_s += BP[seq2[self.y]]
                if(seq1[self.x] == seq2[self.y]):
                    action_s += '*'
                else:
                    action_s += ' '
                self.x += 1
                self.y += 1
            
            elif(a==1):
                seq1_s += '-'
                seq2_s += BP[seq2[self.y]]
                action_s += ' '
                self.y += 1

            elif(a==2):
                seq1_s += BP[seq1[self.x]]
                seq2_s += '-'
                action_s += ' '
                self.x += 1
        
        if(self.x < len(seq1)):
            for k in seq1[self.x: ]:
                seq1_s += BP[k]
                seq2_s += '-'
                action_s += ' '
        elif(self.y < len(seq2)):
            for k in seq2[self.y: ]:
                seq2_s += BP[k]
                seq1_s += '-'
                action_s += ' '

        return seq1_s, seq2_s, action_s
    
    def count(self, s):
        return sum(map(s.count, BP))
        
    def align(self, seq1=[], seq2=[], filename='alignment.txt'):
        #temp = np.zeros((1, 1, n_pixels*(window+2)*n_pixels*4*3))
        #a = np.argmax(self.model.predict(temp))
        #temp = np.zeros((1, n_pixels*(window+2)*n_pixels*4*3))
        temp = np.zeros((1, n_pixels*window*n_pixels*2*3))
        a = run_tflite_model(temp)
        print("Alignment started...")

        if (len(seq1)==0) or (len(seq2)==0):
            print("Empty sequence identified! Please try again :)")
            return "","",""
        else:
            if (isinstance(seq1, str)):
                seq1 = [BP.index(k) for k in seq1]
            if (isinstance(seq2, str)):
                seq2 = [BP.index(k) for k in seq2]
            seqs = self.preprocess(seq1, seq2)
            seq1_s = ""
            seq2_s = ""
            action_s = ""
            bnd = 50

            global tot_time
            tot_time = 0

            for k in seqs['c']:
                seq1_s += BP[k]
                seq2_s += BP[k]
                action_s += '*'

            if 'r' in seqs.keys():
                seq1_r = seqs['r'][0]
                seq2_r = seqs['r'][1]
                s1, s2, a = self.subAlign(seq1_r, seq2_r)
                seq1_s = s1[::-1] + seq1_s
                seq2_s = s2[::-1] + seq2_s
                action_s = a[::-1] + action_s
            else:
                if(seqs['s1'] == 0 and seqs['s2'] > 0):
                    seq2_s = "".join([BP[k] for k in seq2[: seqs['s2']]]) + seq2_s
                    seq1_s = '-'*seqs['s2'] + seq1_s
                    action_s = ' '*seqs['s2'] + action_s
                elif(seqs['s2'] == 0 and seqs['s1'] > 0):
                    seq1_s = "".join([BP[k] for k in seq1[: seqs['s1']]]) + seq1_s
                    seq2_s = '-'*seqs['s1'] + seq2_s
                    action_s = ' '*seqs['s1'] + action_s

            if 'f' in seqs.keys():
                seq1_f = seqs['f'][0]
                seq2_f = seqs['f'][1]
                s1, s2, a = self.subAlign(seq1_f, seq2_f)
                seq1_s += s1
                seq2_s += s2
                action_s += a
            else:
                s1 = seqs['s1']
                s2 =seqs['s2']
                lcs = seqs['lcs']
                if(s1 + lcs >= len(seq1)) and (s2 + lcs < len(seq2)):
                    seq2_s += "".join([BP[k] for k in seq2[s2+lcs: ]])
                    seq1_s += '-'*len(seq2[s2+lcs: ])
                    action_s += ' '*len(seq2[s2+lcs: ])
                elif(s1 + lcs < len(seq1)) and (s2 + lcs >= len(seq2)):
                    seq1_s += "".join([BP[k] for k in seq1[s1+lcs: ]])
                    seq2_s += '-'*len(seq1[s1+lcs: ])
                    action_s += ' '*len(seq1[s1+lcs: ])

            print("Total Matches : {} | Total time elapsed(s) : {}".format(action_s.count('*'), tot_time))

            result = ""
            i = 0
            while(i < len(action_s)):
                if (i + bnd) < len(action_s):
                    result += "test1      " + seq1_s[i: i+bnd] + " " + str(self.count(seq1_s[: i+bnd])) + '\n' + "test2      " + seq2_s[i: i+bnd] + " " + str(self.count(seq2_s[: i+bnd])) + '\n' + "           " + action_s[i: i+bnd]
                    result += '\n\n'
                else:
                    result += "test1      " + seq1_s[i: ] + " " + str(str(self.count(seq1_s))) + '\n' + "test2      " +  seq2_s[i: ] + " " + str(str(self.count(seq2_s))) + '\n' + "           " + action_s[i: ]
                i += bnd

            with open(filename, 'w') as f:
                f.write(result)

            return seq1_s, seq2_s, action_s, action_s.count('*'), tot_time
