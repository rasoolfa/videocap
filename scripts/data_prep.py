"""
This function prepares the data to be consumed by the DataLayer 
Input paramters:
  	--input_json: a json file.
	    The json files must have the following keys:
	     {'captions': [u'a woman in purple', u'a woman in a purple blouse stands'], 
	     'video_id': 'video103'},
	     It can have other parts but those are essential     
	--num_val: Number of images to assign to validation data
	--output_json: output json file
	--output_h5: output h5 file
	--max_length: Max number of words in a caption. captions longer than this get clipped.')
	 if max_lengh is -1, the code doesn't clip the captions 
	--frame_dir: root location in which video frames are stored
	--word_count_threshold: only words that occur more than this number of times will be put in vocab
	--num_test: number of test images (to withold until very very end)
	--encoding: encoding for the captions
Cite: part of this code is adopted from https://github.com/karpathy/neuraltalk2/blob/master/prepro.py	
"""

from __future__ import print_function
import os
import json
import numpy as np
import string
import argparse
import h5py
from scipy.misc import imread, imresize
from utils import read_json_file
import random 


def get_UNK():
	"""
	 Will be sued for infreq words
	"""
	return 'UNK'

def prepro_captions(vids, encoding):
	"""
	 This function preprocess the captions, i.e. remove caps, punctuation, and etc. 
	"""
	import sys
	cwd = os.getcwd()
	print(cwd)
	sys.path.append(os.path.abspath(cwd))
	print(sys.path)
	from eval_caption.tokenizer.ptbtokenizer import PTBTokenizer
	tokenizer = PTBTokenizer()

	temp_vid = {}
	cp_temp ={}

	##############################
	# Input for PTBTokenizer should be sth like following:
	# captions_for_image ={"vid44": [{"caption": "A person ", "video_id": "vid44"}, {"caption": "A cuts.", "video_id": "vid44"},],
	#                      "vid52": [{"caption": "Another One", "video_id": "vid52"}, {"caption": "Just meat.", "video_id": "vid52"}]}
	# and output {'vid44': ['a person', 'a cuts'], 'vid52': ['another one', 'just meat']} 
	# In the followings, just created a temp_vid for this function and then send to the PTBTokenizer. This is Lazy implemnation
	##############################
	for  v in vids:
		v_id_t = v['video_id']
		temp_vid[v_id_t] = []
		for j in v['captions']:
			cp_temp['caption'] = j.encode(encoding)
			temp_vid[v_id_t].append(cp_temp)
			cp_temp = {}
	tokenized_by_corenlp = tokenizer.tokenize(temp_vid)
	if 'vid237' in tokenized_by_corenlp:
		print(tokenized_by_corenlp['vid237'])
	else:
		op_tmp = tokenized_by_corenlp.keys()[0]
		print(tokenized_by_corenlp[op_tmp])
	print("\nSome exmaples of tokenization:")
	
	random_indx = {i:1 for i in np.random.randint(0, len(vids), 10)}
	for i, v in enumerate(vids):
		v['processed_tokens'] = []
		curr_vid = v['video_id']
		for j, s in enumerate(tokenized_by_corenlp[curr_vid]):
			#token = str(s.encode(encoding)).lower().translate(None, string.punctuation).strip().split()
			token = s.split()
			v['processed_tokens'].append(token)
			if i in random_indx and j ==0:
				print(token)

def build_vocab(vids, params):
	"""
	 This function builds the vocab and remove infrequent words
	"""
	w_count_thr = params.word_count_threshold
   
	# Let's collect some info about each works
	word_counter = {}
	for v in vids:
		for cap in v['processed_tokens']:
			for w in cap:
				word_counter[w] = word_counter.get(w, 0) + 1
	c_sorted = sorted([(c,w) for w, c in word_counter.iteritems()], reverse=True)	
	print("\nTop words in these dataset's captions")
	print('\n'.join(map(str, c_sorted[:20])))
	print('\n'.join(map(str, c_sorted[-20:-1])))


	len_captions = {} # each len would be one key in dict 
	for v in vids:
		for cap in v['processed_tokens']:
			if len(cap) in len_captions:
				len_captions[len(cap)] += 1
			else:
				len_captions[len(cap)] = 1

	max_len_cap = max(len_captions.keys())
	sum_len  = sum(len_captions.values())
	# Now print out some statistics	
	total_ws = sum(word_counter.values())
	infreq_w = [ w for w, c in word_counter.iteritems() if c <= w_count_thr]
	vocab =    [ w for w, c in word_counter.iteritems() if c >  w_count_thr]
	total_infreq = sum( word_counter[w] for w in infreq_w)
	print("Total words:", total_ws)
	print('Total of infreq words: %d/%d = %.2f%%' % (len(infreq_w), len(word_counter), len(infreq_w)*100.0/len(word_counter)))
 	print('Number of words in vocab would be %d' % (len(vocab) ))
 	print('Number of UNKs: %d/%d = %.2f%%' % (total_infreq, total_ws, total_infreq*100.0/total_ws))

 	print('Max_len sentence in this raw data: ', max_len_cap)
  	print('Sentence len distribution (count, number of words):')
  	for l, c in len_captions.items():
  		 print('%2d: %10d   %f%%' % (l, c, c*100.0/sum_len))

  	# Add UNK to vocab
  	if (total_infreq > 0):
  		vocab.append(get_UNK())

  	for v in vids:
  		v['final_captions'] = []
  		for cap in v['processed_tokens']:
	  		final_captions = []
  			for w in cap:
  				# replace UNK with infreq words
  				if word_counter[w] > w_count_thr:
  					final_captions.append(w)
  				else:
  					final_captions.append(get_UNK())
            #after processing each captions, add them to the dict
  			v['final_captions'].append(final_captions)
  	return vocab	

def encode_captions(vids, params, wtoi, dtype):
	"""
	This functions codes all captions in one long array that can be used in torch.
	In addation, we built start and end caption holders that point to the start of a caption 
	and end of it.
	idea adpoted from: https://github.com/karpathy/neuraltalk2/blob/master/prepro.py
	"""

	max_len = params.max_length
	total_captions = sum([ len(c['final_captions']) for c in vids]) 
	N = len(vids) #total number of examples
	print("\nThere are total of %d samples and %d captions in this dataset" %(N, total_captions))

	# just a counter count number of captions
	caps_counter = 0

	#For each videos, the all_labels rows contains n*max_length where n is number of captions in current sample 
	all_labels = []
	
	#label_start_idx and label_end_idx keeps the the start and end index for each captions 
	# Since the result will be used, we do one indexing 
	label_start_idx = np.zeros(N, dtype = dtype) 
	label_end_idx  = np.zeros(N, dtype = dtype)
	counter_caps = 1
	label_len = np.zeros(total_captions, dtype = dtype) # used to keep length of sequence
	
	#####
	# Since the np.concatenate needs to words_cur_vid have same number of columns 
	# first the max caption length among all captions and set that one as the size words_cur_vid
	# number of columns
	#####
	max_len_all_list = []
	for idx, v in enumerate(vids):
		captions = v['final_captions']
		max_len_all_list.append(max([len(m_c) for m_c in captions]))
	max_len_all = max(max_len_all_list)	

	for idx, v in enumerate(vids):
		#some sanity check to make sure that everyone has atleast one caption
		captions = v['final_captions']
		len_cur_caps = len(captions) 
		assert len_cur_caps > 0, "No caption for %s sample" % (v['video_id'])

		if max_len == -1:
			#Set the max of all captions
			words_cur_vid = np.zeros((len_cur_caps, max_len_all), dtype=dtype)
		else:
			words_cur_vid = np.zeros((len_cur_caps, max_len), dtype=dtype)
		
		for j, cap in enumerate(captions):

			# keeps track of each caption length. If the caption larger than max_len, it will be clipped
			if max_len == -1: # no clipping 
				label_len[caps_counter] = len(cap)
			else:
				label_len[caps_counter] = min(max_len, len(cap))	

			caps_counter += 1

			#now we start clipping: max number of words in a caption should be < max_len. Captions longer than this get clipped.
			for k, w in enumerate(cap):

				if max_len !=-1 and k < max_len: # cliping
					words_cur_vid[j,k] = wtoi[w]
				elif max_len == -1: # no clipping
					words_cur_vid[j,k] = wtoi[w]

		all_labels.append(words_cur_vid)
		label_start_idx[idx] = counter_caps
		label_end_idx[idx] = counter_caps + len_cur_caps - 1
		counter_caps += len_cur_caps 

    # create a long list of all labels: each row is n*max_length where n is number of captions in current sample
	L = np.concatenate(all_labels, axis = 0)  

	# sanity check 
	assert L.shape[0] ==  total_captions, 'The length(L) and total_captions not match?'
	assert np.all(label_len > 0 ), 'Some captions are empty??'

	print ("Size of encoded captions is", L.shape)

	return L, label_start_idx, label_end_idx, label_len
def split_data(vids, params):
	"""
	 This function just split the data to val/test/train
	"""
	count_val = 0
	count_test = 0
	count_train = 0
	for i, v in enumerate(vids):
		v['split'] = []
		if 'i_split' in v:
			v['split'] = v['i_split']
			if v['i_split'] == 'test':
				count_test = count_test + 1
			elif v['i_split'] == 'train':
				count_train = count_train + 1
			elif v['i_split'] == 'val':
				count_val = count_val + 1
			else:
				raise ValueError("Sth wrong with split", v['i_split'] )
		else:
			if params.only_test == 1:  # if process test data  
				v['split'] = 'test'
				count_test = count_test + 1
			else:                      # process training data and create train/val/test splits
				if i < params.num_val:
					v['split'] = 'val'
					count_val = count_val + 1
				elif i < params.num_val + params.num_test:
					v['split'] = 'test'
					count_test = count_test + 1
				else:
					v['split'] = 'train'
					count_train = count_train + 1

	if params.only_test == 1:	
		print ("This dataset will be used only for TEST puprpose. %d assigned to test"% len(vids) )
	else:				
		print ("%d assigned to test, %d to val, and %d to train" % (count_test, count_val, count_train) )		

def get_framrate_msrvtt(sample_video, params):
	"""
	 return the frame rate of MSR_VTT
	 this function is MSR_VTT specific
	"""
	return np.shape(np.load(params.frame_dir+'/'+sample_video['video_id']+params.v_f_name_ext)['arr_0'])[0]

def get_framrate_HW(sample_video, params):
	"""
	 return the frame rate of Hollywood
	 this function is Hollywood specific
	"""
	return np.shape(np.load(params.frame_dir+'/'+sample_video['video_id']+params.v_f_name_ext)['arr_0'])[0]

def get_framrate_YT(sample_video, params):
	"""
	 return the frame rate of YT
	 this function is YT specific
	"""
	return np.shape(np.load(params.frame_dir+'/'+sample_video['video_id']+params.v_f_name_ext)['arr_0'])[0]

def resize_video(params, frame_rate, input_vid):
	"""
	Need to loop over video frames to resize it
	"""
	num_channel = 3 # thsi one should be a parameters
	resized_vid = np.zeros((frame_rate, num_channel, params.img_size, params.img_size))

	for i in range(frame_rate):
		tmp = imresize(input_vid[i], (params.img_size, params.img_size))
		resized_vid[i] = tmp.transpose(2,0,1)

	return resized_vid	

def main(params):
	"""
	Here is the pipeline
	- shuffle the videos
	- build vocab
	- preprocess captions, i.e. remove caps, punctuation, and etc
	- encode the captions
	- read the video frames and save them in h5py
	-
	"""
	dtype = 'uint32'
	dtype_vid = 'uint8' # this should be a parameter
	num_channel = 3     # this should be calculated from input data

	videos = read_json_file(params.input_json)
	random.seed(123)
	random.shuffle(videos) 

	######
	#preprocess the captions
	######
	prepro_captions(videos, params.encoding)

	######
	#Build the vocabs
	######
	vocabs = build_vocab(videos, params)

	#word to id --since it later will be used in torch, we start index for this from one
	wtoi = { w: idx+1 for idx, w in enumerate(vocabs)}
	#id to word
	itow = { idx+1:w for idx, w in enumerate(vocabs)}
	
	#######
	# Encode captions
	#######
	L, l_start_idx, l_end_idx, label_len = encode_captions(videos, params, wtoi, dtype)

	######
	# Split data
	#####
	split_data(videos, params)

	######
	# Now everything is inplace and time to create h5 files
	# Each video frame is saved in a npz file, i.e. video67_f.npz 
	# _f should be added to the video id to read the file.
	# In addtion, we save some information to the jason file.
	######

    #json part 
	dict_to_json = {}
	dict_to_json['ix_to_word'] = itow
	dict_to_json['word_to_ix'] = wtoi
	dict_to_json['video'] = []

	#h5 file
	h5_f = h5py.File(params.output_h5, 'w')
	h5_f.create_dataset('label_start_idx', dtype =dtype, data =l_start_idx)
	h5_f.create_dataset('label_end_idx', dtype =dtype, data =l_end_idx)
	h5_f.create_dataset('label_len', dtype =dtype, data =label_len)
	h5_f.create_dataset('labels', dtype =dtype, data =L)

	#Since we are dealing with videos, we need to get numebr of frame per video
	#Each video has the following format : 16*w*H*channel 
	if (params.dataset_name == 'msr_vtt'):
		frame_rate = get_framrate_msrvtt(videos[np.random.randint(len(videos))], params)
	elif ('HW'.lower() in params.dataset_name.lower() ):
		frame_rate = get_framrate_HW(videos[np.random.randint(len(videos))], params)
	elif ('YT'.lower() in params.dataset_name.lower() ):
		frame_rate = get_framrate_YT(videos[np.random.randint(len(videos))], params)
	else:
		raise ValueError("No support for %s dataset yet" % params.dataset_name)

	N = len(videos)	
	vid_data = h5_f.create_dataset("videos", (N,frame_rate, num_channel, params.img_size, params.img_size), dtype = dtype_vid)	

	for idx, vid in enumerate(videos):
		
		#read each video and resize it 
		i_file = params.frame_dir+'/'+vid['video_id']+params.v_f_name_ext
		assert os.path.exists(i_file) , "The %s file is not there, something is worng" % (i_file) 

		input_vid = np.load(i_file)['arr_0']
		vid_data[idx] = resize_video(params, frame_rate, input_vid)
		# show some progress
		if idx % 15 == 0:
			print("%d out of %d have been processed" % (idx + 1, len(videos)))

		# add data to a dict to be dumped into a json final	
		dict_to_json['video'].append(vid)  
			
	h5_f.close()
	print("Done with H5 file")
	json.dump(dict_to_json, open(params.output_json, 'w'))
	print("Done with everything")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# input json
	parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
	parser.add_argument('--output_json', default='msrvtt_final.json', help='output json file')
	parser.add_argument('--output_h5', default='msrvtt.h5', help='output h5 file')
	parser.add_argument('--frame_dir', required=True, help='root location in which video frames are stored')
	parser.add_argument('--max_length',required=True, type=int, help='max number of words in a caption. captions longer than this get clipped. -1 means no clipping')
	parser.add_argument('--only_test', required=True, type=int, default=0, help='This is used when use this code for process only test data 0|1(means test only)' )

	# options
	parser.add_argument('--word_count_threshold', default= 5, type=int, help='only words that occur more than this number of times will be put in vocab')
	parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
	parser.add_argument('--num_val',  default=0, type=int, help='number of images to assign to validation data')

	parser.add_argument('--encoding', default='utf-8')
	parser.add_argument('--img_size', default=256, type =int, help = 'Each frame will be resize to this size')
	parser.add_argument('--v_f_name_ext', default='_f.npz', help = 'the format and extension of saved video name, e.g. video10_f.npz')
	parser.add_argument('--dataset_name', default='msr_vtt', help = 'name of input dataset, only support msr_vtt as of now')

	args = parser.parse_args()
	print('\nParsed input parameters:')
	print(json.dumps(vars(args), indent = 2))
	main(args)