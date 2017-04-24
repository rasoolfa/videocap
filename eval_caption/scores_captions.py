'''
This original codes are downloaded from https://github.com/tylin/coco-caption
However, it has been updated to work with video captioning dataset
Input to this function is ground truth and the result:
Example:
res_table = {380932: [{u'caption': u'group of people are on the side of a snowy field',
   u'image_id': 380932}],
 404464: [{u'caption': u'black and white photo of a man standing in front of a building',
   u'image_id': 404464}] }

gts_table {203564: [{u'caption': u'A bicycle replica with a clock as the front wheel.',
   u'id': 37,
   u'image_id': 203564},
  {u'caption': u'The bike has a clock as a tire.',
   u'id': 181,
   u'image_id': 203564}],
 322141: [{u'caption': u'A room with blue walls and a white sink and door.',
   u'id': 49,
   u'image_id': 322141},
  {u'caption': u'A blue boat themed bathroom with a life preserver on the wall',
   u'id': 163,
   u'image_id': 322141}]} }

   To use:
   from scores_captions import *
   myeval = score_captions(gt_dict, res_dict, key_name = 'image_id')
   myeval.evaluate()

'''
import os
import json
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
#from spice.spice import Spice

class score_captions:
    def __init__(self, gts_table, res_table, no_print = True,  key_name = 'video_id'):
            self.evalImgs = []
            self.eval = {}
            self.key_name = key_name
            self.imgToEval = {}
            self.ground_captions = gts_table
            self.input_captions =  res_table 
            self.no_print = no_print

    def evaluate(self):
        
        gts = {}
        res = {}
        counter = 0
        for i in self.input_captions['v_preds']:
            imgId = i[self.key_name]
            if imgId not in res:
                res[imgId] = []
            res[imgId].append(i) 
            gts[imgId] = self.ground_captions[imgId]


        # =================================================
        # Set up scorers
        # =================================================
        if self.no_print == False:
            print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if self.no_print == False:
            print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            if self.no_print == False:
                print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    if self.no_print == False:
                        print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                if self.no_print == False:
                    print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

        res_diff_method = {} 
        for metric, score in self.eval.items():
            score_round ='%.3f'%(score)
            res_diff_method[metric] = float(score_round)

        return res_diff_method  


    def setEval(self, score, method):
            self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
            for imgId, score in zip(imgIds, scores):
                if not imgId in self.imgToEval:
                    self.imgToEval[imgId] = {}
                    self.imgToEval[imgId][self.key_name] = imgId
                self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
            self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

def read_json_file(input_json):
    ## load the json file 
    file_info = json.load(open(input_json, 'r'))

    return file_info


import argparse

if __name__ == "__main__":

    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    parser = argparse.ArgumentParser()
    parser.add_argument('--f_res', required=True )
    parser.add_argument('--f_gt', required=True )
    args = parser.parse_args()
    res_dict = read_json_file(args.f_res)
    gt_dict  = read_json_file(args.f_gt)

    myeval = score_captions(gt_dict, res_dict, key_name = 'video_id')
    result = myeval.evaluate()
    import ntpath
    temp_name ='c'+ ntpath.basename(args.f_res)
    temp_name = os.path.join(ntpath.dirname(args.f_res),temp_name)
    json.dump(result, open(temp_name, 'w'))
    print("Socres are in",temp_name)
