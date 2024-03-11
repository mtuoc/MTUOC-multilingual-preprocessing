#    MTUOC-multilingual-preprocessing
#    Copyright (C) 2024  Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
from datetime import datetime
import os
import glob
import codecs
import importlib
import importlib.util
import re
import random

import pickle

from shutil import copyfile

import yaml
from yaml import load, dump

from itertools import (takewhile,repeat)



import sentencepiece as spm

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
def file_len(fname):
    num_lines = sum(1 for line in open(fname))
    return(num_lines)
    
def findEMAILs(string): 
    email=re.findall('\S+@\S+', string)   
    return email
    
def findURLs(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
    
def replace_EMAILs(string,code="@EMAIL@"):
    EMAILs=findEMAILs(string)
    cont=0
    for EMAIL in EMAILs:
        string=string.replace(EMAIL,code)
    return(string)

def replace_URLs(string,code="@URL@"):
    URLs=findURLs(string)
    cont=0
    for URL in URLs:
        string=string.replace(URL,code)
    return(string)

def split_corpus(filename,valsize,evalsize,slcode,tlcode):
    count=rawincount(filename)
    numlinestrain=count-valsize-evalsize
    numlinestrain2=numlinestrain
    if numlinestrain<0: numlinestrain2=0
    entrada=codecs.open(filename,"r",encoding="utf-8")
    filenametrain="train-"+slcode+"-"+tlcode+".txt"
    sortidaTrain=codecs.open(filenametrain,"w",encoding="utf-8")
    filenameval="val-"+slcode+"-"+tlcode+".txt"
    sortidaVal=codecs.open(filenameval,"w",encoding="utf-8")
    filenameeval="eval-"+slcode+"-"+tlcode+".txt"
    sortidaEval=codecs.open(filenameeval,"w",encoding="utf-8")
    cont=0
    for linia in entrada:
        if cont < numlinestrain:
            sortidaTrain.write(linia)
        elif cont>= numlinestrain2 and cont < numlinestrain2+valsize:
            sortidaVal.write(linia)
        else:
            sortidaEval.write(linia)
        cont+=1
    sortidaTrain.close()
    sortidaVal.close()
    sortidaEval.close()

def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

  
       
def sentencepiece_train(corpusL1,L1code2,SP_MODEL_PREFIX="spmodel",MODEL_TYPE="bpe",VOCAB_SIZE=32000,CHARACTER_COVERAGE=1,INPUT_SENTENCE_SIZE=1000000,SPLIT_DIGITS=True,CONTROL_SYMBOLS="",USER_DEFINED_SYMBOLS=""):
    options=[]
    if SPLIT_DIGITS:
        options.append("--split_digits=true")
    else:
        options.append("--split_digits=false")
    if not CONTROL_SYMBOLS=="":
        options.append("--control_symbols=\""+CONTROL_SYMBOLS+"\"")
    if not USER_DEFINED_SYMBOLS=="":
        options.append("--user_defined_symbols=\""+USER_DEFINED_SYMBOLS+"\"") 
    options=" ".join(options)
    if True:
        train=corpusL1       
        command="spm_train --input=train --model_prefix="+SP_MODEL_PREFIX+" --model_type="+MODEL_TYPE+" --vocab_size="+str(VOCAB_SIZE)+" --character_coverage="+str(CHARACTER_COVERAGE)+" --split_digits --input_sentence_size="+str(INPUT_SENTENCE_SIZE)+" "+options
        print(command)
        os.system(command)
        command="spm_encode --model="+SP_MODEL_PREFIX+".model --generate_vocabulary < "+corpusL1+" > vocab_file."+L1code2
        os.system(command)
        
def sentencepiece_encode(corpusPre,OUTFILE,SP_MODEL,VOCABULARY,VOCABULARY_THRESHOLD=50, EOS=True, BOS=True):
    if EOS and BOS:
        extraoptions="--extra_options eos:bos"
    elif EOS:
        extraoptions="--extra_options eos"
    elif BOS:
        extraoptions="--extra_options bos"
    else:
        extraoptions=""
    command="spm_encode --model="+SP_MODEL+" "+extraoptions+" --vocabulary="+VOCABULARY+" --vocabulary_threshold="+str(VOCABULARY_THRESHOLD)+" < "+corpusPre+" > "+OUTFILE
    os.system(command)
def check_guided_alignment(SLcorpus,TLcorpus,forwardalignment):
    copyfile(SLcorpus,"slcorpustemp.txt")
    copyfile(TLcorpus,"tlcorpustemp.txt")
    copyfile(forwardalignment,"forwardalignmenttemp.txt")
    
    slcorpus=codecs.open("slcorpustemp.txt","r",encoding="utf-8")
    tlcorpus=codecs.open("tlcorpustemp.txt","r",encoding="utf-8")
    alignforward=codecs.open("forwardalignmenttemp.txt","r",encoding="utf-8")


    slcorpusmod=codecs.open(SLcorpus,"w",encoding="utf-8")
    tlcorpusmod=codecs.open(TLcorpus,"w",encoding="utf-8")
    alignforwardmod=codecs.open(forwardalignment,"w",encoding="utf-8")
    
    
    
    
    cont=0
    while 1:
        cont+=1
        liniaSL=slcorpus.readline().rstrip()
        if not liniaSL:
            break
        liniaTL=tlcorpus.readline().rstrip()
        liniaalignforward=alignforward.readline().rstrip()

        tokensSL=liniaSL.split(" ")
        tokensTL=liniaTL.split(" ")
        tokensAlignForward=liniaalignforward.split(" ")
        
        towrite=True
        for token in tokensAlignForward:
            camps=token.split("-")
            if not len(camps)==2:
                print("ERROR",cont)
                towrite=False
        
        if towrite:
            slcorpusmod.write(liniaSL+"\n")
            tlcorpusmod.write(liniaTL+"\n")
            alignforwardmod.write(liniaalignforward+"\n")
    
    os.remove("slcorpustemp.txt")
    os.remove("tlcorpustemp.txt")
    os.remove("forwardalignmenttemp.txt")

def guided_alignment_fast_align(MTUOC="/MTUOC",ROOTNAME_ALI="train.sp",ROOTNAME_OUT="train.sp",SL="en",TL="es",BOTH_DIRECTIONS=False,VERBOSE=True):
    if VERBOSE: print("Alignment using fast_align:",ROOTNAME_ALI,SL,TL)
    sys.path.append(MTUOC)
    #from MTUOC_check_guided_alignment import check_guided_alignment
    FILE1=ROOTNAME_ALI+"."+SL
    FILE2=ROOTNAME_ALI+"."+TL
    FILEOUT="corpus."+SL+"."+TL+"."+"fa"
    FORWARDALI=ROOTNAME_OUT+"."+SL+"."+TL+".align"
    REVERSEALI=ROOTNAME_OUT+"."+TL+"."+SL+".align"
    command="paste "+FILE1+" "+FILE2+" | sed 's/\t/ ||| /g' > "+FILEOUT
    if VERBOSE: print(command)
    os.system(command)
    command=MTUOC+"/fast_align -vdo -i corpus."+SL+"."+TL+".fa > forward."+SL+"."+TL+".align"
    if VERBOSE: print(command)
    os.system(command)
    command=MTUOC+"/fast_align -vdor -i corpus."+SL+"."+TL+".fa > reverse."+SL+"."+TL+".align"
    if VERBOSE: print(command)
    os.system(command)
    command=MTUOC+"/atools -c grow-diag-final -i forward."+SL+"."+TL+".align -j reverse."+SL+"."+TL+".align > "+FORWARDALI
    if VERBOSE: print(command)
    os.system(command)
    if VERBOSE: print("Checking guided alignment")
    check_guided_alignment(FILE1,FILE2,ROOTNAME_OUT+"."+SL+"."+TL+".align")
    listfiles = os.listdir(".")
    try:
        os.remove(FILEOUT)
    except:
        pass
    try:
        os.remove("forward."+SL+"."+TL+".align")
    except:
        pass
    try:
        os.remove("reverse."+SL+"."+TL+".align")
    except:
        pass
    try:
        os.remove("forward."+TL+"."+SL+".align")
    except:
        pass
    try:
        os.remove("reverse."+TL+"."+SL+".align")
    except:
        pass

def guided_alignment_eflomal(MTUOC="/MTUOC",ROOTNAME_ALI="train.sp",ROOTNAME_OUT="train.sp",SL="en",TL="es",SPLIT_LIMIT=1000000,VERBOSE=True):
    if VERBOSE: print("Alignment using eflomal:",ROOTNAME_ALI,SL,TL)
    sys.path.append(MTUOC)
    from MTUOC_check_guided_alignment import check_guided_alignment
    FILE1=ROOTNAME_ALI+"."+SL
    FILE2=ROOTNAME_ALI+"."+TL
    FILEOUT="corpus."+SL+"."+TL+"."+"fa"
    command="paste "+FILE1+" "+FILE2+" | sed 's/\t/ ||| /g' > "+FILEOUT
    if VERBOSE: print(command)
    os.system(command)
    command="split -l "+str(SPLIT_LIMIT)+" "+FILEOUT+" tempsplitted-"
    if VERBOSE: print(command)
    os.system(command)
    listfiles = os.listdir(".")
    for file in listfiles:
        if file.startswith("tempsplitted-"):
            tempaliforward="tempaliforward-"+file.split("-")[1]
            tempalireverse="tempalireverse-"+file.split("-")[1]
            command=MTUOC+"/eflomal-align.py -i "+file+" --model 3 -f "+tempaliforward+" -r "+tempalireverse
            if VERBOSE: print(command)
            os.system(command)
    
    command="cat tempaliforward-* > "+ROOTNAME_OUT+"."+SL+"."+TL+".align"
    if VERBOSE: print(command)
    os.system(command)
    command="cat tempalireverse-* > todelete.align"
    if VERBOSE: print(command)
    os.system(command)
    if VERBOSE: print("Checking guided alignment")
    check_guided_alignment(FILE1,FILE2,ROOTNAME_OUT+"."+SL+"."+TL+".align")
    listfiles = os.listdir(".")
    os.remove("todelete.align")
    os.remove(FILEOUT)
    for file in listfiles:
        if file.startswith("tempsplitted-") or file.startswith("tempaliforward") or file.startswith("tempalireverse"):
            os.remove(file)
  





stream = open('config-multilingual-preprocessing.yaml', 'r',encoding="utf-8")
config=yaml.load(stream, Loader=yaml.FullLoader)
MTUOC=config["MTUOC"]
sys.path.append(MTUOC)

from MTUOC_train_truecaser import TC_Trainer
from MTUOC_truecaser import Truecaser
from MTUOC_splitnumbers import splitnumbers

VERBOSE=config["VERBOSE"]
LOG_FILE=config["LOG_FILE"]
DELETE_TEMP=config["DELETE_TEMP"]

corpora=config["corpora"].split(" ")
to_tags=config["to_tags"].split(" ")
L1codes3=config["L1codes3"].split(" ")
L1codes2=config["L1codes2"].split(" ")
L2codes3=config["L2codes3"].split(" ")
L2codes2=config["L2codes2"].split(" ")

evalsizes=config["evalsizes"].split(" ")
valsizes=config["valsizes"].split(" ")

L1_DICTS=config["L1_DICTS"].split(" ")
L2_DICTS=config["L2_DICTS"].split(" ")

L1_TOKENIZERS=config["L1_TOKENIZERS"].split(" ")
TOKENIZE_L1=config["TOKENIZE_L1"].split(" ")
L2_TOKENIZERS=config["L2_TOKENIZERS"].split(" ")
TOKENIZE_L2=config["TOKENIZE_L2"].split(" ")



TRAIN_L1_TRUECASER=config["TRAIN_L1_TRUECASER"].split(" ")
TRUECASE_L1=config["TRUECASE_L1"].split(" ")
L1_TC_MODELS=config["L1_TC_MODELS"].split(" ")

TRAIN_L2_TRUECASER=config["TRAIN_L2_TRUECASER"].split(" ")
TRUECASE_L2=config["TRUECASE_L2"].split(" ")
L2_TC_MODELS=config["L2_TC_MODELS"].split(" ")

#verify itegrity of data

len_corpora=len(corpora)

if not len(to_tags)==len_corpora:
    print("ERROR: len of to_tags not matching len of corpora")
    sys.exit()
if not len(L1codes3)==len_corpora:
    print("ERROR: len of L1codes3 not matching len of corpora")
    sys.exit() 
if not len(L1codes2)==len_corpora:
    print("ERROR: len of L1codes2 not matching len of corpora")
    sys.exit() 
if not len(L2codes3)==len_corpora:
    print("ERROR: len of L2codes3 not matching len of corpora")
    sys.exit() 
if not len(L2codes2)==len_corpora:
    print("ERROR: len of L2codes2 not matching len of corpora")
    sys.exit() 
if not len(L1_DICTS)==len_corpora:
    print("ERROR: len of L1_DICTS not matching len of corpora")
    sys.exit() 
if not len(L2_DICTS)==len_corpora:
    print("ERROR: len of L2_DICTS not matching len of corpora")
    sys.exit() 
if not len(L1_TOKENIZERS)==len_corpora:
    print("ERROR: len of L1_TOKENIZERS not matching len of corpora")
    sys.exit() 
if not len(TOKENIZE_L1)==len_corpora:
    print("ERROR: len of TOKENIZE_L1 not matching len of corpora")
    sys.exit() 
if not len(L2_TOKENIZERS)==len_corpora:
    print("ERROR: len of L2_TOKENIZERS not matching len of corpora")
    sys.exit() 
if not len(TOKENIZE_L2)==len_corpora:
    print("ERROR: len of TOKENIZE_L2 not matching len of corpora")
    sys.exit() 
if not len(TRAIN_L1_TRUECASER)==len_corpora:
    print("ERROR: len of TRAIN_L1_TRUECASER not matching len of corpora")
    sys.exit() 
if not len(TRUECASE_L1)==len_corpora:
    print("ERROR: len of TRUECASE_L1 not matching len of corpora")
    sys.exit() 
if not len(L1_TC_MODELS)==len_corpora:
    print("ERROR: len of L1_TC_MODEL not matching len of corpora")
    sys.exit() 

if not len(TRAIN_L2_TRUECASER)==len_corpora:
    print("ERROR: len of TRAIN_L2_TRUECASER not matching len of corpora")
    sys.exit() 
if not len(TRUECASE_L2)==len_corpora:
    print("ERROR: len of TRUECASE_L2 not matching len of corpora")
    sys.exit() 
if not len(L2_TC_MODELS)==len_corpora:
    print("ERROR: len of L2_TC_MODEL not matching len of corpora")
    sys.exit() 
if not len(evalsizes)==len_corpora:
    print("ERROR: len of evalsizes not matching len of corpora")
    sys.exit() 
if not len(valsizes)==len_corpora:
    print("ERROR: len of valsizes not matching len of corpora")
    sys.exit()
    
###PREPARATION
REPLACE_EMAILS=config["REPLACE_EMAILS"]
EMAIL_CODE=config["EMAIL_CODE"]
REPLACE_URLS=config["REPLACE_URLS"]
URL_CODE=config["URL_CODE"]

CLEAN=config["CLEAN"]
MIN_TOK=config["MIN_TOK"]
MAX_TOK=config["MAX_TOK"]

MIN_CHAR=config["MIN_CHAR"]
MAX_CHAR=config["MAX_CHAR"]


TRAIN_SENTENCEPIECE=config["TRAIN_SENTENCEPIECE"]

SAMPLE_SIZE=config["SAMPLE_SIZE"]
bos=config["bos"]
#<s> or None
eos=config["eos"]
#</s> or None
bosSP=True
eosSP=True
if bos=="None": bosSP=False
if eos=="None": eosSP=False
JOIN_LANGUAGES=config["JOIN_LANGUAGES"]
SPLIT_DIGITS=config["SPLIT_DIGITS"]
VOCABULARY_THRESHOLD=config["VOCABULARY_THRESHOLD"]

CONTROL_SYMBOLS=config["CONTROL_SYMBOLS"]
USER_DEFINED_SYMBOLS=config["USER_DEFINED_SYMBOLS"]
USER_DEFINED_SYMBOLS=USER_DEFINED_SYMBOLS+",".join(to_tags)+","
SP_MODEL_PREFIX=config["SP_MODEL_PREFIX"]
MODEL_TYPE=config["MODEL_TYPE"]
#one of unigram, bpe, char, word
VOCAB_SIZE=config["VOCAB_SIZE"]
CHARACTER_COVERAGE=config["CHARACTER_COVERAGE"]
CHARACTER_COVERAGE_SL=config["CHARACTER_COVERAGE_SL"]
CHARACTER_COVERAGE_TL=config["CHARACTER_COVERAGE_TL"]
INPUT_SENTENCE_SIZE=config["INPUT_SENTENCE_SIZE"]

#GUIDED ALIGNMENT
#TRAIN CORPUS
GUIDED_ALIGNMENT=config["GUIDED_ALIGNMENT"]
ALIGNER=config["ALIGNER"]
#one of eflomal, fast_align, simalign, awesome
DELETE_EXISTING=config["DELETE_EXISTING"]
SPLIT_LIMIT=config["SPLIT_LIMIT"]
#For efomal, max number of segments to align at a time

#VALID CORPUS
GUIDED_ALIGNMENT_VALID=config["GUIDED_ALIGNMENT_VALID"]
ALIGNER_VALID=config["ALIGNER_VALID"]
#one of eflomal, fast_align, simalign, awesome
DELETE_EXISTING_VALID=config["DELETE_EXISTING_VALID"]

DELETE_TEMP=config["DELETE_TEMP"]

if VERBOSE:
    logfile=codecs.open(LOG_FILE,"w",encoding="utf-8")

for i in range(0,len(corpora)):
    corpus=corpora[i] 
    to_tag=to_tags[i]
    L1code3=L1codes3[i]
    L1code2=L1codes2[i]
    L2code3=L2codes3[i]
    L2code2=L2codes2[i]
    tempfileL1="corpus-"+L1code3+".temp"
    tempfileL2="corpus-"+L2code3+".temp"
    if i==0:
        if os.path.exists(tempfileL1):
            os.remove(tempfileL1)
        if os.path.exists(tempfileL2):
            os.remove(tempfileL2)
    sortidaL1=codecs.open(tempfileL1,"a",encoding="utf-8")
    sortidaL2=codecs.open(tempfileL2,"a",encoding="utf-8")
    entrada=codecs.open(corpus,"r",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        sortidaL1.write(camps[0]+"\n")
        sortidaL2.write(camps[1]+"\n")
    entrada.close()
    sortidaL1.close()
    sortidaL2.close()
    command = "cat "+tempfileL1+" | shuf > tempfileL1.temp"
    os.system(command)
    command="mv tempfileL1.temp "+tempfileL1
    os.system(command)
    command = "cat "+tempfileL2+" | shuf > tempfileL2.temp"
    os.system(command)
    command="mv tempfileL2.temp "+tempfileL2
    os.system(command)

L3=[]
L3.extend(L1codes3)
L3.extend(L2codes3)
languages3=set(L3)


TOKENIZE_LANG={}
for language in languages3:
    TOKENIZE_LANG[language]=False
for i in range(0,len(corpora)):
    if TOKENIZE_L1[i]:
        TOKENIZE_LANG[L1codes3[i]]=True
for i in range(0,len(corpora)):
    if TOKENIZE_L2[i]:
        TOKENIZE_LANG[L2codes3[i]]=True
        
TRUECASE_LANG={}
for language in languages3:
    TRUECASE_LANG[language]=False
for i in range(0,len(corpora)):
    if TRUECASE_L1[i]:
        TRUECASE_LANG[L1codes3[i]]=True
for i in range(0,len(corpora)):
    if TRUECASE_L2[i]:
        TRUECASE_LANG[L2codes3[i]]=True
L2=[]
L2.extend(L1codes2)
L2.extend(L2codes2)
languages2=set(L2)
L23={}
L32={}
for i in range(0,len(corpora)):
    L23[L1codes2[i]]=L1codes3[i]
    L23[L2codes2[i]]=L2codes3[i]
    L32[L1codes3[i]]=L1codes2[i]
    L32[L2codes3[i]]=L2codes2[i]

alreadytrainedtruecaser=[]
alreadycreatedtruecaser=[]
alreadycreatedtokenizers=[]
truecasers={}
tokenizers={}
for i in range(0,len(corpora)):
    corpus=corpora[i] 
    to_tag=to_tags[i]
    L1code3=L1codes3[i]
    L1code2=L1codes2[i]
    L2code3=L2codes3[i]
    L2code2=L2codes2[i]      
    L1_TC_MODEL=L1_TC_MODELS[i]
    L2_TC_MODEL=L2_TC_MODELS[i]
    if L1_TC_MODEL=="auto":
        L1_TC_MODEL="tc."+L1code2
    if L2_TC_MODEL=="auto":
            L2_TC_MODEL="tc."+L2code2
    if TRAIN_L1_TRUECASER[i]=="True" and not L1code2 in alreadytrainedtruecaser:
        if VERBOSE:
            cadena="Training "+L1codes2[i]+" Truecaser: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        corpusfile="corpus-"+L1code3+".temp"
        SLTrainer=TC_Trainer(MTUOC, L1_TC_MODEL, corpusfile, L1_DICTS[i], L1_TOKENIZERS[i])
        SLTrainer.train_truecaser()
        alreadytrainedtruecaser.append(L1codes2[i])
        
    if TRAIN_L2_TRUECASER[i]=="True" and not L2code2 in alreadytrainedtruecaser:
        if VERBOSE:
            cadena="Training "+L2codes2[i]+" Truecaser: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        corpusfile="corpus-"+L2code3+".temp"
        SLTrainer=TC_Trainer(MTUOC, L2_TC_MODEL, corpusfile, L2_DICTS[i], L2_TOKENIZERS[i])
        SLTrainer.train_truecaser()
        alreadytrainedtruecaser.append(L2codes2[i])
    if TRUECASE_L1[i] and not L1code3 in alreadycreatedtruecaser:
        truecaser=Truecaser()
        truecaser.set_MTUOCPath(MTUOC)
        truecaser.set_tokenizer(L1_TOKENIZERS[i])
        truecaser.set_tc_model(L1_TC_MODEL)
        truecasers[L1code3]=truecaser
        alreadycreatedtruecaser.append(L1code3)
    if TRUECASE_L2[i] and not L2code3 in alreadycreatedtruecaser:
        truecaser=Truecaser()
        truecaser.set_MTUOCPath(MTUOC)
        truecaser.set_tokenizer(L2_TOKENIZERS[i])
        truecaser.set_tc_model(L2_TC_MODEL)
        truecasers[L2code3]=truecaser
        alreadycreatedtruecaser.append(L2code3)
    
    if not L1_TOKENIZERS[i]==None and not L1code3 in alreadycreatedtokenizers:
        L1_TOKENIZER=MTUOC+"/"+L1_TOKENIZERS[i]
        if not L1_TOKENIZER.endswith(".py"): L1_TOKENIZER=L1_TOKENIZER+".py"
        spec = importlib.util.spec_from_file_location('', L1_TOKENIZER)
        tokenizerL1mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizerL1mod)
        tokenizerL1=tokenizerL1mod.Tokenizer()
        tokenizers[L1code3]=tokenizerL1
        alreadycreatedtokenizers.append(L1code3)
    if not L2_TOKENIZERS[i]==None and not L2code3 in alreadycreatedtokenizers:
        L2_TOKENIZER=MTUOC+"/"+L2_TOKENIZERS[i]
        if not L2_TOKENIZER.endswith(".py"): L2_TOKENIZER=L2_TOKENIZER+".py"
        spec = importlib.util.spec_from_file_location('', L2_TOKENIZER)
        tokenizerL2mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizerL2mod)
        tokenizerL2=tokenizerL2mod.Tokenizer()
        tokenizers[L2code3]=tokenizerL2
        alreadycreatedtokenizers.append(L2code3)    
        
        
if VERBOSE:
    cadena="Preprocessing corpora: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")


toSPTrainCorpora=[]
toSPValCorpora=[]
toSPTraintotag=[]
toSPValtotag=[]

for i in range(0,len(corpora)):
    corpus=corpora[i] 
    to_tag=to_tags[i]
    L1code3=L1codes3[i]
    L1code2=L1codes2[i]
    L2code3=L2codes3[i]
    L2code2=L2codes2[i]    
    valsize=int(valsizes[i])
    evalsize=int(evalsizes[i])
    L1_TC_MODEL=L1_TC_MODELS[i]
    L2_TC_MODEL=L2_TC_MODELS[i]
    
    #SPLITTING CORPUS
    if VERBOSE:
        cadena="Splitting corpus "+corpus+": "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    split_corpus(corpus,valsize,evalsize,L1code3,L2code3)

    if VERBOSE:
        cadena="Preprocessing train corpus "+corpus+": "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    
    trainCorpus="train-"+L1code3+"-"+L2code3+".txt"
    trainPreCorpus="train-pre-"+L1code3+"-"+L2code3+".txt"
    toSPTrainCorpora.append(trainPreCorpus)
    toSPTraintotag.append(to_tag)
    entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
    sortida=codecs.open(trainPreCorpus,"w",encoding="utf-8")

    for linia in entrada:
        toWrite=True
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            l1=camps[0]
            l2=camps[1]
            #if len(camps)>=3:
            #    weight=camps[2]
            #else:
            #    weight=None
            lensl=len(l1)
            lentl=len(l2)
            if TOKENIZE_L1[i]:
                toksl=tokenizers[L1code3].tokenize(l1)
            else:
                toksl=l1
            if TOKENIZE_L2[i]:
                toktl=tokenizers[L2code3].tokenize(l2)
            else:
                toktl=l2
            lentoksl=len(toksl.split(" "))
            lentoktl=len(toktl.split(" "))
            if CLEAN and lensl<MIN_CHAR: toWrite=False
            if CLEAN and lentl<MIN_CHAR: toWrite=False
            if CLEAN and lensl>MAX_CHAR: toWrite=False
            if CLEAN and lentl>MAX_CHAR: toWrite=False
            
            if CLEAN and lentoksl<MIN_TOK: toWrite=False
            if CLEAN and lentoktl<MIN_TOK: toWrite=False
            if CLEAN and lentoksl>MAX_TOK: toWrite=False
            if CLEAN and lentoktl>MAX_TOK: toWrite=False
            if toWrite:
                if REPLACE_EMAILS:
                    toksl=replace_EMAILs(toksl,EMAIL_CODE)
                    toktl=replace_EMAILs(toktl,EMAIL_CODE)
                if REPLACE_URLS:
                    toksl=replace_URLs(toksl)
                    toktl=replace_URLs(toktl)
                if TRUECASE_L1[i]=="True":
                    toksl=truecasers[L1code3].truecase(toksl)
                if TRUECASE_L2[i]=="True":
                    toktl=truecasers[L2code3].truecase(toktl)
                #if not weight==None:
                #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
                #else:
                cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
                sortida.write(cadena+"\n")
        
    entrada.close()
    sortida.close()
    
    if VERBOSE:
        cadena="Preprocessing val corpus "+corpus+": "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    
    valCorpus="val-"+L1code3+"-"+L2code3+".txt"
    valPreCorpus="val-pre-"+L1code3+"-"+L2code3+".txt"
    toSPValCorpora.append(valPreCorpus)
    toSPValtotag.append(to_tag)
    entrada=codecs.open(valCorpus,"r",encoding="utf-8")
    sortida=codecs.open(valPreCorpus,"w",encoding="utf-8")

    for linia in entrada:
        toWrite=True
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            l1=camps[0]
            l2=camps[1]
            #if len(camps)>=3:
            #    weight=camps[2]
            #else:
            #    weight=None
            lensl=len(l1)
            lentl=len(l2)
            if TOKENIZE_L1[i]=="True":
                toksl=tokenizers[L1code3].tokenize(l1)
            else:
                toksl=l1
            if TOKENIZE_L2[i]=="True":
                toktl=tokenizers[L2code3].tokenize(l2)
            else:
                toktl=l2
            lentoksl=len(toksl.split(" "))
            lentoktl=len(toktl.split(" "))
            if CLEAN and lensl<MIN_CHAR: toWrite=False
            if CLEAN and lentl<MIN_CHAR: toWrite=False
            if CLEAN and lensl>MAX_CHAR: toWrite=False
            if CLEAN and lentl>MAX_CHAR: toWrite=False
            
            if CLEAN and lentoksl<MIN_TOK: toWrite=False
            if CLEAN and lentoktl<MIN_TOK: toWrite=False
            if CLEAN and lentoksl>MAX_TOK: toWrite=False
            if CLEAN and lentoktl>MAX_TOK: toWrite=False
            if toWrite:
                if REPLACE_EMAILS:
                    toksl=replace_EMAILs(toksl,EMAIL_CODE)
                    toktl=replace_EMAILs(toktl,EMAIL_CODE)
                if REPLACE_URLS:
                    toksl=replace_URLs(toksl)
                    toktl=replace_URLs(toktl)
                if TRUECASE_L1[i]=="True":
                    toksl=truecasers[L1code3].truecase(toksl)
                if TRUECASE_L2[i]=="True":
                    toktl=truecasers[L2code3].truecase(toktl)
                #if not weight==None:
                #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
                #else:
                cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
                sortida.write(cadena+"\n")
        
    entrada.close()
    sortida.close()

#preprocessing mono corpora

for language in languages3:
    corpus="corpus-"+language+".temp"
    corpusPre="corpus-pre-"+language+".temp"
    entrada=codecs.open(corpus,"r",encoding="utf-8")
    sortida=codecs.open(corpusPre,"w",encoding="utf-8")
    for linia in entrada:
        toWrite=True
        linia=linia.rstrip()
        if TOKENIZE_LANG[language]:
            tok=tokenizers[language].tokenize(linia)
        else:
            tok=linia
        leno=len(linia)
        lentok=len(tok.split(" "))
        if CLEAN and leno<MIN_CHAR: toWrite=False
        if CLEAN and leno>MAX_CHAR: toWrite=False
        
        if CLEAN and lentok<MIN_TOK: toWrite=False
        if CLEAN and lentok>MAX_TOK: toWrite=False
        if toWrite:
            if REPLACE_EMAILS:
                tok=replace_EMAILs(tok,EMAIL_CODE)
            if REPLACE_URLS:
                tok=replace_URLs(tok)
            if TRUECASE_LANG[language]=="True":
                tok=truecasers[language].truecase(toksl)
            #if not weight==None:
            #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            #else:
            cadena=" ".join(tok.split())
            sortida.write(cadena+"\n")           
    entrada.close()
    sortida.close()

#SENTENCEPIECE

if VERBOSE:
    cadena="Start of sentencepiece process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")


if TRAIN_SENTENCEPIECE:
    if VERBOSE:
        cadena="Start of sentencepiece training: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    #We randomly take a given number of segments from the trainA.L1+trainB.L1 and the trainA.L3 and the trainB.L2 to calculate the sentencepiece model
    options=[]
    if SPLIT_DIGITS:
        options.append("--split_digits=true")
    else:
        options.append("--split_digits=false")
    if not CONTROL_SYMBOLS=="":
        options.append("--control_symbols=\""+CONTROL_SYMBOLS+"\"")
    if not USER_DEFINED_SYMBOLS=="":
        options.append("--user_defined_symbols=\""+USER_DEFINED_SYMBOLS+"\"") 
    options=" ".join(options)
    sortidatemp=codecs.open("toSP.temp","w",encoding="utf-8")
    for language in languages3:
        corpusfile="corpus-"+language+".temp"
        entradatemp=codecs.open(corpusfile,"r",encoding="utf-8")
        cont=0
        for linia in entradatemp:
            linia=linia.rstrip()
            cont+=1
            if cont>SAMPLE_SIZE:
                break
            sortidatemp.write(linia+"\n")
        entradatemp.close()
    sortidatemp.close()
    command="spm_train --input=toSP.temp --model_prefix="+SP_MODEL_PREFIX+" --model_type="+MODEL_TYPE+" --vocab_size="+str(VOCAB_SIZE)+" --character_coverage="+str(CHARACTER_COVERAGE)+" --split_digits --input_sentence_size="+str(INPUT_SENTENCE_SIZE)+" "+options
    print(command)
    os.system(command)
    for language in languages3:
        corpusfile="corpus-pre-"+language+".temp"
        outfile="vocab_file."+L32[language]
        command="spm_encode --model="+SP_MODEL_PREFIX+".model --generate_vocabulary < "+corpusfile+" > "+outfile
        os.system(command)

#ENCODING CORPORA WITH SENTENCEPIECE
if VERBOSE:
    cadena="Encoding corpora with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

SP_MODEL=SP_MODEL_PREFIX+".model"

spProcessor=spm.SentencePieceProcessor(model_file=SP_MODEL,out_type=str,add_bos=True,add_eos=True)

toFinalStep=[]
cont=0
commandL=[]
commandL.append("cat ")
for corpus in toSPTrainCorpora: 
    to_tag=toSPTraintotag[cont]
    cont+=1
    entrada=codecs.open(corpus,"r",encoding="utf-8")
    nomsortida=corpus.replace("train-pre","train-sp").replace("val-pre","val-sp")
    toFinalStep.append(nomsortida)
    commandL.append(nomsortida)
    commandL.append(" ")
    sortida=codecs.open(nomsortida,"w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        pieces1=" ".join(spProcessor.encode(camps[0]))
        pieces2=" ".join(spProcessor.encode(camps[1]))
        cadena=to_tag+" "+pieces1+"\t"+pieces2
        sortida.write(cadena+"\n")
        

command=" ".join(commandL)+" | sort | uniq | shuf > TRAIN.SP.temp"
print(command)
os.system(command)    
command="cut -f 1 TRAIN.SP.temp > train.sp.SL"
print(command)
os.system(command) 
command="cut -f 2 TRAIN.SP.temp > train.sp.TL"
print(command)
os.system(command) 

commandL=[]
commandL.append("cat ")
cont=0
for corpus in toSPValCorpora: 
    to_tag=toSPValtotag[cont]
    cont+=1
    entrada=codecs.open(corpus,"r",encoding="utf-8")
    nomsortida=corpus.replace("train-pre","train-sp").replace("val-pre","val-sp")
    toFinalStep.append(nomsortida)
    commandL.append(nomsortida)
    commandL.append(" ")
    sortida=codecs.open(nomsortida,"w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        pieces1=" ".join(spProcessor.encode(camps[0]))
        pieces2=" ".join(spProcessor.encode(camps[1]))
        cadena=to_tag+" "+pieces1+"\t"+pieces2
        sortida.write(cadena+"\n")
        

command=" ".join(commandL)+" | sort | uniq | shuf > VAL.SP.temp"
print(command)
os.system(command)    
command="cut -f 1 VAL.SP.temp > val.sp.SL"
print(command)
os.system(command) 
command="cut -f 2 VAL.SP.temp > val.sp.TL"
print(command)
os.system(command) 

if GUIDED_ALIGNMENT:
    if VERBOSE:
        cadena="Guided alignment training: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    if DELETE_EXISTING:
        FILE="train.sp.SL.TL.align" 
        if os.path.exists(FILE):
            os.remove(FILE)
    if ALIGNER=="fast_align":
        if VERBOSE:
            cadena="Fast_align: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_fast_align(MTUOC,"train.sp","train.sp","SL","TL",False,VERBOSE)
        
    elif ALIGNER=="eflomal":
        if VERBOSE:
            cadena="Eflomal: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_eflomal(MTUOC,"train.sp","train.sp","SL","TL",SPLIT_LIMIT,VERBOSE)
 
if GUIDED_ALIGNMENT_VALID:
    if VERBOSE:
        cadena="Guided alignment valid: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    if DELETE_EXISTING:
        FILE="val.sp.SL.TL.align" 
        if os.path.exists(FILE):
            os.remove(FILE)
    if ALIGNER=="fast_align":
        if VERBOSE:
            cadena="Fast_align: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_fast_align(MTUOC,"val.sp","val.sp","SL","TL",False,VERBOSE)
    elif ALIGNER=="eflomal":
        if VERBOSE:
            cadena="Eflomal: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_eflomal(MTUOC,"val.sp","val.sp","SL","TL",SPLIT_LIMIT,VERBOSE)
if VERBOSE:
    cadena="End of process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")



if DELETE_TEMP:
    if VERBOSE:
        cadena="Deleting temporal files: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    for f in glob.glob("*.temp"):
        os.remove(f)
