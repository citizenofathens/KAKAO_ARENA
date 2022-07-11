import os
import re
import json
import h5py
import logging
import numpy as np
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm


#RAW_DATA_DIR = "../input/raw_data" # 카카오에서 다운로드 받은 데이터의 디렉토리
RAW_DATA_DIR = "../dataset/train.csv"
PROCESSED_DATA_DIR = '../input/processed' # 전처리된 데이터가 저장될 디렉터리
VOCAB_DIR = os.path.join(PROCESSED_DATA_DIR, 'vocab') # 전처리에 사용될 사전 파일이 저장될 디렉터리


# 학습에 사용될 파일 리스트
TRAIN_FILE_LIST = [
    "train.chunk.011"
]


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level = logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
logger = get_logger()


# 문장의 특수기호 제거 함수
def remove_special_characters(sentence , lower=True):
    p = re.compile('[\!@#$%\^&\*\(\)\-\=\[\]\{\}\.,/\?~\+\'"|_:;><`┃]')

    sentence = p.sub(' ', sentence) # 패턴 객체로  sentence 내의 특수 기호를 공백문자로 치환한다
    sentence = ' '.join(sentence.split()) # sentence 내의 두개 이상 연속된 빈공백들을 하나의 공백으로 치환
    if lower:
        sentence = sentence.lower()
    return sentence

#path_list의 파일에서 col 변수에 해당하는 컬럼 값들을 가져온다
def get_column_data(path_list, div, col ):
    col_data = []
    for path in path_list:
        h = h5py.File(path, 'r')
        col_data.append(h[div][col][:])
        h.close()
    return np.concatenate(col_data)


# path_list 의 파일에서 학습에 필요한 컬럼들을 DataFrame 포맷으로 반환한다
def get_dataframe(path_list , div):


    #pids = get_column_data(path_list, div , col='pid')
    #products = get_column_data(path_list, div, col='product')
    #bcates = get_column_data(path_list, div, col='bcateid')
    #mcates = get_column_data(path_list, div , col='mcateid')
    #scates = get_column_data(path_list,  div , col='scateid')
    #dcates = get_column_data(path_list, div ,col='dcateid')

    df = pd.read_csv(path_list)
    # 바이트 열로 인코딩 상품제목과 상품ID 를 유니코드 변환한다d
    #df['discourse_id'] = df['discourse_id'].map(lambda x : x.decode('utf-8'))
    #df['discourse_text'] = df['discourse_text'].map(lambda x : x.decode('utf-8'))
    return df

# sentencepiece 모델을 학습시키는 함수이다.
def train_spm(txt_path, spm_path,
              vocab_size=32000, input_sentence_sizes =1000000):
    # input_sentence_size : 개수 만큼만 학습데이터로 사용된다.
    # vocab_size : 사전 크기
    spm.SentencePieceTrainer.Train(
        f' --input={txt_path} --model_type=bpe'
        f' --model_prefix={spm_path} --vocab_size={vocab_size}'
        f' --input_sentence_size={input_sentence_sizes}'
        f' --shuffle_input_sentence=true'
        f' --minloglevel=2'
    )


# image_feature는 데이터의 크기가 크므로 처리함수를 별도로 분리하였다
def save_column_data(input_path_list , div, col , n_img_rows , output_path):
    # img_feat를 저장할 h5 파일을 생성
    h_out = h5py.File(output_path, 'w')
    # 대회데이터의 상품개수 x 2048(img_feat 크기)로 dataset을 할당한다.
    h_out.create_dataset(col, (n_img_rows, 2048) ,dtype=np.float32)

    offset_out = 0

    # h5 포맷의 대회데이터에서 img_feat 칼럼만 읽어서 h5포맷으로 다시 저장한다.
    for in_path in tqdm(input_path_list, desc=f'{div},{col}'):
        h_in = h5py.File(in_path, 'r')
        sz = h_in[div][col].shape[0]
        h_out[col][offset_out: offset_out+sz] = h_in[div][:]
        offset_out += sz
        h_in.close()
    h_out.close()

def preprocess():
    # 파일명과 실제 파일이 위치한 디렉토리를 결합한다
    #train_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in TRAIN_FILE_LIST]
    train_path_list = RAW_DATA_DIR

    #os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    #os.makedirs(VOCAB_DIR, exist_ok=True)

    logger.info('loading ...')

    train_df = get_dataframe(train_path_list, 'train')
    # text as product
    # product 칼럼에 특수기호를 제거하는 함수를 적용한 결과를 반환한다
    train_df['discourse_text'] = train_df['discourse_text'].map(remove_special_characters)

    # product 칼러므이 상품명을 product.txt 파일명으로 저장한다.
    with open(os.path.join(VOCAB_DIR, 'discourse_text.txt'),'w',encoding='utf-8') as f:
        f.write(train_df['discourse_text'].str.cat(sep='\n'))

    # product.txt 파일로 setencepiece 모델을 학습 시킨다.
    # 학습이 완료되면 spm.model , spm.vocab 파일이 생성된다.
    logger.info('training sentencepiece model ...')
    train_spm(txt_path=os.path.join(VOCAB_DIR, 'discourse_text.txt'),
              spm_path=os.path.join(VOCAB_DIR, 'spm')) # spm 접두어

    # 센텐스피스 모델 학습이 완료되면 product.txt는 삭제
    os.remove(os.path.join(VOCAB_DIR, 'discourse_text.txt'))

    # 필요한 파일이 제대로 생성됐는지 확인
    for dirname, _, filenames in os.walk(VOCAB_DIR):
        for filename in filenames:
            logger.info(os.path.join(dirname, filename))

    logger.info('tokenizing product ...')

    # 센텐스피스 모델을 로드한다.
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(VOCAB_DIR, 'spm.model'))
    # product 칼럼의 상품명을 분절한 결과를 tokenized_productg
    train_df['tokens'] = train_df['discourse_text'].map(lambda x: " ".join(sp.EncodeAsPieces(x)))


    #columns = ['pid' , 'tokens', 'bcateid' , 'mcateid' , 'scateid' , 'dcateid']
    #train_df = train_df[columns]


    # csv 포맷으로 저장한다.
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv') , index=False)

    # logger.info('processing img_feat ...')
    # save_column_data(train_path_list, div='train' , col='img_feat', n_img_rows=len(train_df),
    #                  output_path=os.path.join(PROCESSED_DATA_DIR, 'train_img_feat.h5'))
    #
    # for dirname, _ , filenames in os.walk(PROCESSED_DATA_DIR):
    #     for filename in filenames:
    #         logger.info(os.path.join(dirname, filename))

if __name__ == '__main__':
    preprocess()



