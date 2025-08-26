#CUDA_VISIBLE_DEVICES=1 python make_multilingual_sys.py parallel-sentences/*/*-train.tsv.gz --dev parallel-sentences/*/*-dev.tsv.gz 
from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

import os
import logging
import gzip
import numpy as np
import sys
import zipfile
import io
from shutil import copyfile
import csv
import sys
import torch.multiprocessing as mp
import torch

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #teacher_model_name = 'paraphrase-mpnet-base-v2'  # Our monolingual teacher model, we want to convert to multiple languages
    #student_model_name = 'output/make-multilingual-large-paraphrase-mpnet-base-v2-2021-05-21_08-33-36' #  Multilingual base model we use to imitate the teacher model
    
    teacher_model_name = 'paraphrase-MiniLM-L12-v2'
    student_model_name = 'output/make-multilingual-large-paraphrase-MiniLM-L12-v2-2021-05-21_07-24-15'
    
    student_tokenizer_name = None #'xlm-roberta-base'
    
    max_seq_length = 128  # Student model max. lengths for inputs (number of word pieces)
    train_batch_size = 64  # Batch size for training
    inference_batch_size = 64  # Batch size at inference
    max_sentences_per_language = 5000*1000  # Maximum number of  parallel sentences for training
    train_max_sentence_length = 384  # Maximum length (characters) for parallel training sentences

    num_epochs = 5  # Train for x epochs
    num_warmup_steps = 10000  # Warumup steps

    num_evaluation_steps = 50000  # Evaluate performance after every xxxx steps

    output_path = "output/make-multilingual-large-" + teacher_model_name.replace("/", "_") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Write self to path
    os.makedirs(output_path, exist_ok=True)

    train_script_path = os.path.join(output_path, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    # Read passed arguments
    train_files = []
    dev_files = []
    is_dev_file = False
    for arg in sys.argv[1:]:
        if arg.lower() == '--dev':
            is_dev_file = True
        else:
            if not os.path.exists(arg):
                print("File could not be found:", arg)
                exit()

            if is_dev_file:
                dev_files.append(arg)
            else:
                train_files.append(arg)

    if len(train_files) == 0:
        print("Please pass at least some train files")
        print("python make_multilingual_sys.py file1.tsv.gz file2.tsv.gz --dev dev1.tsv.gz dev2.tsv.gz")
        exit()

    logging.info("Train files: {}".format(", ".join(train_files)))
    logging.info("Dev files: {}".format(", ".join(dev_files)))

    ######## Start the extension of the teacher model to multiple languages ########
    logging.info("Load teacher model")
    teacher_model = SentenceTransformer(teacher_model_name)
    teacher_model.to('cuda')

    logging.info("Create student model: {}".format(student_model_name))
    #word_embedding_model = models.Transformer(student_model_name, tokenizer_name_or_path=student_tokenizer_name, max_seq_length=max_seq_length)
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
    #dense = models.Dense(pooling_model.get_sentence_embedding_dimension(), 512, bias=False, activation_function=torch.nn.Identity())
    #student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    student_model = SentenceTransformer(student_model_name)

    ###### Read Parallel Sentences Dataset ######
    train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=False)
    for train_file in train_files:
        train_data.load_data(train_file, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MSELoss(model=student_model)

    #### Evaluate cross-lingual performance on different tasks #####
    mse_evaluators = []  # evaluators has a list of different evaluator classes we call periodically
    trans_evaluator = []

    for dev_file in dev_files:
        logging.info("Create evaluator for " + dev_file)
        src_sentences = []
        trg_sentences = []
        with gzip.open(dev_file, 'rt', encoding='utf8') if dev_file.endswith('.gz') else open(dev_file, encoding='utf8') as fIn:
            for line in fIn:
                splits = line.strip().split('\t')
                if splits[0] != "" and splits[1] != "":
                    src_sentences.append(splits[0])
                    trg_sentences.append(splits[1])

        # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
        dev_mse = evaluation.MSEEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), teacher_model=teacher_model, batch_size=inference_batch_size)
        mse_evaluators.append(dev_mse)

        # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
        dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, trg_sentences, name=os.path.basename(dev_file), batch_size=inference_batch_size)
        trans_evaluator.append(dev_trans_acc)

    # Open the ZIP File of STS2017-extended.zip and check for which language combinations we have STS data
    sts_data = {}
    sts_evaluators = []
    with zipfile.ZipFile("datasets/STS2017-extended.zip") as zip:
        filelist = zip.namelist()
        sts_files = []

        for filepath in filelist:
            filename = os.path.basename(filepath)
            if filename.startswith('STS'):
                sts_data[filename] = {'sentences1': [], 'sentences2': [], 'scores': []}

                fIn = zip.open(filepath)
                for line in io.TextIOWrapper(fIn, 'utf8'):
                    sent1, sent2, score = line.strip().split("\t")
                    score = float(score)
                    sts_data[filename]['sentences1'].append(sent1)
                    sts_data[filename]['sentences2'].append(sent2)
                    sts_data[filename]['scores'].append(score)

    for filename, data in sts_data.items():
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator(data['sentences1'], data['sentences2'], data['scores'], batch_size=inference_batch_size, name=filename, show_progress_bar=False)
        sts_evaluators.append(test_evaluator)

    # Train the model
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluation.SequentialEvaluator(mse_evaluators + trans_evaluator + sts_evaluators, main_score_function=lambda scores: np.mean(scores[0:len(mse_evaluators)])),
                      epochs=num_epochs,
                      warmup_steps=num_warmup_steps,
                      evaluation_steps=num_evaluation_steps,
                      output_path=output_path,
                      save_best_model=True,
                      optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}, use_amp=True,
                      checkpoint_path=output_path,
                      checkpoint_save_steps=num_evaluation_steps,
                      checkpoint_save_total_limit=3
                      )


# Script was called via:
#python make_multilingual_sys.py parallel-sentences/Europarl/Europarl-en-bg-train.tsv.gz parallel-sentences/Europarl/Europarl-en-cs-train.tsv.gz parallel-sentences/Europarl/Europarl-en-da-train.tsv.gz parallel-sentences/Europarl/Europarl-en-de-train.tsv.gz parallel-sentences/Europarl/Europarl-en-el-train.tsv.gz parallel-sentences/Europarl/Europarl-en-es-train.tsv.gz parallel-sentences/Europarl/Europarl-en-et-train.tsv.gz parallel-sentences/Europarl/Europarl-en-fi-train.tsv.gz parallel-sentences/Europarl/Europarl-en-fr-train.tsv.gz parallel-sentences/Europarl/Europarl-en-hu-train.tsv.gz parallel-sentences/Europarl/Europarl-en-it-train.tsv.gz parallel-sentences/Europarl/Europarl-en-lt-train.tsv.gz parallel-sentences/Europarl/Europarl-en-lv-train.tsv.gz parallel-sentences/Europarl/Europarl-en-nl-train.tsv.gz parallel-sentences/Europarl/Europarl-en-pl-train.tsv.gz parallel-sentences/Europarl/Europarl-en-pt-train.tsv.gz parallel-sentences/Europarl/Europarl-en-ro-train.tsv.gz parallel-sentences/Europarl/Europarl-en-sk-train.tsv.gz parallel-sentences/Europarl/Europarl-en-sl-train.tsv.gz parallel-sentences/Europarl/Europarl-en-sv-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-ar-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-bg-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-ca-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-cs-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-da-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-de-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-el-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-es-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-fa-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-fr-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-he-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-hi-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-hu-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-id-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-it-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-ko-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-mk-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-my-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-nl-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-pl-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-pt-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-ro-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-ru-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-sq-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-sr-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-sv-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-tr-train.tsv.gz parallel-sentences/GlobalVoices/GlobalVoices-en-ur-train.tsv.gz parallel-sentences/JW300/JW300-en-ar-train.tsv.gz parallel-sentences/JW300/JW300-en-bg-train.tsv.gz parallel-sentences/JW300/JW300-en-cs-train.tsv.gz parallel-sentences/JW300/JW300-en-da-train.tsv.gz parallel-sentences/JW300/JW300-en-de-train.tsv.gz parallel-sentences/JW300/JW300-en-el-train.tsv.gz parallel-sentences/JW300/JW300-en-es-train.tsv.gz parallel-sentences/JW300/JW300-en-et-train.tsv.gz parallel-sentences/JW300/JW300-en-fa-train.tsv.gz parallel-sentences/JW300/JW300-en-fi-train.tsv.gz parallel-sentences/JW300/JW300-en-fr-train.tsv.gz parallel-sentences/JW300/JW300-en-gu-train.tsv.gz parallel-sentences/JW300/JW300-en-he-train.tsv.gz parallel-sentences/JW300/JW300-en-hi-train.tsv.gz parallel-sentences/JW300/JW300-en-hr-train.tsv.gz parallel-sentences/JW300/JW300-en-hu-train.tsv.gz parallel-sentences/JW300/JW300-en-hy-train.tsv.gz parallel-sentences/JW300/JW300-en-id-train.tsv.gz parallel-sentences/JW300/JW300-en-it-train.tsv.gz parallel-sentences/JW300/JW300-en-ja-train.tsv.gz parallel-sentences/JW300/JW300-en-ka-train.tsv.gz parallel-sentences/JW300/JW300-en-ko-train.tsv.gz parallel-sentences/JW300/JW300-en-lt-train.tsv.gz parallel-sentences/JW300/JW300-en-lv-train.tsv.gz parallel-sentences/JW300/JW300-en-mk-train.tsv.gz parallel-sentences/JW300/JW300-en-mn-train.tsv.gz parallel-sentences/JW300/JW300-en-mr-train.tsv.gz parallel-sentences/JW300/JW300-en-my-train.tsv.gz parallel-sentences/JW300/JW300-en-nl-train.tsv.gz parallel-sentences/JW300/JW300-en-pl-train.tsv.gz parallel-sentences/JW300/JW300-en-pt-train.tsv.gz parallel-sentences/JW300/JW300-en-ro-train.tsv.gz parallel-sentences/JW300/JW300-en-ru-train.tsv.gz parallel-sentences/JW300/JW300-en-sk-train.tsv.gz parallel-sentences/JW300/JW300-en-sl-train.tsv.gz parallel-sentences/JW300/JW300-en-sq-train.tsv.gz parallel-sentences/JW300/JW300-en-sv-train.tsv.gz parallel-sentences/JW300/JW300-en-th-train.tsv.gz parallel-sentences/JW300/JW300-en-tr-train.tsv.gz parallel-sentences/JW300/JW300-en-uk-train.tsv.gz parallel-sentences/JW300/JW300-en-ur-train.tsv.gz parallel-sentences/JW300/JW300-en-vi-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-ar-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-cs-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-de-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-es-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-fr-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-it-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-ja-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-nl-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-pt-train.tsv.gz parallel-sentences/News-Commentary/News-Commentary-en-ru-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ar-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-bg-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ca-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-cs-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-da-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-de-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-el-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-es-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-et-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-fa-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-fi-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-fr-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-gl-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-he-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-hi-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-hr-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-hu-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-hy-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-id-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-it-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ja-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ka-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ko-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-lt-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-lv-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-mk-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ms-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-nl-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-pl-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-pt-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ro-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ru-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-sk-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-sl-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-sq-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-sr-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-sv-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-th-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-tr-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-uk-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-ur-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-vi-train.tsv.gz parallel-sentences/OpenSubtitles/OpenSubtitles-en-zh_cn-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ar-train.tsv.gz parallel-sentences/TED2020/TED2020-en-bg-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ca-train.tsv.gz parallel-sentences/TED2020/TED2020-en-cs-train.tsv.gz parallel-sentences/TED2020/TED2020-en-da-train.tsv.gz parallel-sentences/TED2020/TED2020-en-de-train.tsv.gz parallel-sentences/TED2020/TED2020-en-el-train.tsv.gz parallel-sentences/TED2020/TED2020-en-es-train.tsv.gz parallel-sentences/TED2020/TED2020-en-et-train.tsv.gz parallel-sentences/TED2020/TED2020-en-fa-train.tsv.gz parallel-sentences/TED2020/TED2020-en-fi-train.tsv.gz parallel-sentences/TED2020/TED2020-en-fr-ca-train.tsv.gz parallel-sentences/TED2020/TED2020-en-fr-train.tsv.gz parallel-sentences/TED2020/TED2020-en-gl-train.tsv.gz parallel-sentences/TED2020/TED2020-en-gu-train.tsv.gz parallel-sentences/TED2020/TED2020-en-he-train.tsv.gz parallel-sentences/TED2020/TED2020-en-hi-train.tsv.gz parallel-sentences/TED2020/TED2020-en-hr-train.tsv.gz parallel-sentences/TED2020/TED2020-en-hu-train.tsv.gz parallel-sentences/TED2020/TED2020-en-hy-train.tsv.gz parallel-sentences/TED2020/TED2020-en-id-train.tsv.gz parallel-sentences/TED2020/TED2020-en-it-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ja-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ka-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ko-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ku-train.tsv.gz parallel-sentences/TED2020/TED2020-en-lt-train.tsv.gz parallel-sentences/TED2020/TED2020-en-lv-train.tsv.gz parallel-sentences/TED2020/TED2020-en-mk-train.tsv.gz parallel-sentences/TED2020/TED2020-en-mn-train.tsv.gz parallel-sentences/TED2020/TED2020-en-mr-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ms-train.tsv.gz parallel-sentences/TED2020/TED2020-en-my-train.tsv.gz parallel-sentences/TED2020/TED2020-en-nb-train.tsv.gz parallel-sentences/TED2020/TED2020-en-nl-train.tsv.gz parallel-sentences/TED2020/TED2020-en-pl-train.tsv.gz parallel-sentences/TED2020/TED2020-en-pt-br-train.tsv.gz parallel-sentences/TED2020/TED2020-en-pt-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ro-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ru-train.tsv.gz parallel-sentences/TED2020/TED2020-en-sk-train.tsv.gz parallel-sentences/TED2020/TED2020-en-sl-train.tsv.gz parallel-sentences/TED2020/TED2020-en-sq-train.tsv.gz parallel-sentences/TED2020/TED2020-en-sr-train.tsv.gz parallel-sentences/TED2020/TED2020-en-sv-train.tsv.gz parallel-sentences/TED2020/TED2020-en-th-train.tsv.gz parallel-sentences/TED2020/TED2020-en-tr-train.tsv.gz parallel-sentences/TED2020/TED2020-en-uk-train.tsv.gz parallel-sentences/TED2020/TED2020-en-ur-train.tsv.gz parallel-sentences/TED2020/TED2020-en-vi-train.tsv.gz parallel-sentences/TED2020/TED2020-en-zh-cn-train.tsv.gz parallel-sentences/TED2020/TED2020-en-zh-tw-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eg-mr-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ar-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-bg-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ca-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-cs-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-da-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-de-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-el-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-es-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-et-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-fi-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-fr-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-gl-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-gu-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-he-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-hi-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-hr-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-hu-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-hy-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-id-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-it-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ja-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ka-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ko-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ku-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-lt-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-lv-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-nl-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ru-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-tr-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-zh-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-mkd-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-mon-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-mya-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-nob-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-pes-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-pol-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-por-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-ron-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-slk-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-slv-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-sqi-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-srp-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-swe-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-tha-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-ukr-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-urd-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-vie-train.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-zsm-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ar-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-bg-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ca-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-cs-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-da-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-de-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-el-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-es-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-et-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-fa-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-fi-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-fr-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-gl-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-he-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-hi-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-hr-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-hu-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-id-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-it-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ja-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ka-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ko-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-lt-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-mk-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-mr-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-nl-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-pl-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-pt-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ro-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ru-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sk-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sl-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sq-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sr-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sv-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-tr-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-uk-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-vi-train.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-zh-train.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/AllNLI.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/S2ORC_citation_pairs.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/SimpleWiki.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/altlex.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/coco_captions-with-guid.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/msmarco-triplets.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/quora_duplicates.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/sentence-compression.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/stackexchange_duplicate_questions.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/wiki-atomic-edits.tsv.gz /home/ukp-reimers/sbert/sentence-transformers/examples/training/paraphrases/data/single-sentences/yahoo_answers_title_question.tsv.gz --dev parallel-sentences/TED2020/TED2020-en-ar-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-bg-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ca-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-cs-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-da-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-de-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-el-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-es-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-et-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-fa-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-fi-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-fr-ca-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-fr-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-gl-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-gu-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-he-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-hi-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-hr-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-hu-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-hy-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-id-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-it-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ja-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ka-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ko-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ku-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-lt-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-lv-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-mk-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-mn-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-mr-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ms-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-my-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-nb-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-nl-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-pl-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-pt-br-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-pt-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ro-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ru-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-sk-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-sl-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-sq-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-sr-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-sv-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-th-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-tr-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-uk-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-ur-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-vi-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-zh-cn-dev.tsv.gz parallel-sentences/TED2020/TED2020-en-zh-tw-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ar-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-bg-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-cs-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-da-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-de-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-el-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-es-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-fi-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-fr-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-he-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-hu-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-it-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ja-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-mr-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-nl-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-ru-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-tr-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-en-zh-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-mkd-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-pol-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-por-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-ron-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-srp-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-swe-dev.tsv.gz parallel-sentences/Tatoeba/Tatoeba-eng-ukr-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ar-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-bg-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ca-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-cs-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-da-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-de-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-el-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-es-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-et-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-fa-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-fi-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-fr-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-gl-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-he-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-hi-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-hr-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-hu-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-id-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-it-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ja-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ka-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ko-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-lt-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-mk-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-mr-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-nl-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-pl-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-pt-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ro-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-ru-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sk-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sl-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sq-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sr-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-sv-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-tr-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-uk-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-vi-dev.tsv.gz parallel-sentences/WikiMatrix/WikiMatrix-en-zh-dev.tsv.gz