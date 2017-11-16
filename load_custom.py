import pandas as pd

'''
def custom_load(filename, vocab, extend_with=0):
    """ Loads the given benchmark dataset.

        Tokenizes the texts using the provided vocabulary, extending it with
        words from the training dataset if extend_with > 0. Splits them into
        three lists: training, validation and testing (in that order).

        Also calculates the maximum length of the texts and the
        suggested batch_size.

    # Arguments:
        path: Path to the dataset to be loaded.
        vocab: Vocabulary to be used for tokenizing texts.
        extend_with: If > 0, the vocabulary will be extended with up to
            extend_with tokens from the training set before tokenizing.

    # Returns:
        A dictionary with the following fields:
            texts: List of three lists, containing tokenized inputs for
                training, validation and testing (in that order).
            labels: List of three lists, containing labels for training,
                validation and testing (in that order).
            added: Number of tokens added to the vocabulary.
            batch_size: Batch size.
            maxlen: Maximum length of an input.
    """
    # Pre-processing dataset
    with open(path) as dataset:
        data = pickle.load(dataset)

    # Decode data
    try:
        texts = [unicode(x) for x in data['texts']]
    except UnicodeDecodeError:
        texts = [x.decode('utf-8') for x in data['texts']]

    # Extract labels
    labels = [x['label'] for x in data['info']]

    batch_size, maxlen = calculate_batchsize_maxlen(texts)

    st = SentenceTokenizer(vocab, maxlen)

    # Split up dataset. Extend the existing vocabulary with up to extend_with
    # tokens from the training dataset.
    texts, labels, added = st.split_train_val_test(texts,
                                                   labels,
                                                   [data['train_ind'],
                                                    data['val_ind'],
                                                    data['test_ind']],
                                                   extend_with=extend_with)
    return {'texts': texts,
            'labels': labels,#mannully split
            'added': added,
            'batch_size': batch_size,
            'maxlen': maxlen}

'''

train=pd.read_csv('Microblog_Trainingdata.csv')
test = pd.read_csv('Microblogs_Testdata_withscores.csv')
X_train=np.asarray(train['spans__001'])
y_train=np.asarray(train['sentiment score'])
X_test=np.asarray(test['spans'])
y_test=np.asarray(test['sentiment score'])


#tokenize use

texts=X_train+X_test
labels=y_train+y_test

batch_size, maxlen = calculate_batchsize_maxlen(texts)


vocab=xxx
st = SentenceTokenizer(vocab, maxlen)
# Split up dataset. Extend the existing vocabulary with up to extend_with
    # tokens from the training dataset.
texts, labels, added = st.split_train_val_test(texts,
                                                   labels,
                                                   [0.7,0.1,0.2],#mannually calculate
                                                   extend_with=extend_with)