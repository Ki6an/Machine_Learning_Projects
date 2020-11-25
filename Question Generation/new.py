from tensorflow.keras import Model
from keras.layers import Dense

from tensorflow.keras import Model
from keras.layers import Dense


class QG_BERT(Model):
    def __init__(self, size, bert):
        super(QG_BERT, self).__init__()
        self.bert = bert  # todo
        self.dense = Dense(size, activation='softmax')

    def call(self, input_id):
        while True:  # works as a do-while loop
            print(f'the input inside the bert {input_id}')
            embedding = self.bert(input_id)[0][:, -1, :]
            dense_out = self.dense(embedding)
            q_string, decoder_out = decoder(input_id, dense_out)
            input_id = decoder_out
            print(f'for loop  the input id is {input_id}, and decoder output is {decoder_out} and string is {q_string}')
            if input_id.numpy()[0][-1] == tokenizer.sep_token_id:
                break
        return q_string, dense_out






