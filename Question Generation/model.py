from transformers import AutoTokenizer, TFBertModel, tf_top_k_top_p_filtering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")

text = "We can never see the dark of the moon from earth naturally [MASK]."
inputs = tokenizer.encode(text, return_tensors="tf")


def main_model(inputs):
    embedding = bert(inputs)[0]
    dense_out = tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')(embedding)
    model = tf.keras.Model(inputs=inputs, outputs=dense_out)
    return model, dense_out


# $C' = (c1, c2,... [HL], a1,a2,...ai,[HL],...ci)$

def decoder(model_output):
    filtered_next_token_logits = tf_top_k_top_p_filtering(model_output, top_k=50, top_p=1.0)
    next_token = tf.random.categorical(filtered_next_token_logits, dtype=tf.int32, num_samples=1)
    generated = tf.concat([inputs, next_token], axis=1)
    output_id = generated.numpy().tolist()[0]
    resulting_string = tokenizer.decode(output_id)
    return resulting_string, output_id


def seq2seq(model_input):
    while model_input[-1] != tokenizer.sep_token_id:
        model, model_output = main_model(model_input)
        gen_q, decoder_output = decoder(model_output, )
        model_input = decoder_output

    return gen_q, model


# $X = ([CLS], C', [SEP], q1,q2...qx`i[MASK])$

question, model = seq2seq(inputs)

optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = model.fit({
    "input_ids": inputs
}, epochs=2)

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


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)

        loss = compute_loss(y, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


from tqdm import tqdm


def decoder(i, x):
    filtered_next_token_logits = tf_top_k_top_p_filtering(x, top_k=50, top_p=1.0)
    next_token = tf.random.categorical(filtered_next_token_logits, dtype=tf.int32, num_samples=1)
    generated = tf.concat([i, next_token], axis=1)
    output_id = generated.numpy().tolist()[0]
    resulting_string = tokenizer.decode(output_id)
    return resulting_string, generated


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, y_hat


epochs = 5

x_train = tokenizer('smth')
y_train = tokenizer('somth')

loss = 0.0
for iter in tqdm(range(epochs)):

    print(f' Epoch {iter} the total loss is {loss}')
    for i, y in tqdm(enumerate(y_train)):
        print(f' context  {i + 1} loss is {loss}')

        for word in y:
            loss, y_hat = train_step(x_train, word)
            text, x_train = decoder(x_train, y_hat)
            history.append(loss.numpy().mean())

            # plt.plot(history)
            print(f' for {word} the generated q is {text}')
