from keras.layers import Average
from keras.models import Model

class Ensemble:
    def __init__(self, config, outputs, model_input):
        self.config = config
        target = Average()(outputs)
        self.model = Model(model_input, target, name='avg_ensemble')

    def predict(self, encoder_predict_input):
        beam_size = self.config.beam_size
        max_target_len = encoder_predict_input.shape[0]
        k_beam = [(0, [0] * max_target_len)]

        for i in range(max_target_len):
            all_hypotheses = []
            for prob, hyp in k_beam:
                train_input = np.hstack((encoder_predict_input, np.array(hyp)))
                train_input = np.expand_dims(train_input, axis=0)

                predicted = self.model.predict(train_input)
                predicted = np.squeeze(predicted, axis=0)

                new_hypotheses = predicted[i, :].argsort(axis=-1)[-beam_size:]
                for next_hyp in new_hypotheses:
                    all_hypotheses.append((
                            sum(np.log(predicted[j, hyp[j + 1]]) for j in range(i)) + np.log(predicted[i, next_hyp]),
                            list(hyp[:(i + 1)]) + [next_hyp] + ([0] * (encoder_predict_input.shape[0] - i - 1))
                        ))
            k_beam = sorted(all_hypotheses, key=lambda x: x[0])[-beam_size:]  # Sort by probability
        return k_beam[-1][1]  # Pick hypothesis with highest probability

    def evaluate(self, encoder_predict_input, decoder_predict_target):
        y_pred = np.apply_along_axis(self.predict, 1, encoder_predict_input)
        print("BLEU Score:", bleu_score(decoder_predict_target, y_pred))
        # An error in the sacrebleu library prevents multi_bleu_score from working on WMT '14 EN-DE test split
        # print("Multi-BLEU Score", multi_bleu_score(y_pred, self.config.target_vocab, self.config.dataset))
