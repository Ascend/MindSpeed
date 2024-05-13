from functools import wraps


def shared_embedding_or_output_weight_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.pre_process:
            self.language_model.embedding.word_embeddings.weight.data = \
                self.language_model.embedding.word_embeddings.weight.data.cuda()
        else:
            self.word_embeddings.weight.data = self.word_embeddings.weight.data.cuda()
        return fn(self, *args, **kwargs)
    return wrapper

