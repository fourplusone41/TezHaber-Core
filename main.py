import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from text_summarizer import CentroidWordEmbeddingsSummarizer, GensimEmbeddingLoader, Zemberek

class DocLabelIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(doc, [self.labels_list[idx]])

class SimilarityRatio():
    def __init__(self, docs):
        self.docs = docs
        self.labels = [i for i in range(len(self.docs))]

    def calculate(self):
        documents = DocLabelIterator(self.docs, self.labels)
        model = Doc2Vec(vector_size=100, min_count=1, alpha=0.025, min_alpha=0.025)
        model.build_vocab(documents)

        # training of model
        for _ in range(10):
            #print ('iteration '+str(epoch+1))
            model.train(documents, total_examples=model.corpus_count, epochs=10)
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        #model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        for i in self.labels:
            sims = model.docvecs.most_similar(positive = i, topn = 3)
            print("### MOST SIMILAR TO {} ###".format(i))
            print(sims)
            print("\n")


if __name__ == "__main__":
    data = None
    with open('news.json') as json_file:
        data = json.load(json_file)
    
    full_articles = []
    full_links = []
    for src in data['newspapers']:
        articles = data['newspapers'][src]['articles']
        for article in articles:
            tmp = article['title'] + "\n" + article['text']
            full_articles.append(tmp)
            full_links.append(article['link'])

    comparator = SimilarityRatio(full_articles)
    comparator.calculate()

    zmbrk = Zemberek()
    lines = zmbrk.stopwords()
    #print(lines)
    print(len(lines))

    ge_loader = GensimEmbeddingLoader("word2vec_trwiki")
    w2v_model = ge_loader.get_model()
    summarizer = CentroidWordEmbeddingsSummarizer(w2v_model, debug=True)
    summary = summarizer.summarize(full_articles[0])
    print("###############")
    print(full_articles[0])
    print("###############")
    print(summary)
