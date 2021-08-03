import streamlit as st
from fastai.tabular.all import *
from PIL import Image

path = Path()
learn_inf = load_learner(path/'production_model.pkl', cpu=True)
book_factors = learn_inf.model.i_weight.weight
# TODO: add logo for app
#image = Image.open('logo.png')
books = pd.read_csv('books.csv')

def selectbox_with_default(text, values, default, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

def get_similar_books(title, number):
    idx = learn_inf.dls.classes['original_title'].o2i[title]
    distances = nn.CosineSimilarity(dim=1)(book_factors, book_factors[idx][None])
    idx = distances.argsort(descending=True)[1:number+1]
    similar = [learn_inf.dls.classes['original_title'][i] for i in idx]
    ids = [int(books.loc[books['original_title']==str(i)]['goodreads_book_id'].values[0]) for i in similar]
    urls = [f'https://www.goodreads.com/book/show/{id}' for id in ids]
    return similar, urls


st.title('Book Recommendation')
st.subheader('A Book Recommendation App')
"Here's the [GitHub](https://github.com/jmtzt/book-recomendation) repository."

st.info("Type a book title and get similar recommendations!")
title = selectbox_with_default("Which book do you want recommendations from:",
                            books['original_title'], default='Select a book')
number = st.slider("How many similar books do you want?", 1, 20, value=10)

if(st.button("Suggest similar books")):
    similar, urls = get_similar_books(title, number)
    st.subheader('Here are your book recommendations. Enjoy!')
    for book, url in zip(similar, urls):
        st.write(f'{book}: {url}')

st.title('Developer Info')
'''
My name is João Marcelo Tozato. You can find my other projects on [my GitHub](https://github.com/jmtzt).
'''
del books
