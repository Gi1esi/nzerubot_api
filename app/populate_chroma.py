from app.core.pipeline import process_pdf_folder, split_docs, vector_store

docs = process_pdf_folder('../data')
all_splits = split_docs(docs)
vector_store.add_documents(all_splits)