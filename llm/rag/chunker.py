from langchain_text_splitters import RecursiveCharacterTextSplitter 

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)