# new package to add to requirement are scikit-learn, requests, pdfplumber
import os
import math
import chromadb.api
import pythoncom
import chromadb
import numpy as np
import torch
import openai
import fitz  
from docx2pdf import convert
from io import BytesIO
from PIL import Image
import json
from docx import Document as DocxDocument
from docx.shared import Inches
from transformers import CLIPProcessor, CLIPModel
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from docx import Document as DocxDocument
from docx.shared import Inches
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from openai import OpenAI
from dotenv import load_dotenv
from header_Footer_watermark import crop_fixed_header_footer_and_remove_images

load_dotenv()  # Load environment variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
#setting up the OpenAI API key 
#chroma_client = chromadb.PersistentClient(path=r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB")# can also use server local

path= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Transformers"
clip_model = CLIPModel.from_pretrained(path)
clip_processor = CLIPProcessor.from_pretrained(path)
device = "cpu"
clip_model.to(device)



class CSVProcessor:
    def __init__(self, file_path, persist_directory='persist_chroma_csv'):
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.loader = CSVLoader(file_path=self.file_path, encoding='utf-8', csv_args={'delimiter': '|'})
        self.embedding = OpenAIEmbeddings()

    def process(self, filename):
        self.subject = filename
        docs = self.loader.load()
        # vectordb = Chroma.from_documents(
        #     documents=docs,
        #     embedding=self.embedding,
        #     persist_directory=self.persist_directory
        # )
        # vectordb.persist()

class PDFProcessor:
    def __init__(self, file_path, file_extension):
        self.file_path = file_path
        self.file_extension=file_extension
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # CLIP for both text and image embeddings
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP processor for images
        self.subject = None
        self.device= "cpu"
        self.embedding = OpenCLIPEmbeddingFunction()

    def process(self, filename):
        """Process the PDF, extract text, images, and store them in Chroma."""
        self.subject = filename
        doc = fitz.open(self.file_path)
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        output_image_dir = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\extracted_images"
        output_text_file_path=fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Summary\{self.subject}{self.file_extension}.text"
        text_file_for_summary = open(output_text_file_path, 'w',encoding='utf-8')
        
        # persisting and defining the chromadb for storing the embedding through CLIP
        persist_directory= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB"
        Chroma_client = chromadb.PersistentClient(persist_directory)
        collection3=Chroma_client.get_or_create_collection(
        name='multimodel_collection_1',
        embedding_function=self.embedding,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 100
        })
        metadata=[]
        chunks=[]
        image_match_list=[[None for _ in range(5)] for _ in range(len(doc))]
        image_mapping_file=fr"{output_image_dir}\image_map_{filename}.json"
        for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text_file_for_summary.write(f"Page {page_num + 1}:\n{text}\n\n")
                chunk=self.Text_chunker(text)
                if not os.path.exists(output_image_dir):
                    os.makedirs(output_image_dir)
                images = page.get_images(full=True)
                if images:
                    for img_index, img in enumerate(images):
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]

                                image = Image.open(BytesIO(image_bytes))
                                if image.mode == "RGBA":
                                    image = image.convert("RGB")

                                image_filename = fr"{filename}_page_{page_num + 1}_img_{img_index + 1}.jpg"
                                image_path = os.path.join(output_image_dir, image_filename)
                                image.save(image_path)
                                print(image_path)
                                image_match_list[page_num][img_index]=image_path
                metadata=metadata+[ {'subject': self.subject,'type':'text','file_type': 'PDF','image_ref_num':page_num,"image_file":image_mapping_file} for i in chunk]
                chunks=chunks+chunk
        with open(image_mapping_file, 'w') as f:
            json.dump(image_match_list, f)
        ids_text=[f"{filename}_text{id+1}" for id in range(len(chunks))]
        collection3.add(
            documents=chunks,
            ids=ids_text,
            metadatas=metadata
        )
        print(output_text_file_path)    
        return output_text_file_path

    def summarize_text(self, name, text, file_type="pdf", vectordb="persist_chroma_pdf"):
        # self.summarize_summary(text)
        document = {}
        document["name"] = name
        document["summary"] = self.summarize_summary(text)
        document["file_type"] = file_type
        document["vectordb"] = vectordb
        print(document)
        return document

    # Two different chunking strategies
        # Sementic
    def Sementic_split(Text):
            text_splitter=SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
            chunks = text_splitter.split_text(Text)
            return chunks
       
        # Recursive
    def Text_chunker(self,text):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n"]
        )
        chunks = text_splitter.split_text(text)
        return chunks

# Main processing function
def process_file(file_path):
    """Process a file (CSV, PDF, etc.) by determining its type and calling the appropriate processor."""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file = os.path.splitext(file_name)
    filename = file[0]
    path_without_ext = os.path.splitext(file_path)[0]
    
    if file_extension == '.csv':
        processor = CSVProcessor(file_path)
    elif file_extension in ['.txt','.docx']:
        pythoncom.CoInitialize()
        output_path=f"{os.path.splitext(file_path)[0]}.pdf"
        convert(file_path,output_path)
        processor=PDFProcessor(output_path,file_extension)
    elif file_extension=='.pdf':
        output_file=f"{path_without_ext}_updated.pdf"
        crop_fixed_header_footer_and_remove_images(file_path,output_file)
        processor = PDFProcessor(output_file,file_extension)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    return processor.process(filename)

def get_text_from_Pdf(file_path):
        doc = fitz.open(file_path)
    
    # Initialize a variable to hold the text
        full_text = ""
    
    # Iterate over each page in the document
        for page_num in range(doc.page_count):
        # Get a page
            page = doc.load_page(page_num)
        
        # Extract text from the page
            page_text = page.get_text()
        
        # Append the extracted text to the full_text string
            full_text += page_text
        
    # Close the document
        doc.close()
    
        return full_text
