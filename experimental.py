from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 

tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa")

# Load a PDF file
pdf_file = "docs/ChatGPT_five_priorities_for_research.pdf"

# Load the PDF file using the PDFMinerLoader
loader = PDFMinerLoader(pdf_file)
documents = loader.load()

# Preprocess the documents using the tokenizer
input_ids = []
attention_masks = []
for document in documents:
    inputs = tokenizer.encode_plus(
        document,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(inputs['input_ids'].flatten())
    attention_masks.append(inputs['attention_mask'].flatten())

# Convert the input IDs and attention masks to tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

# Create a batch of input IDs and attention masks
batch_input_ids = input_ids.unsqueeze(0)
batch_attention_masks = attention_masks.unsqueeze(0)

# Pass the batch through the model
outputs = model(batch_input_ids, attention_mask=batch_attention_masks)

# Get the predicted answers
answers = outputs.predictions

# Print the answers
print(answers)